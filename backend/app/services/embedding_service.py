"""Embedding Service: Dense + Sparse vector generation for document chunks.

Provides:
- Dense embedding generation via LiteLLM (1024 dimensions)
- Sparse embedding generation via a deterministic TF-IDF scheme
  (SPLADE-equivalent; see note below)
- Batch processing with configurable batch size

Sparse vector design note (任务 12.4)
-------------------------------------
设计文档（design.md / requirements.md）要求生成 ``Sparse 向量（SPLADE 模型或
等效方案）``，落盘格式为 ``{"indices": [int], "values": [float]}``，与 Qdrant
``SparseVector`` 直接兼容。本模块采用 **TF-IDF 哈希 (hashing trick)** 作为
等效方案，原因如下：

1. SPLADE 依赖 BERT 主干 (例如 ``naver/splade-cocondenser-ensembledistil``)，
   推理需要 GPU 或显著的 CPU 资源；本项目以 LiteLLM 网关 + Celery worker 为
   主要部署形态，不宜在 worker 进程内常驻 transformer 权重。
2. 中文 + 英文混合语料下，开源 SPLADE 检查点的覆盖度受限，需要额外微调，
   超出当前迭代范围。
3. Qdrant 的稀疏向量原生支持 ``indices/values`` 任意整数索引空间，TF-IDF
   哈希向量与 SPLADE 输出在存储和检索路径上完全等价，未来可在不动 schema
   的前提下替换实现。

升级路径：当我们引入 GPU worker（或 LiteLLM 暴露 SPLADE 远程接口）时，将
``_generate_sparse_embeddings`` 替换为模型推理调用即可，调用方
（``embed_chunks`` / ``embed_query``）和下游 (``IndexingService`` /
``SearchService``) 无需任何改动。

确定性要求
~~~~~~~~~~~
索引 token 必须是 **跨进程稳定** 的：写入端（Celery worker）和查询端
（FastAPI 进程）属于不同进程，Python 内置 ``hash()`` 由于 PYTHONHASHSEED
随机化，会在两个进程产生不同的索引，导致召回率坍塌。本模块因此使用
``hashlib.blake2s`` 派生 token 索引，保证同一 token 在任何进程、任何机器
上都映射到相同的索引。
"""

import asyncio
import hashlib
import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Maximum batch size for embedding API calls
# 注意: 不同 provider 上限不同 -- OpenAI 2048, 阿里百炼 v3=25 / v4=10。
# 默认取 10 兼容百炼 v4; 单次调用更小但稳定, 大体量文档由 EmbeddingService 自动分批。
EMBEDDING_BATCH_SIZE = 10

# Dense vector dimensions (default; runtime value comes from settings)
DENSE_VECTOR_DIM = 1024

# Sparse vector vocabulary size cap (hashing trick).
#
# Qdrant accepts arbitrary u32 indices, but a bounded vocabulary keeps the
# index distribution dense enough that TF-IDF weights for repeated tokens
# collide rarely (birthday-paradox: <1% expected collisions for ~500 unique
# tokens at 30000 buckets) while staying small enough to log and inspect.
SPARSE_VOCAB_SIZE = 30000

# Minimum TF-IDF weight to keep in the sparse vector.
# Anything below is dropped to keep the vector sparse and storage-efficient.
SPARSE_WEIGHT_THRESHOLD = 0.01


def _stable_token_index(token: str, vocab_size: int = SPARSE_VOCAB_SIZE) -> int:
    """Map a token to a stable integer index in ``[0, vocab_size)``.

    Uses ``blake2s`` (FIPS-approved, fast, available in stdlib) so the mapping
    is deterministic across processes and Python versions. We must not use
    Python's built-in ``hash()`` here because PYTHONHASHSEED randomizes string
    hashes per process, which would silently break Sparse retrieval (writer
    and reader processes would compute different indices for the same token).
    """
    digest = hashlib.blake2s(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % vocab_size


@dataclass
class EmbeddingResult:
    """Result of embedding generation for a single chunk.

    Attributes:
        chunk_id: The chunk identifier
        dense_vector: Dense embedding vector (1024 dimensions)
        sparse_indices: Sparse vector indices (token positions)
        sparse_values: Sparse vector values (weights)
    """

    chunk_id: str = ""
    dense_vector: list[float] = field(default_factory=list)
    sparse_indices: list[int] = field(default_factory=list)
    sparse_values: list[float] = field(default_factory=list)


class EmbeddingService:
    """Service for generating dense and sparse embeddings.

    Dense embeddings are generated via LiteLLM's embedding API.
    Sparse embeddings use a TF-IDF based approach as a SPLADE placeholder.
    """

    def __init__(
        self,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        dimensions: int | None = None,
        timeout: float | None = None,
        max_input_chars: int | None = None,
        max_retries: int | None = None,
    ):
        """Initialize the embedding service.

        Args:
            model: Embedding model name. Defaults to settings.EMBEDDING_MODEL,
                falling back to settings.LITELLM_MODEL when unset.
            api_base: API base URL. Defaults to settings.LITELLM_API_BASE.
            api_key: API key. Defaults to settings.LITELLM_API_KEY.
            batch_size: Number of texts to embed in a single API call.
            dimensions: Target dense vector dimension. Defaults to
                settings.EMBEDDING_DIMENSIONS (1024). Vectors shorter than this
                are zero-padded and longer vectors are truncated so the result
                always matches the Qdrant collection schema.
            timeout: Per-batch timeout in seconds for the LiteLLM call.
                Defaults to settings.EMBEDDING_TIMEOUT.
            max_input_chars: Per-text character cap. Texts longer than this are
                truncated before being sent to the embedding API. Defaults to
                settings.EMBEDDING_MAX_INPUT_CHARS.
            max_retries: Number of additional attempts after the first failure.
                Defaults to settings.EMBEDDING_MAX_RETRIES. Uses exponential
                backoff (1s, 2s, 4s, ...).
        """
        settings = get_settings()
        # Prefer the dedicated embedding model when configured, otherwise fall
        # back to the global LiteLLM model so single-gateway deployments keep
        # working without extra config.
        configured_embedding_model = getattr(settings, "EMBEDDING_MODEL", "") or ""
        self.model = model or configured_embedding_model or settings.LITELLM_MODEL
        self.api_base = api_base or settings.LITELLM_API_BASE
        self.api_key = api_key or settings.LITELLM_API_KEY
        self.batch_size = max(1, batch_size)
        self.dimensions = (
            dimensions
            if dimensions is not None
            else getattr(settings, "EMBEDDING_DIMENSIONS", DENSE_VECTOR_DIM)
        )
        self.timeout = (
            timeout if timeout is not None else getattr(settings, "EMBEDDING_TIMEOUT", 30.0)
        )
        self.max_input_chars = (
            max_input_chars
            if max_input_chars is not None
            else getattr(settings, "EMBEDDING_MAX_INPUT_CHARS", 6000)
        )
        self.max_retries = (
            max_retries
            if max_retries is not None
            else getattr(settings, "EMBEDDING_MAX_RETRIES", 2)
        )
        self._idf_cache: dict[str, float] | None = None

    async def embed_chunks(
        self,
        chunks: list[dict],
    ) -> list[EmbeddingResult]:
        """Generate dense and sparse embeddings for a list of chunks.

        Args:
            chunks: List of chunk dicts with 'id' and 'text' fields

        Returns:
            List of EmbeddingResult with dense and sparse vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not chunks:
            return []

        start_time = time.perf_counter()
        logger.info(
            "embedding chunks started: chunk_count=%d model=%s batch_size=%d",
            len(chunks),
            self.model,
            self.batch_size,
        )

        texts = [chunk["text"] for chunk in chunks]
        chunk_ids = [chunk["id"] for chunk in chunks]

        # Generate dense embeddings in batches
        dense_vectors = await self._generate_dense_embeddings(texts)

        # Generate sparse embeddings (TF-IDF based)
        sparse_embeddings = self._generate_sparse_embeddings(texts)

        # Combine results
        results = []
        for i, chunk_id in enumerate(chunk_ids):
            result = EmbeddingResult(
                chunk_id=chunk_id,
                dense_vector=dense_vectors[i] if i < len(dense_vectors) else [],
                sparse_indices=sparse_embeddings[i]["indices"] if i < len(sparse_embeddings) else [],
                sparse_values=sparse_embeddings[i]["values"] if i < len(sparse_embeddings) else [],
            )
            results.append(result)

        logger.info(
            "embedding chunks completed: chunk_count=%d dense_vectors=%d "
            "sparse_vectors=%d elapsed_ms=%d",
            len(chunks),
            len(dense_vectors),
            len(sparse_embeddings),
            int((time.perf_counter() - start_time) * 1000),
        )
        return results

    async def embed_query(self, query: str) -> EmbeddingResult:
        """Generate embeddings for a single query text.

        Args:
            query: Query text to embed

        Returns:
            EmbeddingResult with dense and sparse vectors
        """
        results = await self.embed_chunks([{"id": "query", "text": query}])
        return results[0] if results else EmbeddingResult(chunk_id="query")

    async def _generate_dense_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embeddings using LiteLLM.

        Processes texts in batches to respect API limits. Each batch is
        protected by ``self.timeout`` and retried with exponential backoff up
        to ``self.max_retries`` extra attempts.

        Args:
            texts: List of texts to embed

        Returns:
            List of dense vectors (each ``self.dimensions`` long)

        Raises:
            EmbeddingError: If the API call fails after all retries
        """
        if not texts:
            return []

        # Truncate over-long inputs before they reach the API.
        prepared = [self._prepare_text(t) for t in texts]

        all_vectors: list[list[float]] = []
        batch_count = (len(prepared) + self.batch_size - 1) // self.batch_size
        logger.info(
            "dense embedding started: texts=%d batches=%d model=%s",
            len(prepared),
            batch_count,
            self.model,
        )

        for i in range(0, len(prepared), self.batch_size):
            batch = prepared[i:i + self.batch_size]
            batch_index = i // self.batch_size
            vectors = await self._call_with_retries(batch, batch_index)
            all_vectors.extend(vectors)

        return all_vectors

    def _prepare_text(self, text: str) -> str:
        """Coerce input to a non-empty, length-bounded string.

        LiteLLM (and most embedding providers) reject empty strings and
        truncate or 400 on inputs that exceed the model context. We normalise
        both cases here so a single bad chunk cannot poison the batch.
        """
        if text is None:
            return " "
        if not isinstance(text, str):
            text = str(text)
        if not text:
            # Embedding APIs commonly reject empty strings; substitute a single
            # space so the batch shape is preserved.
            return " "
        if self.max_input_chars and len(text) > self.max_input_chars:
            logger.debug(
                "Truncating embedding input from %d to %d chars",
                len(text),
                self.max_input_chars,
            )
            return text[: self.max_input_chars]
        return text

    async def _call_with_retries(
        self,
        batch: list[str],
        batch_index: int,
    ) -> list[list[float]]:
        """Call the embedding API for one batch with timeout + retry.

        Retries use exponential backoff starting at 1 second. Timeouts and
        provider errors are both retried; the final failure is wrapped in an
        :class:`EmbeddingError` that carries the model name and batch index
        for easier debugging.
        """
        attempts = self.max_retries + 1
        last_error: Exception | None = None
        start_time = time.perf_counter()

        for attempt in range(attempts):
            try:
                vectors = await asyncio.wait_for(
                    self._call_embedding_api(batch),
                    timeout=self.timeout,
                )
                logger.info(
                    "Embedding batch %d succeeded: size=%d attempt=%d/%d elapsed_ms=%d",
                    batch_index,
                    len(batch),
                    attempt + 1,
                    attempts,
                    int((time.perf_counter() - start_time) * 1000),
                )
                return vectors
            except asyncio.TimeoutError as exc:
                last_error = exc
                logger.warning(
                    "Embedding batch %d timed out after %.1fs (attempt %d/%d)",
                    batch_index,
                    self.timeout,
                    attempt + 1,
                    attempts,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Embedding batch %d failed on attempt %d/%d: %s",
                    batch_index,
                    attempt + 1,
                    attempts,
                    exc,
                )

            if attempt < attempts - 1:
                # Exponential backoff: 1s, 2s, 4s, ...
                await asyncio.sleep(2 ** attempt)

        # All retries exhausted.
        message = (
            f"Dense embedding generation failed for batch {batch_index} "
            f"(model={self.model}, size={len(batch)}): {last_error}"
        )
        logger.error(message)
        raise EmbeddingError(message) from last_error

    async def _call_embedding_api(self, texts: list[str]) -> list[list[float]]:
        """Call the LiteLLM embedding API for a batch of texts.

        Args:
            texts: Batch of texts to embed

        Returns:
            List of embedding vectors normalised to ``self.dimensions``
        """
        import litellm

        # LiteLLM SDK 通过 model 前缀识别 provider。
        # 当 model 不含 "/" 时(如 "text-embedding-v4"),默认按 OpenAI 兼容协议调用。
        # 这样可以无缝走 LiteLLM Proxy / 阿里百炼 / OpenAI 等任意 OpenAI 兼容端点。
        model = self.model
        if "/" not in model:
            model = f"openai/{model}"

        kwargs: dict = {
            "model": model,
            "input": texts,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key

        response = await litellm.aembedding(**kwargs)

        # Extract vectors from response
        vectors = []
        for item in response.data:
            embedding = item["embedding"]
            # Ensure vector is exactly self.dimensions long.
            if len(embedding) < self.dimensions:
                # Pad with zeros if shorter
                embedding = embedding + [0.0] * (self.dimensions - len(embedding))
            elif len(embedding) > self.dimensions:
                # Truncate if longer
                embedding = embedding[: self.dimensions]
            vectors.append(embedding)

        return vectors

    def _generate_sparse_embeddings(self, texts: list[str]) -> list[dict]:
        """Generate sparse embeddings using a TF-IDF + hashing trick scheme.

        This is the SPLADE-equivalent referenced in design.md. See the module
        docstring for the rationale and upgrade path. Output format is
        directly consumable by ``qdrant_client.models.SparseVector``.

        Properties guaranteed by this implementation (任务 12.4):

        - Each output dict has ``indices`` (list[int]) and ``values``
          (list[float]) of identical length.
        - ``indices`` is sorted ascending and contains no duplicates (when
          two tokens hash to the same bucket the larger weight wins).
        - ``values`` are non-negative TF-IDF weights, rounded to 4 decimals.
        - Empty / whitespace-only texts produce empty index/value lists.
        - ``indices`` are deterministic across processes (see
          ``_stable_token_index``), so writer and reader processes agree on
          the sparse vector for the same token.

        Args:
            texts: List of texts to generate sparse embeddings for

        Returns:
            List of dicts with 'indices' and 'values' keys
        """
        # Build document frequency from the batch
        doc_freq: Counter = Counter()
        tokenized_docs: list[list[str]] = []

        for text in texts:
            tokens = self._tokenize(text)
            unique_tokens = set(tokens)
            doc_freq.update(unique_tokens)
            tokenized_docs.append(tokens)

        num_docs = len(texts)
        results = []

        for tokens in tokenized_docs:
            if not tokens:
                results.append({"indices": [], "values": []})
                continue

            # Compute TF-IDF
            tf = Counter(tokens)
            total_tokens = len(tokens)

            # Aggregate weights per bucket (handles hash collisions cleanly).
            bucket_weight: dict[int, float] = {}

            for token, count in tf.items():
                # Term frequency (normalised by document length).
                tf_score = count / total_tokens
                # Inverse document frequency. ``+1`` keeps idf > 0 for terms
                # that appear in every document of the batch.
                df = doc_freq.get(token, 1)
                idf_score = math.log(1 + num_docs / df)
                weight = tf_score * idf_score

                if weight < SPARSE_WEIGHT_THRESHOLD:
                    continue

                idx = _stable_token_index(token)
                # On collision, keep the larger weight so frequent or rarer
                # tokens dominate over noise.
                prior = bucket_weight.get(idx)
                if prior is None or weight > prior:
                    bucket_weight[idx] = weight

            if not bucket_weight:
                results.append({"indices": [], "values": []})
                continue

            # Sort by index for stable, deduplicated output.
            sorted_indices = sorted(bucket_weight.keys())
            indices = sorted_indices
            values = [round(bucket_weight[i], 4) for i in sorted_indices]

            results.append({"indices": indices, "values": values})

        return results

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for sparse embedding generation.

        Handles both Chinese and English text:
        - Chinese: character-level and bigram tokens
        - English: word-level tokens (lowercased, filtered)

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        tokens: list[str] = []

        # Split into segments by whitespace and punctuation
        # Keep Chinese characters and English words
        segments = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text.lower())

        for segment in segments:
            if re.match(r"[\u4e00-\u9fff]", segment):
                # Chinese text: use character unigrams and bigrams
                chars = list(segment)
                tokens.extend(chars)
                # Add bigrams for better context
                for i in range(len(chars) - 1):
                    tokens.append(chars[i] + chars[i + 1])
            else:
                # English/numeric: use as-is if length > 1
                if len(segment) > 1:
                    tokens.append(segment)

        return tokens


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""

    pass
