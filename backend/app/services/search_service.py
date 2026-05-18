"""复合搜索服务：多路召回 + RRF 融合 + Cross-Encoder 精排。

模块功能：
- BM25 检索（OpenSearch + IK 分词器，含权限过滤）
- Dense 向量检索（Qdrant，Pre-Filtering）
- Sparse 向量检索（Qdrant，Pre-Filtering）
- 权限 Filter 构建（Qdrant / OpenSearch）
- RRF 融合算法（k=60）
- Cross-Encoder 精排（DashScope gte-rerank-v2 优先,本地 BGE-Reranker fallback）
- 检索超时降级（单路 3 秒超时，跳过未返回路）
- 搜索结果格式化（相关性分数 0-1、来源信息、高亮片段 ≤200 字符）
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass

import httpx

from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

# Constants
RRF_K = 60
RETRIEVER_TIMEOUT = 3.0  # 单路检索器超时（秒）
TOP_K_PER_RETRIEVER = 50
RRF_CANDIDATE_LIMIT = 100
RERANK_TOP_N = 20
DEFAULT_PAGE_SIZE = 10
MAX_PAGE_SIZE = 50


# ─── DashScope (云上) Rerank 配置 ──────────────────────────────────────
#
# 优先级: DashScope rerank → 本地 BGE Cross-Encoder → bigram fallback
# 中文场景 gte-rerank-v2 实测优于 bge-reranker-base, 且不占本地内存。
# 关掉:在 .env 设 RERANK_PROVIDER=local 或 RERANK_PROVIDER=disabled
# 注意:DashScope rerank 用独立 API 端点 (services/rerank/text-rerank/text-rerank),
#       不是 OpenAI 兼容协议, 所以不走 LiteLLM 而是直连。

RERANK_PROVIDER = os.environ.get("RERANK_PROVIDER", "dashscope").strip().lower()
DASHSCOPE_RERANK_MODEL = os.environ.get(
    "DASHSCOPE_RERANK_MODEL", "gte-rerank-v2"
)
DASHSCOPE_RERANK_URL = os.environ.get(
    "DASHSCOPE_RERANK_URL",
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
)
DASHSCOPE_RERANK_TIMEOUT = float(
    os.environ.get("DASHSCOPE_RERANK_TIMEOUT", "5.0")
)
# DashScope rerank 单次接受的候选 doc 上限 (官方文档: 不超过 500)
# 我们先在客户端截断到 100, 后端的 RRF_CANDIDATE_LIMIT 也是 100。
DASHSCOPE_RERANK_MAX_DOCS = 100


async def _dashscope_rerank_scores(
    query: str, documents: list[str]
) -> list[float] | None:
    """调 DashScope rerank API 拿候选 doc 的相关性分数。

    返回 ``None`` 表示调用失败,调用方应回退到本地 cross-encoder。
    成功时返回与 ``documents`` 等长的 float list (分数已归一到 [0, 1])。
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        logger.warning("DashScope rerank skipped: DASHSCOPE_API_KEY not set")
        return None
    if not documents:
        return []

    payload = {
        "model": DASHSCOPE_RERANK_MODEL,
        "input": {
            "query": query,
            "documents": documents[:DASHSCOPE_RERANK_MAX_DOCS],
        },
        "parameters": {
            "return_documents": False,
            "top_n": len(documents[:DASHSCOPE_RERANK_MAX_DOCS]),
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=DASHSCOPE_RERANK_TIMEOUT) as client:
            resp = await client.post(
                DASHSCOPE_RERANK_URL, json=payload, headers=headers
            )
            resp.raise_for_status()
            body = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("DashScope rerank failed: %s", exc)
        return None

    # 返回结构: {"output": {"results": [{"index": 0, "relevance_score": 0.93}, ...]}}
    results = (body.get("output") or {}).get("results") or []
    if not results:
        return None

    n = len(documents[:DASHSCOPE_RERANK_MAX_DOCS])
    scores: list[float] = [0.0] * n
    for item in results:
        idx = item.get("index")
        score = item.get("relevance_score") or item.get("score")
        if idx is None or score is None:
            continue
        if 0 <= idx < n:
            scores[idx] = float(score)

    # 候选超过 100 的 (理论上不会触发, RRF_CANDIDATE_LIMIT=100), 补 0
    if len(documents) > n:
        scores.extend([0.0] * (len(documents) - n))

    logger.info(
        "DashScope rerank ok: model=%s, docs=%d, top_score=%.3f",
        DASHSCOPE_RERANK_MODEL,
        len(documents),
        max(scores) if scores else 0.0,
    )
    return scores


# ─── Cross-Encoder model singleton ────────────────────────────────────
#
# 加载 BAAI/bge-reranker-base 大约要 400MB 内存 + 5-30 秒, 不能每次搜索都加载。
# 用模块级缓存确保同一 worker 进程内只加载一次。
# 多个 Celery worker 进程会各自加载自己的副本, 在合理范围内。
#
# 并发安全:
# - main.py startup 通过 run_in_executor 后台预热 (非阻塞 startup)
# - 用户请求并发到达时: 已加载 → 直接返回; 加载中 → 立即返回 None 走 bigram fallback
#   不阻塞请求路径; 加载失败 → 永久 fallback。
import threading

_cross_encoder_instance = None
_cross_encoder_load_failed = False
_cross_encoder_loading = False
_cross_encoder_lock = threading.Lock()


def _get_cross_encoder():
    """惰性加载 CrossEncoder 单例。

    - 已加载: 直接返回 instance
    - 加载中 (其它线程预热): 返回 None,本次请求走 fallback,避免阻塞
    - 加载失败: 返回 None,后续不再尝试

    后台线程在 main.py 的 startup hook 调用本函数完成预热。
    """
    global _cross_encoder_instance, _cross_encoder_load_failed, _cross_encoder_loading

    if _cross_encoder_instance is not None:
        return _cross_encoder_instance
    if _cross_encoder_load_failed:
        return None

    # 拿不到锁说明已经有线程在加载,本次请求走 fallback
    if not _cross_encoder_lock.acquire(blocking=False):
        if not _cross_encoder_loading:
            # 极少数竞态: 锁刚释放但还没设置 instance 状态,再判断一次
            return _cross_encoder_instance
        return None

    try:
        # 双检锁: 拿到锁后再次检查避免重复加载
        if _cross_encoder_instance is not None:
            return _cross_encoder_instance
        if _cross_encoder_load_failed:
            return None

        _cross_encoder_loading = True

        try:
            from sentence_transformers import CrossEncoder

            # 用 BAAI/bge-reranker-base: 中英双语效果好, 模型 280MB
            # 首次启动会从 HuggingFace 下载, 后续走 ~/.cache/huggingface 缓存
            logger.info("Loading CrossEncoder model: BAAI/bge-reranker-base ...")
            _cross_encoder_instance = CrossEncoder("BAAI/bge-reranker-base")
            logger.info("CrossEncoder loaded successfully")
            return _cross_encoder_instance
        except ImportError:
            logger.info("sentence-transformers not installed; using fallback scoring")
            _cross_encoder_load_failed = True
            return None
        except Exception as e:
            logger.warning(f"Failed to load CrossEncoder model: {e}")
            _cross_encoder_load_failed = True
            return None
        finally:
            _cross_encoder_loading = False
    finally:
        _cross_encoder_lock.release()
HIGHLIGHT_MAX_CHARS = 200
HIGHLIGHT_MARK_OPEN = "<mark>"
HIGHLIGHT_MARK_CLOSE = "</mark>"


@dataclass
class SearchHit:
    """单路召回的原始命中。"""

    chunk_id: str
    document_id: str
    space_id: str = ""
    chunk_index: int = 0
    title_chain: str = ""
    source_file: str = ""
    content: str = ""
    score: float = 0.0
    page_number: int = 0


@dataclass
class SearchResult:
    """API 返回给前端的搜索结果。

    字段说明：
    - score: 相关性分数，固定夹紧到 [0.0, 1.0] 区间
    - source_file/title_chain/page_number/document_id/chunk_index: 来源信息
    - highlight: 高亮片段，长度 ≤ 200 字符，命中关键词会被
      ``<mark>...</mark>`` 包裹
    """

    chunk_id: str
    document_id: str
    chunk_index: int
    title_chain: str
    source_file: str
    score: float  # 0.0 - 1.0
    highlight: str  # 最多 200 字符
    page_number: int = 0


@dataclass
class SearchResponse:
    """Complete search response with pagination."""

    results: list[SearchResult]
    total: int
    page: int
    page_size: int


class SearchService:
    """Composite search engine with multi-recall, RRF fusion, and Cross-Encoder reranking."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
    ):
        """Initialize the search service.

        Args:
            embedding_service: Service for generating query embeddings.
                             If None, a default instance will be created.
        """
        self._embedding_service = embedding_service or EmbeddingService()

    # ─── Main Search Entry Point ───────────────────────────────────────

    async def search(
        self,
        query: str,
        user_id: str,
        allowed_space_ids: list[str],
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> SearchResponse:
        """Execute composite search with multi-recall, RRF fusion, and reranking.

        Args:
            query: User search query text
            user_id: Current user's ID for permission filtering
            allowed_space_ids: List of space IDs the user has access to
            page: Page number (1-based)
            page_size: Number of results per page (max 50)

        Returns:
            SearchResponse with paginated, ranked results
        """
        page_size = min(page_size, MAX_PAGE_SIZE)
        if page < 1:
            page = 1

        # If user has no accessible spaces, return empty results
        if not allowed_space_ids:
            return SearchResponse(results=[], total=0, page=page, page_size=page_size)

        # Generate query embeddings
        query_embedding = await self._embedding_service.embed_query(query)

        # Build permission filters
        qdrant_filter = self._build_qdrant_filter(user_id, allowed_space_ids)
        opensearch_filter = self._build_opensearch_filter(user_id, allowed_space_ids)

        # Multi-recall with timeout degradation
        recall_results = await self._multi_recall(
            query=query,
            dense_vector=query_embedding.dense_vector,
            sparse_indices=query_embedding.sparse_indices,
            sparse_values=query_embedding.sparse_values,
            qdrant_filter=qdrant_filter,
            opensearch_filter=opensearch_filter,
        )

        # RRF Fusion
        candidates = self._rrf_fusion(recall_results)

        # Cross-Encoder Reranking on top candidates
        reranked = await self._cross_encoder_rerank(query, candidates[:RERANK_TOP_N])

        # Merge reranked with remaining candidates
        all_results = reranked + candidates[RERANK_TOP_N:]

        # Format results
        formatted = self._format_results(all_results, query)

        # Paginate
        total = len(formatted)
        start = (page - 1) * page_size
        end = start + page_size
        paginated = formatted[start:end]

        return SearchResponse(
            results=paginated,
            total=total,
            page=page,
            page_size=page_size,
        )

    # ─── Permission Filter Construction ────────────────────────────────

    def _build_qdrant_filter(
        self, user_id: str, allowed_space_ids: list[str]
    ) -> dict:
        """Build Qdrant filter for permission-based Pre-Filtering.

        Uses payload filter on allowed_user_ids field.

        Args:
            user_id: Current user's ID
            allowed_space_ids: List of accessible space IDs

        Returns:
            Qdrant filter dict for use in search requests
        """
        return {
            "should": [
                {
                    "key": "allowed_user_ids",
                    "match": {"value": user_id},
                },
                {
                    "key": "space_id",
                    "match": {"any": allowed_space_ids},
                },
            ]
        }

    def _build_opensearch_filter(
        self, user_id: str, allowed_space_ids: list[str]
    ) -> dict:
        """Build OpenSearch filter for permission-based filtering.

        Uses bool query with should clause on allowed_user_ids and space_id.

        Args:
            user_id: Current user's ID
            allowed_space_ids: List of accessible space IDs

        Returns:
            OpenSearch filter dict for use in search requests
        """
        return {
            "bool": {
                "should": [
                    {"term": {"allowed_user_ids": user_id}},
                    {"terms": {"space_id": allowed_space_ids}},
                ],
                "minimum_should_match": 1,
            }
        }

    # ─── Multi-Recall with Timeout ─────────────────────────────────────

    async def _multi_recall(
        self,
        query: str,
        dense_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        qdrant_filter: dict,
        opensearch_filter: dict,
    ) -> list[list[SearchHit]]:
        """Execute multi-recall with 3-second timeout per retriever.

        Runs BM25, Dense, and Sparse retrievers concurrently.
        Skips any retriever that doesn't return within 3 seconds.

        Returns:
            List of result lists from each successful retriever
        """
        tasks = [
            self._bm25_recall(query, opensearch_filter),
            self._dense_recall(dense_vector, qdrant_filter),
            self._sparse_recall(sparse_indices, sparse_values, qdrant_filter),
        ]

        results: list[list[SearchHit]] = []

        # Use asyncio.gather with return_exceptions to handle timeouts
        gathered = await asyncio.gather(
            *[
                asyncio.wait_for(task, timeout=RETRIEVER_TIMEOUT)
                for task in tasks
            ],
            return_exceptions=True,
        )

        for i, result in enumerate(gathered):
            retriever_names = ["BM25", "Dense", "Sparse"]
            if isinstance(result, Exception):
                logger.warning(
                    f"{retriever_names[i]} retriever failed or timed out: {result}"
                )
                continue
            results.append(result)

        return results

    # ─── BM25 Retriever (OpenSearch) ───────────────────────────────────

    async def _bm25_recall(
        self, query: str, permission_filter: dict
    ) -> list[SearchHit]:
        """Execute BM25 retrieval via OpenSearch with IK tokenizer.

        Args:
            query: Search query text
            permission_filter: OpenSearch filter for permission control

        Returns:
            Top 50 search hits from BM25
        """
        from app.core.opensearch import INDEX_NAME, get_opensearch_client

        client = get_opensearch_client()

        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["content^2", "title_chain"],
                                "analyzer": "ik_smart",
                            }
                        }
                    ],
                    "filter": [permission_filter],
                }
            },
            "size": TOP_K_PER_RETRIEVER,
            # OpenSearch 服务端查询超时，与外层 asyncio.wait_for 形成双重保障
            "timeout": f"{int(RETRIEVER_TIMEOUT)}s",
            "_source": [
                "chunk_id",
                "document_id",
                "space_id",
                "chunk_index",
                "title_chain",
                "source_file",
                "page_number",
                "content",
            ],
            "highlight": {
                "fields": {"content": {"fragment_size": HIGHLIGHT_MAX_CHARS}},
                "pre_tags": ["<em>"],
                "post_tags": ["</em>"],
            },
        }

        # Run in executor since opensearch-py is synchronous
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.search(index=INDEX_NAME, body=search_body),
        )

        hits: list[SearchHit] = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            hits.append(
                SearchHit(
                    chunk_id=source.get("chunk_id", ""),
                    document_id=source.get("document_id", ""),
                    space_id=source.get("space_id", ""),
                    chunk_index=source.get("chunk_index", 0),
                    title_chain=source.get("title_chain", ""),
                    source_file=source.get("source_file", ""),
                    page_number=source.get("page_number", 0),
                    content=source.get("content", ""),
                    score=hit.get("_score", 0.0),
                )
            )

        return hits

    # ─── Dense Vector Retriever (Qdrant) ───────────────────────────────

    async def _dense_recall(
        self, dense_vector: list[float], permission_filter: dict
    ) -> list[SearchHit]:
        """Execute Dense vector retrieval via Qdrant with Pre-Filtering.

        Args:
            dense_vector: Query dense embedding (1024-dim)
            permission_filter: Qdrant filter for permission control

        Returns:
            Top 50 search hits from dense vector search
        """
        from qdrant_client.models import (
            NamedVector,
            SearchParams,
        )

        from app.core.qdrant import COLLECTION_NAME, get_qdrant_client

        client = get_qdrant_client()

        # Build Qdrant filter from permission dict
        qdrant_filter = self._dict_to_qdrant_filter(permission_filter)

        # Run in executor since qdrant_client is synchronous
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: client.search(
                collection_name=COLLECTION_NAME,
                query_vector=NamedVector(name="dense", vector=dense_vector),
                query_filter=qdrant_filter,
                limit=TOP_K_PER_RETRIEVER,
                with_payload=True,
                search_params=SearchParams(hnsw_ef=128, exact=False),
            ),
        )

        hits: list[SearchHit] = []
        for point in results:
            payload = point.payload or {}
            hits.append(
                SearchHit(
                    chunk_id=str(point.id),
                    document_id=payload.get("document_id", ""),
                    space_id=payload.get("space_id", ""),
                    chunk_index=payload.get("chunk_index", 0),
                    title_chain=payload.get("title_chain", ""),
                    source_file=payload.get("source_file", ""),
                    page_number=payload.get("page_number", 0),
                    content=payload.get("content", ""),
                    score=point.score,
                )
            )

        return hits

    # ─── Sparse Vector Retriever (Qdrant) ──────────────────────────────

    async def _sparse_recall(
        self,
        sparse_indices: list[int],
        sparse_values: list[float],
        permission_filter: dict,
    ) -> list[SearchHit]:
        """Execute Sparse vector retrieval via Qdrant with Pre-Filtering.

        Args:
            sparse_indices: Sparse vector indices
            sparse_values: Sparse vector values
            permission_filter: Qdrant filter for permission control

        Returns:
            Top 50 search hits from sparse vector search
        """
        from qdrant_client.models import (
            NamedSparseVector,
            SparseVector,
        )

        from app.core.qdrant import COLLECTION_NAME, get_qdrant_client

        if not sparse_indices:
            return []

        client = get_qdrant_client()

        # Build Qdrant filter from permission dict
        qdrant_filter = self._dict_to_qdrant_filter(permission_filter)

        # Run in executor since qdrant_client is synchronous
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: client.search(
                collection_name=COLLECTION_NAME,
                query_vector=NamedSparseVector(
                    name="sparse",
                    vector=SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                ),
                query_filter=qdrant_filter,
                limit=TOP_K_PER_RETRIEVER,
                with_payload=True,
            ),
        )

        hits: list[SearchHit] = []
        for point in results:
            payload = point.payload or {}
            hits.append(
                SearchHit(
                    chunk_id=str(point.id),
                    document_id=payload.get("document_id", ""),
                    space_id=payload.get("space_id", ""),
                    chunk_index=payload.get("chunk_index", 0),
                    title_chain=payload.get("title_chain", ""),
                    source_file=payload.get("source_file", ""),
                    page_number=payload.get("page_number", 0),
                    content=payload.get("content", ""),
                    score=point.score,
                )
            )

        return hits

    # ─── Qdrant Filter Helper ──────────────────────────────────────────

    def _dict_to_qdrant_filter(self, filter_dict: dict) -> "Filter":
        """Convert permission filter dict to Qdrant Filter object.

        Args:
            filter_dict: Dict with 'should' conditions

        Returns:
            Qdrant Filter object
        """
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            MatchAny,
            MatchValue,
        )

        conditions = []
        for condition in filter_dict.get("should", []):
            key = condition["key"]
            match = condition["match"]
            if "value" in match:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=match["value"]))
                )
            elif "any" in match:
                conditions.append(
                    FieldCondition(key=key, match=MatchAny(any=match["any"]))
                )

        return Filter(should=conditions)

    # ─── RRF Fusion ────────────────────────────────────────────────────

    def _rrf_fusion(
        self, recall_results: list[list[SearchHit]]
    ) -> list[SearchHit]:
        """Apply Reciprocal Rank Fusion to merge multi-recall results.

        RRF formula: score(d) = Σ 1/(k + rank_i(d)) where k=60

        Args:
            recall_results: List of result lists from each retriever

        Returns:
            Merged and deduplicated candidates sorted by RRF score (top 100)
        """
        # Calculate RRF scores
        rrf_scores: dict[str, float] = {}
        chunk_data: dict[str, SearchHit] = {}

        for results in recall_results:
            for rank, hit in enumerate(results, start=1):
                chunk_id = hit.chunk_id
                rrf_score = 1.0 / (RRF_K + rank)
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_score

                # Keep the hit data (prefer the one with higher original score)
                if chunk_id not in chunk_data or hit.score > chunk_data[chunk_id].score:
                    chunk_data[chunk_id] = hit

        # Sort by RRF score descending
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Build result list with RRF scores
        candidates: list[SearchHit] = []
        for chunk_id, rrf_score in sorted_chunks[:RRF_CANDIDATE_LIMIT]:
            hit = chunk_data[chunk_id]
            hit.score = rrf_score
            candidates.append(hit)

        return candidates

    # ─── Cross-Encoder Reranking ───────────────────────────────────────

    async def _cross_encoder_rerank(
        self, query: str, candidates: list[SearchHit]
    ) -> list[SearchHit]:
        """Rerank candidates using Cross-Encoder (BGE-Reranker).

        Uses sentence-transformers CrossEncoder for precise relevance scoring.
        Falls back to original order if reranking fails.

        Args:
            query: Original search query
            candidates: Top candidates to rerank

        Returns:
            Reranked candidates sorted by Cross-Encoder score
        """
        if not candidates:
            return candidates

        try:
            # Try to use sentence-transformers CrossEncoder
            scores = await self._compute_cross_encoder_scores(query, candidates)

            # Assign normalized scores and sort
            max_score = max(scores) if scores else 1.0
            min_score = min(scores) if scores else 0.0
            score_range = max_score - min_score if max_score != min_score else 1.0

            for i, candidate in enumerate(candidates):
                # Normalize to 0-1 range
                candidate.score = (scores[i] - min_score) / score_range

            # Sort by score descending
            candidates.sort(key=lambda x: x.score, reverse=True)

        except Exception as e:
            logger.warning(f"Cross-Encoder reranking failed, using RRF scores: {e}")
            # Normalize RRF scores to 0-1 range as fallback
            if candidates:
                max_rrf = max(c.score for c in candidates)
                if max_rrf > 0:
                    for c in candidates:
                        c.score = c.score / max_rrf

        return candidates

    async def _compute_cross_encoder_scores(
        self, query: str, candidates: list[SearchHit]
    ) -> list[float]:
        """Compute Cross-Encoder scores for query-candidate pairs.

        优先级:
        1. DashScope gte-rerank-v2 (中文最佳, 不占本地内存) — 默认
        2. 本地 sentence-transformers CrossEncoder (BAAI/bge-reranker-base)
        3. bigram heuristic fallback (无网或所有 reranker 都不可用时)

        通过 ``RERANK_PROVIDER`` 环境变量切换:
        - ``dashscope`` (默认): DashScope → bge → bigram
        - ``local``: 跳过 DashScope, 直接用 bge → bigram
        - ``disabled``: 跳过所有 cross-encoder, 直接 bigram fallback

        模型在首次调用时延迟加载 + 缓存为单例, 避免每次搜索都重新加载 400MB 权重。
        """
        if RERANK_PROVIDER == "disabled":
            return self._fallback_rerank_scores(query, candidates)

        # 1) DashScope rerank
        if RERANK_PROVIDER == "dashscope":
            try:
                docs = [c.content for c in candidates]
                ds_scores = await _dashscope_rerank_scores(query, docs)
                if ds_scores is not None:
                    return ds_scores
                # ds_scores is None → DashScope 不可用, 继续往下尝试本地模型
            except Exception as e:  # noqa: BLE001 - defensive
                logger.warning("DashScope rerank exception: %s", e)

        # 2) 本地 BGE Cross-Encoder
        try:
            cross_encoder = _get_cross_encoder()
            if cross_encoder is None:
                return self._fallback_rerank_scores(query, candidates)

            pairs = [[query, c.content] for c in candidates]

            # Run in executor since model inference is CPU-bound
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None,
                lambda: cross_encoder.predict(pairs).tolist(),
            )
            return scores
        except Exception as e:
            logger.warning(f"CrossEncoder model failed: {e}")
            return self._fallback_rerank_scores(query, candidates)

    def _fallback_rerank_scores(
        self, query: str, candidates: list[SearchHit]
    ) -> list[float]:
        """Fallback scoring when Cross-Encoder is unavailable.

        中英文混合启发式: 同时计算空格分词的 token 命中率与字符级 bigram 命中率,
        取两者平均。这样既能处理英文/拉丁词汇 (按 token 命中),
        也能处理无空格的中文 (按字符 bigram 命中)。

        Args:
            query: Search query
            candidates: Candidates to score

        Returns:
            List of heuristic relevance scores in [0.0, 1.0]
        """
        def _bigrams(text: str) -> set[str]:
            t = text.lower()
            return {t[i : i + 2] for i in range(len(t) - 1)} if len(t) >= 2 else set()

        query_lower = query.lower()
        query_tokens = set(query_lower.split())
        query_bigrams = _bigrams(query_lower)

        scores: list[float] = []
        for candidate in candidates:
            content_lower = candidate.content.lower()

            token_score = 0.0
            if query_tokens:
                content_tokens = set(content_lower.split())
                token_score = len(query_tokens & content_tokens) / len(query_tokens)

            bigram_score = 0.0
            if query_bigrams:
                content_bigrams = _bigrams(content_lower)
                bigram_score = len(query_bigrams & content_bigrams) / len(query_bigrams)

            scores.append(max(token_score, bigram_score))
        return scores

    # ─── Result Formatting ─────────────────────────────────────────────

    def _format_results(
        self, candidates: list[SearchHit], query: str
    ) -> list[SearchResult]:
        """将检索命中按 API 契约格式化。

        - ``score`` 严格夹紧到 [0.0, 1.0]，并保留 4 位小数
        - 输出来源信息：``document_id`` / ``source_file`` /
          ``title_chain`` / ``chunk_index`` / ``page_number``
        - 高亮片段长度始终 ≤ 200 字符，命中关键词被
          ``<mark>...</mark>`` 包裹

        Args:
            candidates: 排序后的候选命中
            query: 原始查询，用于高亮窗口选择

        Returns:
            可直接序列化为 API 响应的 ``SearchResult`` 列表
        """
        results: list[SearchResult] = []
        for hit in candidates:
            highlight = self._generate_highlight(hit.content, query)
            results.append(
                SearchResult(
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    chunk_index=hit.chunk_index,
                    title_chain=hit.title_chain,
                    source_file=hit.source_file,
                    page_number=hit.page_number,
                    score=self._clamp_score(hit.score),
                    highlight=highlight,
                )
            )
        return results

    @staticmethod
    def _clamp_score(score: float) -> float:
        """将分数夹紧到 [0.0, 1.0] 区间，并保留 4 位小数。

        - 任何 ``NaN`` 或非数值都会被视作 0.0
        - 超出区间的输入直接饱和到边界
        """
        try:
            value = float(score)
        except (TypeError, ValueError):
            return 0.0
        # NaN != NaN 是把 NaN 兜底成 0.0 的最简洁判断
        if value != value:
            return 0.0
        if value < 0.0:
            value = 0.0
        elif value > 1.0:
            value = 1.0
        return round(value, 4)

    def _generate_highlight(self, content: str, query: str) -> str:
        """生成不超过 200 字符的高亮片段。

        策略：
        1. 内容为空 → 返回空串
        2. 把查询拆成关键词（同时处理英文与 CJK），并在内容中找出所有出现位置
        3. 用滑动窗口选择"命中关键词最多"的 200 字符窗口
           - 多处并列时优先选择"出现命中数最多 + 起点最靠前"的窗口
        4. 在窗口内对所有命中位置加上 ``<mark>...</mark>`` 包裹（不会让最终
           可见字符数超过 200，标签本身不计入字符上限）
        5. 当查询为空或没有命中时，回退到内容前 200 字符（不加任何标签）
        """
        if not content:
            return ""

        terms = self._extract_query_terms(query)

        # 没有任何可用关键词时，回退到开头窗口
        if not terms:
            return content[:HIGHLIGHT_MAX_CHARS]

        matches = self._find_term_matches(content, terms)

        # 任何关键词都没命中时同样回退到开头
        if not matches:
            return content[:HIGHLIGHT_MAX_CHARS]

        # 选择命中最多的 200 字符窗口（围绕首个命中展开），保证窗口边界合法
        window_start = self._select_best_window(content, matches)
        window_end = window_start + HIGHLIGHT_MAX_CHARS

        # 仅保留落在窗口内的命中，并按起点排序，便于按序拼接
        window_matches = sorted(
            ((s, e) for s, e in matches if s >= window_start and e <= window_end),
            key=lambda x: x[0],
        )

        if not window_matches:
            # 极端情况下窗口边界裁掉了所有命中，回退到不加标签的窗口
            return content[window_start:window_end]

        return self._wrap_marks(content, window_start, window_end, window_matches)

    @staticmethod
    def _extract_query_terms(query: str) -> list[str]:
        """从查询中抽取关键词，兼容英文单词和 CJK 字符。

        - 英文：连续字母数字串视作一个 term
        - 中文/日文/韩文：按 2-gram 切分（更贴合 BM25/IK 的命中习惯）
        - 单字 CJK 也作为兜底 term，避免极短查询无命中

        返回：去重后保留首次出现顺序的关键词列表
        """
        if not query:
            return []
        text = query.lower()
        terms: list[str] = []
        seen: set[str] = set()

        def _add(term: str) -> None:
            if term and term not in seen:
                seen.add(term)
                terms.append(term)

        # 单遍扫描，保留 token 在原始查询中出现的顺序
        token_re = re.compile(
            r"[a-z0-9]+|[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+"
        )
        for match in token_re.finditer(text):
            token = match.group(0)
            first_char = token[0]
            if first_char.isascii():
                _add(token)
            else:
                if len(token) == 1:
                    _add(token)
                else:
                    for i in range(len(token) - 1):
                        _add(token[i : i + 2])
        return terms

    @staticmethod
    def _find_term_matches(
        content: str, terms: list[str]
    ) -> list[tuple[int, int]]:
        """在 content 中定位所有关键词命中。

        返回 ``(start, end)`` 列表（end 为闭后区，左闭右开），相互重叠的命中
        会合并成一个连续区间，便于后续高亮包裹时不出现嵌套 ``<mark>``。
        """
        if not content or not terms:
            return []

        lower = content.lower()
        spans: list[tuple[int, int]] = []
        for term in terms:
            if not term:
                continue
            start = 0
            while True:
                idx = lower.find(term, start)
                if idx == -1:
                    break
                spans.append((idx, idx + len(term)))
                start = idx + 1  # 允许重叠扫描，避免错过相邻命中

        if not spans:
            return []

        # 合并相邻/重叠区间
        spans.sort()
        merged: list[tuple[int, int]] = [spans[0]]
        for s, e in spans[1:]:
            last_s, last_e = merged[-1]
            if s <= last_e:
                merged[-1] = (last_s, max(last_e, e))
            else:
                merged.append((s, e))
        return merged

    @staticmethod
    def _select_best_window(
        content: str, matches: list[tuple[int, int]]
    ) -> int:
        """从命中区间中挑选 200 字符窗口起点。

        策略：
        - 内容本身 ≤ 200 字符时直接返回 0
        - 否则枚举每个命中作为锚点，把命中放在窗口左侧（少量前置上下文）
        - 选择"窗口内命中数最多"的起点；并列时取窗口起点最靠右的
          一个，以尽量让命中聚集在窗口右侧的密集区
        """
        if not matches:
            return 0

        if len(content) <= HIGHLIGHT_MAX_CHARS:
            return 0

        max_start = len(content) - HIGHLIGHT_MAX_CHARS
        # 留少量上下文与标签预算给最靠前的那个命中
        anchor_offset = 16
        best_start = 0
        best_count = -1

        for s, _ in matches:
            candidate_start = max(0, s - anchor_offset)
            candidate_start = min(candidate_start, max_start)
            candidate_end = candidate_start + HIGHLIGHT_MAX_CHARS

            count = sum(
                1 for ms, me in matches if ms >= candidate_start and me <= candidate_end
            )
            # ``>`` 而不是 ``>=``：命中数严格更多才更新；但当窗口起点更靠右
            # 时（覆盖密集区的概率更大），即便命中数相同也优先选择
            if count > best_count or (
                count == best_count and candidate_start > best_start
            ):
                best_count = count
                best_start = candidate_start

        return best_start

    @staticmethod
    def _wrap_marks(
        content: str,
        window_start: int,
        window_end: int,
        matches: list[tuple[int, int]],
    ) -> str:
        """在窗口内对命中包裹 ``<mark>...</mark>``，并保证总长度 ≤ 200。

        预算分配：
        - 总预算 = 200 字符
        - 优先保证所有命中（连同标签）都能完整放进预算；当命中本身超长时，
          再按需截断
        - 剩余预算分配给 gap（命中之间的非命中文本），gap 过长时保留靠近
          命中的尾部
        - 永远不会输出未闭合标签
        """
        budget = HIGHLIGHT_MAX_CHARS
        tag_overhead = len(HIGHLIGHT_MARK_OPEN) + len(HIGHLIGHT_MARK_CLOSE)
        pieces: list[str] = []
        used = 0
        cursor = window_start

        # 后续命中（含当前）所需的总最小开销
        remaining_overhead = sum(tag_overhead + (e - s) for s, e in matches)

        for s, e in matches:
            match_size = e - s
            match_overhead = tag_overhead + match_size
            # 移除当前命中开销，留作后续命中预留预算
            remaining_overhead -= match_overhead

            # 当前 gap 能用的最大预算 = 总预算 - 已用 - 当前命中开销 - 后续命中开销
            gap_budget = budget - used - match_overhead - remaining_overhead
            if gap_budget < 0:
                gap_budget = 0

            gap = content[cursor:s]
            gap_len = len(gap)
            if gap_len > gap_budget:
                # gap 过长时保留靠近命中的尾部
                gap = gap[gap_len - gap_budget :]
                gap_len = gap_budget

            pieces.append(gap)
            used += gap_len

            # 真正能放下的命中长度（受预算约束）
            available_match = budget - used - tag_overhead
            if available_match <= 0:
                # 没空间再加一对 ``<mark></mark>``，丢掉这个命中并停止
                break
            match_text = content[s:e][:available_match]
            pieces.append(HIGHLIGHT_MARK_OPEN)
            pieces.append(match_text)
            pieces.append(HIGHLIGHT_MARK_CLOSE)
            used += tag_overhead + len(match_text)
            cursor = e

            if len(match_text) < match_size:
                # 当前命中已被截断，后续命中都装不下了
                return "".join(pieces)

        # 末尾追加非命中尾段
        tail = content[cursor:window_end]
        remaining = budget - used
        if remaining > 0 and tail:
            pieces.append(tail[:remaining])
        return "".join(pieces)
