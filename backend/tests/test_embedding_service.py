"""任务 12.3：Embedding 服务（Dense 向量）单元测试。

仅覆盖 Dense 向量相关行为；Sparse 向量由任务 12.4 单独测试，已在
``tests/test_indexing.py`` 中保留。

由于本地 venv 不一定安装 ``litellm``，本测试模块通过 ``sys.modules`` 注入
一个可控的 stub，避免在 CI 上必须安装大体量依赖。
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.embedding_service import (
    DENSE_VECTOR_DIM,
    EmbeddingError,
    EmbeddingResult,
    EmbeddingService,
)


# ─── litellm stub ────────────────────────────────────────────────────


@pytest.fixture
def litellm_stub():
    """注入可控的 ``litellm`` 模块，并保证 ``aembedding`` 是 AsyncMock。

    yield 出 stub 模块，测试可通过 ``litellm_stub.aembedding`` 直接配置返回值
    或抛出异常。退出时还原 ``sys.modules``。
    """
    original = sys.modules.get("litellm")
    stub = types.ModuleType("litellm")
    stub.aembedding = AsyncMock()
    sys.modules["litellm"] = stub
    try:
        yield stub
    finally:
        if original is not None:
            sys.modules["litellm"] = original
        else:
            sys.modules.pop("litellm", None)


def _make_response(vectors: list[list[float]]) -> MagicMock:
    """构造一个 ``litellm.aembedding`` 的响应对象。"""
    response = MagicMock()
    response.data = [{"embedding": v} for v in vectors]
    return response


# ─── 配置默认值 ───────────────────────────────────────────────────────


class TestEmbeddingServiceConfiguration:
    """构造函数读取 settings 并允许显式覆盖。"""

    def test_dense_dimension_defaults_to_1024(self):
        """默认维度必须等于 Settings.EMBEDDING_DIMENSIONS（1024，与 Qdrant 对齐）。"""
        service = EmbeddingService()
        assert service.dimensions == 1024
        assert DENSE_VECTOR_DIM == 1024

    def test_constructor_reads_embedding_model_from_settings(self):
        """优先使用专用 EMBEDDING_MODEL；未配置时回退到 LITELLM_MODEL。"""
        with patch("app.services.embedding_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                EMBEDDING_MODEL="text-embedding-3-large",
                LITELLM_MODEL="gpt-4o",
                LITELLM_API_BASE="",
                LITELLM_API_KEY="",
                EMBEDDING_DIMENSIONS=1024,
                EMBEDDING_TIMEOUT=30.0,
                EMBEDDING_MAX_INPUT_CHARS=6000,
                EMBEDDING_MAX_RETRIES=2,
            )
            service = EmbeddingService()
            assert service.model == "text-embedding-3-large"

    def test_constructor_falls_back_to_litellm_model(self):
        """EMBEDDING_MODEL 为空字符串时，必须回退到 LITELLM_MODEL。"""
        with patch("app.services.embedding_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                EMBEDDING_MODEL="",
                LITELLM_MODEL="gpt-4o",
                LITELLM_API_BASE="",
                LITELLM_API_KEY="",
                EMBEDDING_DIMENSIONS=1024,
                EMBEDDING_TIMEOUT=30.0,
                EMBEDDING_MAX_INPUT_CHARS=6000,
                EMBEDDING_MAX_RETRIES=2,
            )
            service = EmbeddingService()
            assert service.model == "gpt-4o"

    def test_explicit_args_override_settings(self):
        """显式参数优先于 Settings。"""
        service = EmbeddingService(
            model="custom-embed",
            api_base="https://example.com",
            api_key="secret",
            batch_size=8,
            dimensions=512,
            timeout=5.0,
            max_input_chars=100,
            max_retries=0,
        )
        assert service.model == "custom-embed"
        assert service.api_base == "https://example.com"
        assert service.api_key == "secret"
        assert service.batch_size == 8
        assert service.dimensions == 512
        assert service.timeout == 5.0
        assert service.max_input_chars == 100
        assert service.max_retries == 0


# ─── Dense 向量基础行为 ──────────────────────────────────────────────


class TestDenseEmbeddingShape:
    """1024 维输出、批量、空输入。"""

    @pytest.mark.asyncio
    async def test_returns_1024_dim_vectors(self, litellm_stub):
        """每个向量长度必须等于 1024 维（Qdrant 维度）。"""
        litellm_stub.aembedding.return_value = _make_response(
            [[0.1] * 1024, [0.2] * 1024]
        )
        service = EmbeddingService()
        chunks = [
            {"id": "c-1", "text": "alpha"},
            {"id": "c-2", "text": "beta"},
        ]
        results = await service.embed_chunks(chunks)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, EmbeddingResult)
            assert len(r.dense_vector) == 1024

    @pytest.mark.asyncio
    async def test_pads_short_vectors_to_dimensions(self, litellm_stub):
        """API 返回不足 1024 维时，必须用 0 补齐。"""
        litellm_stub.aembedding.return_value = _make_response([[0.5] * 512])
        service = EmbeddingService()
        results = await service.embed_chunks([{"id": "c", "text": "x"}])
        v = results[0].dense_vector
        assert len(v) == 1024
        assert v[511] == 0.5
        assert v[512] == 0.0
        assert v[1023] == 0.0

    @pytest.mark.asyncio
    async def test_truncates_long_vectors_to_dimensions(self, litellm_stub):
        """API 返回多于 1024 维时，必须截断。"""
        litellm_stub.aembedding.return_value = _make_response([[0.3] * 4096])
        service = EmbeddingService()
        results = await service.embed_chunks([{"id": "c", "text": "x"}])
        assert len(results[0].dense_vector) == 1024

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_list(self, litellm_stub):
        """空 chunk 列表必须直接返回 []，不调用 LiteLLM。"""
        service = EmbeddingService()
        results = await service.embed_chunks([])
        assert results == []
        litellm_stub.aembedding.assert_not_called()

    @pytest.mark.asyncio
    async def test_preserves_chunk_ids(self, litellm_stub):
        """每个 EmbeddingResult.chunk_id 必须与输入一一对应。"""
        litellm_stub.aembedding.return_value = _make_response(
            [[0.0] * 1024, [0.0] * 1024, [0.0] * 1024]
        )
        service = EmbeddingService()
        chunks = [
            {"id": "doc1#0", "text": "a"},
            {"id": "doc1#1", "text": "b"},
            {"id": "doc1#2", "text": "c"},
        ]
        results = await service.embed_chunks(chunks)
        assert [r.chunk_id for r in results] == ["doc1#0", "doc1#1", "doc1#2"]


# ─── 批处理 ─────────────────────────────────────────────────────────


class TestDenseEmbeddingBatching:
    """按 batch_size 分批调用 LiteLLM。"""

    @pytest.mark.asyncio
    async def test_splits_into_batches_by_batch_size(self, litellm_stub):
        """5 条文本、batch_size=2 应产生 3 次 API 调用。"""
        # 每次调用根据传入数量返回等量向量。
        async def fake_embed(**kwargs):
            n = len(kwargs["input"])
            return _make_response([[0.1] * 1024 for _ in range(n)])

        litellm_stub.aembedding.side_effect = fake_embed
        service = EmbeddingService(batch_size=2)
        chunks = [{"id": f"c-{i}", "text": f"t{i}"} for i in range(5)]
        results = await service.embed_chunks(chunks)
        assert len(results) == 5
        assert litellm_stub.aembedding.call_count == 3

    @pytest.mark.asyncio
    async def test_passes_correct_model_to_litellm(self, litellm_stub):
        """LiteLLM 调用必须使用配置的 model 参数。

        EmbeddingService 会给不带 provider 前缀的 model 自动加 ``openai/``,
        以便走 LiteLLM 的 OpenAI 兼容协议路由 (适配 LiteLLM Proxy / 阿里 / OpenAI)。
        """
        litellm_stub.aembedding.return_value = _make_response([[0.0] * 1024])
        service = EmbeddingService(model="bge-large-zh", batch_size=4)
        await service.embed_chunks([{"id": "c", "text": "hi"}])
        kwargs = litellm_stub.aembedding.call_args.kwargs
        assert kwargs["model"] == "openai/bge-large-zh"
        assert kwargs["input"] == ["hi"]

    @pytest.mark.asyncio
    async def test_passes_api_base_and_key_when_set(self, litellm_stub):
        """配置了 api_base / api_key 时必须透传给 LiteLLM。"""
        litellm_stub.aembedding.return_value = _make_response([[0.0] * 1024])
        service = EmbeddingService(
            api_base="https://gateway.example.com",
            api_key="sk-test",
        )
        await service.embed_chunks([{"id": "c", "text": "hi"}])
        kwargs = litellm_stub.aembedding.call_args.kwargs
        assert kwargs["api_base"] == "https://gateway.example.com"
        assert kwargs["api_key"] == "sk-test"

    @pytest.mark.asyncio
    async def test_omits_optional_kwargs_when_blank(self, litellm_stub, monkeypatch):
        """空字符串 api_base / api_key 不应作为参数透传，避免覆盖默认。

        显式 mock settings 让所有相关字段为空,以隔离测试环境的真实 .env。
        """
        from unittest.mock import MagicMock

        mock_settings = MagicMock()
        mock_settings.LITELLM_MODEL = "gpt-4o"
        mock_settings.LITELLM_API_BASE = ""
        mock_settings.LITELLM_API_KEY = ""
        mock_settings.EMBEDDING_MODEL = ""
        mock_settings.EMBEDDING_DIMENSIONS = 1024
        mock_settings.EMBEDDING_TIMEOUT = 30.0
        mock_settings.EMBEDDING_MAX_INPUT_CHARS = 6000
        mock_settings.EMBEDDING_MAX_RETRIES = 2
        monkeypatch.setattr(
            "app.services.embedding_service.get_settings",
            lambda: mock_settings,
        )

        litellm_stub.aembedding.return_value = _make_response([[0.0] * 1024])
        service = EmbeddingService(api_base="", api_key="")
        await service.embed_chunks([{"id": "c", "text": "hi"}])
        kwargs = litellm_stub.aembedding.call_args.kwargs
        assert "api_base" not in kwargs
        assert "api_key" not in kwargs


# ─── 输入截断 ────────────────────────────────────────────────────────


class TestDenseEmbeddingInputTruncation:
    """超长文本必须截断；空字符串必须替换。"""

    @pytest.mark.asyncio
    async def test_long_text_is_truncated_before_call(self, litellm_stub):
        """超过 max_input_chars 的文本被裁剪到 max_input_chars。"""
        litellm_stub.aembedding.return_value = _make_response([[0.0] * 1024])
        service = EmbeddingService(max_input_chars=50)
        long_text = "x" * 1000
        await service.embed_chunks([{"id": "c", "text": long_text}])
        passed = litellm_stub.aembedding.call_args.kwargs["input"][0]
        assert len(passed) == 50

    @pytest.mark.asyncio
    async def test_short_text_is_unchanged(self, litellm_stub):
        """短文本必须按原样发送。"""
        litellm_stub.aembedding.return_value = _make_response([[0.0] * 1024])
        service = EmbeddingService(max_input_chars=50)
        await service.embed_chunks([{"id": "c", "text": "hello world"}])
        passed = litellm_stub.aembedding.call_args.kwargs["input"][0]
        assert passed == "hello world"

    @pytest.mark.asyncio
    async def test_empty_text_replaced_with_space(self, litellm_stub):
        """空字符串会被替换为单空格，避免被 embedding API 拒绝。"""
        litellm_stub.aembedding.return_value = _make_response([[0.0] * 1024])
        service = EmbeddingService()
        await service.embed_chunks([{"id": "c", "text": ""}])
        passed = litellm_stub.aembedding.call_args.kwargs["input"][0]
        assert passed != ""
        assert passed.strip() == ""


# ─── 失败处理 ────────────────────────────────────────────────────────


class TestDenseEmbeddingErrorHandling:
    """超时、API 错误、重试。"""

    @pytest.mark.asyncio
    async def test_api_failure_raises_embedding_error(self, litellm_stub):
        """LiteLLM 抛错时必须包装成 EmbeddingError 并暴露上下文。"""
        litellm_stub.aembedding.side_effect = RuntimeError("upstream 503")
        # max_retries=0 让失败直接到达终态，避免测试等待 backoff。
        service = EmbeddingService(max_retries=0, timeout=1.0, model="bge-m3")
        with pytest.raises(EmbeddingError) as exc_info:
            await service.embed_chunks([{"id": "c", "text": "x"}])
        message = str(exc_info.value)
        # 错误信息必须可定位：模型名、原始错误。
        assert "bge-m3" in message
        assert "upstream 503" in message
        # 原始异常作为 cause 链接。
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    @pytest.mark.asyncio
    async def test_timeout_raises_embedding_error(self, litellm_stub):
        """单批次超过 timeout 时必须抛 EmbeddingError。"""

        async def slow_embed(**kwargs):
            await asyncio.sleep(0.5)
            return _make_response([[0.0] * 1024])

        litellm_stub.aembedding.side_effect = slow_embed
        service = EmbeddingService(max_retries=0, timeout=0.05)
        with pytest.raises(EmbeddingError):
            await service.embed_chunks([{"id": "c", "text": "x"}])

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self, litellm_stub, monkeypatch):
        """前两次失败、第三次成功必须最终返回向量。"""
        # 让 backoff 不消耗实际墙钟时间。
        monkeypatch.setattr(
            "app.services.embedding_service.asyncio.sleep",
            AsyncMock(),
        )
        calls = {"n": 0}

        async def flaky(**kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient")
            return _make_response([[0.7] * 1024])

        litellm_stub.aembedding.side_effect = flaky
        service = EmbeddingService(max_retries=2, timeout=1.0)
        results = await service.embed_chunks([{"id": "c", "text": "x"}])
        assert calls["n"] == 3
        assert len(results) == 1
        assert len(results[0].dense_vector) == 1024
        assert results[0].dense_vector[0] == 0.7

    @pytest.mark.asyncio
    async def test_exhausts_retries_then_fails(self, litellm_stub, monkeypatch):
        """重试用尽仍失败时抛 EmbeddingError，调用次数 = max_retries+1。"""
        monkeypatch.setattr(
            "app.services.embedding_service.asyncio.sleep",
            AsyncMock(),
        )
        litellm_stub.aembedding.side_effect = RuntimeError("permanent")
        service = EmbeddingService(max_retries=2, timeout=1.0)
        with pytest.raises(EmbeddingError):
            await service.embed_chunks([{"id": "c", "text": "x"}])
        assert litellm_stub.aembedding.call_count == 3
