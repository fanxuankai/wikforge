"""任务 16.1：LLM Gateway 封装单元测试。

针对 ``app.services.llm_gateway.LLMGateway``，验证 LiteLLM 统一接口在不同
provider（OpenAI / Claude / 通义千问 / Ollama）下的行为，以及配置透传与错
误映射逻辑。

为避免 CI 强依赖 ``litellm``（包体较大且仅 LLMGateway 内部使用），本模块
通过 ``sys.modules`` 注入一个可控的 stub，在测试中精确控制 ``acompletion``
的返回值与异常。

覆盖点（对应需求 8.1）：
- 配置默认值与显式覆盖（model / api_base / api_key / timeout）
- ``complete`` 把 model / messages / api_base / api_key 完整透传给 litellm
- 不同 provider 字符串（``gpt-4o`` / ``claude-3-5-sonnet`` / ``qwen-vl-max``
  / ``ollama/llama3``）都能直通 ``acompletion`` 而无需修改业务代码
- 单次调用的 ``model`` 形参可覆盖网关默认 model
- ``complete_multimodal`` 构造图片 + 文本的多模态 messages
- ``stream`` 异步迭代 token
- 错误映射：``rate_limit`` / ``auth`` / ``model_unavailable`` / ``timeout``
  / ``unknown`` 全部归一为 ``LLMGatewayError(reason=...)``
- ``LLMResponse`` 把 usage / finish_reason / model 正确回填
"""

from __future__ import annotations

import asyncio
import base64
import sys
import types
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.llm_gateway import LLMGateway, LLMGatewayError, LLMResponse


def _expected_model(model_id: str) -> str:
    """LLMGateway 会给不带 provider 前缀的 model 自动加 ``openai/``。

    所有测试的 ``kwargs["model"]`` 期望值通过这个 helper 计算,
    避免 patch 断言里硬编码不带前缀的字符串。
    """
    return model_id if "/" in model_id else f"openai/{model_id}"


# ─── litellm stub ─────────────────────────────────────────────────────


@pytest.fixture
def litellm_stub():
    """注入可控的 ``litellm`` 模块，让测试随意配置 ``acompletion``。

    yield 出 stub，测试通过 ``litellm_stub.acompletion`` 设置返回值或异常。
    退出时还原 ``sys.modules`` 中的原值。
    """
    original = sys.modules.get("litellm")
    stub = types.ModuleType("litellm")
    stub.acompletion = AsyncMock()
    sys.modules["litellm"] = stub
    try:
        yield stub
    finally:
        if original is not None:
            sys.modules["litellm"] = original
        else:
            sys.modules.pop("litellm", None)


def _make_completion_response(
    content: str = "hello",
    *,
    model: str = "gpt-4o",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    """构造一个最小化的 ``litellm.acompletion`` 响应。"""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.model = model
    response.usage = usage
    return response


def _make_stream_response(chunks: list[str]) -> AsyncIterator[MagicMock]:
    """构造一个异步可迭代的流式响应。"""

    async def _gen():
        for token in chunks:
            chunk = MagicMock()
            delta = MagicMock()
            delta.content = token
            choice = MagicMock()
            choice.delta = delta
            chunk.choices = [choice]
            yield chunk

    return _gen()


# ─── 配置默认值 ───────────────────────────────────────────────────────


class TestLLMGatewayConfig:
    """构造函数从 Settings 读取默认值，并允许显式覆盖。"""

    @patch("app.services.llm_gateway.get_settings")
    def test_defaults_from_settings(self, mock_settings):
        """未传入参数时，使用 Settings 中的 LiteLLM 配置。"""
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            LITELLM_API_BASE="https://api.openai.com/v1",
            LITELLM_API_KEY="sk-test",
        )

        gateway = LLMGateway()

        assert gateway.model == "gpt-4o"
        assert gateway.api_base == "https://api.openai.com/v1"
        assert gateway.api_key == "sk-test"
        assert gateway.timeout == 60.0

    @patch("app.services.llm_gateway.get_settings")
    def test_explicit_args_override_settings(self, mock_settings):
        """显式参数优先于 Settings。"""
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            LITELLM_API_BASE="https://default",
            LITELLM_API_KEY="default",
        )

        gateway = LLMGateway(
            model="claude-3-5-sonnet-20241022",
            api_base="https://api.anthropic.com",
            api_key="anthropic-key",
            timeout=15.0,
        )

        assert gateway.model == "claude-3-5-sonnet-20241022"
        assert gateway.api_base == "https://api.anthropic.com"
        assert gateway.api_key == "anthropic-key"
        assert gateway.timeout == 15.0


# ─── complete：单 provider 透传 ──────────────────────────────────────


@pytest.fixture
def gateway(litellm_stub):
    """提供一个用 stub settings + stub litellm 配置好的 gateway。"""
    with patch("app.services.llm_gateway.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            LITELLM_API_BASE="https://api.example.com/v1",
            LITELLM_API_KEY="sk-test-123",
        )
        yield LLMGateway()


class TestLLMGatewayComplete:
    """``complete`` 把所有调用参数透传给 ``litellm.acompletion``。"""

    @pytest.mark.asyncio
    async def test_complete_passes_model_and_messages(
        self, gateway: LLMGateway, litellm_stub
    ):
        """model / messages / temperature / max_tokens 必须全部透传。"""
        litellm_stub.acompletion.return_value = _make_completion_response(
            content="hi there", model="gpt-4o"
        )

        response = await gateway.complete(
            prompt="What is RAG?",
            system_prompt="You are a helpful assistant.",
            temperature=0.2,
            max_tokens=512,
        )

        litellm_stub.acompletion.assert_awaited_once()
        kwargs = litellm_stub.acompletion.await_args.kwargs

        assert kwargs["model"] == _expected_model("gpt-4o")
        assert kwargs["temperature"] == 0.2
        assert kwargs["max_tokens"] == 512
        assert kwargs["timeout"] == 60.0
        # api_base / api_key 来自 Settings 并附带传入
        assert kwargs["api_base"] == "https://api.example.com/v1"
        assert kwargs["api_key"] == "sk-test-123"

        # messages 必须包含 system + user 两条
        messages = kwargs["messages"]
        assert messages == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is RAG?"},
        ]

        # 返回值结构正确
        assert isinstance(response, LLMResponse)
        assert response.content == "hi there"
        assert response.model == "gpt-4o"
        assert response.finish_reason == "stop"
        assert response.usage["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_complete_without_system_prompt(
        self, gateway: LLMGateway, litellm_stub
    ):
        """缺省 system_prompt 时，messages 仅包含 user 一条。"""
        litellm_stub.acompletion.return_value = _make_completion_response()

        await gateway.complete(prompt="hello")

        messages = litellm_stub.acompletion.await_args.kwargs["messages"]
        assert messages == [{"role": "user", "content": "hello"}]

    @pytest.mark.asyncio
    async def test_complete_per_call_model_override(
        self, gateway: LLMGateway, litellm_stub
    ):
        """单次调用的 ``model`` 可覆盖网关默认值，不修改 ``self.model``。"""
        litellm_stub.acompletion.return_value = _make_completion_response(
            model="ollama/llama3"
        )

        await gateway.complete(prompt="hi", model="ollama/llama3")

        assert litellm_stub.acompletion.await_args.kwargs["model"] == "ollama/llama3"
        # 网关默认 model 不变
        assert gateway.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_complete_omits_empty_api_base_and_key(self, litellm_stub):
        """空的 api_base / api_key 不应作为 kwargs 传给 litellm（让其用 env）。"""
        with patch("app.services.llm_gateway.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                LITELLM_MODEL="gpt-4o",
                LITELLM_API_BASE="",
                LITELLM_API_KEY="",
            )
            gw = LLMGateway()

        litellm_stub.acompletion.return_value = _make_completion_response()
        await gw.complete(prompt="hi")

        kwargs = litellm_stub.acompletion.await_args.kwargs
        assert "api_base" not in kwargs
        assert "api_key" not in kwargs


# ─── 多 provider 透传（需求 8.1 核心） ────────────────────────────────


class TestLLMGatewayProviderRouting:
    """需求 8.1：通过 model 字符串路由到不同 provider，无需修改业务代码。"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model_id",
        [
            "gpt-4o",  # OpenAI
            "claude-3-5-sonnet-20241022",  # Anthropic Claude
            "qwen-vl-max",  # 通义千问
            "ollama/llama3",  # Ollama 本地
            "ollama/minicpm-v:latest",  # Ollama 多模态
        ],
    )
    async def test_model_string_passes_through(self, litellm_stub, model_id):
        """各 provider 的 model 字符串能完整地透传给 ``litellm.acompletion``。"""
        with patch("app.services.llm_gateway.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                LITELLM_MODEL=model_id,
                LITELLM_API_BASE="",
                LITELLM_API_KEY="",
            )
            gw = LLMGateway()

        litellm_stub.acompletion.return_value = _make_completion_response(
            model=model_id
        )

        response = await gw.complete(prompt="ping")

        kwargs = litellm_stub.acompletion.await_args.kwargs
        assert kwargs["model"] == _expected_model(model_id)
        assert response.model == model_id

    @pytest.mark.asyncio
    async def test_switch_provider_via_settings_only(self, litellm_stub):
        """切换 provider 只需改 Settings，业务代码（complete 调用）保持不变。"""
        # provider A：OpenAI
        with patch("app.services.llm_gateway.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                LITELLM_MODEL="gpt-4o",
                LITELLM_API_BASE="",
                LITELLM_API_KEY="",
            )
            gw_a = LLMGateway()

        # provider B：通义千问
        with patch("app.services.llm_gateway.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                LITELLM_MODEL="qwen-max",
                LITELLM_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1",
                LITELLM_API_KEY="ali-key",
            )
            gw_b = LLMGateway()

        litellm_stub.acompletion.side_effect = [
            _make_completion_response(model="gpt-4o"),
            _make_completion_response(model="qwen-max"),
        ]

        # 完全相同的业务代码
        await gw_a.complete(prompt="hi")
        await gw_b.complete(prompt="hi")

        first_call = litellm_stub.acompletion.await_args_list[0].kwargs
        second_call = litellm_stub.acompletion.await_args_list[1].kwargs

        assert first_call["model"] == _expected_model("gpt-4o")
        assert "api_base" not in first_call

        assert second_call["model"] == _expected_model("qwen-max")
        assert second_call["api_base"] == (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        assert second_call["api_key"] == "ali-key"


# ─── 多模态 ───────────────────────────────────────────────────────────


class TestLLMGatewayMultimodal:
    """``complete_multimodal`` 把图像编码为 data URL 并构造混合 content。"""

    @pytest.mark.asyncio
    async def test_multimodal_messages_include_image_and_text(
        self, gateway: LLMGateway, litellm_stub
    ):
        """图像在前、文本在后；图像被 base64 + data URL 包装。"""
        litellm_stub.acompletion.return_value = _make_completion_response(
            content="image described", model="gpt-4o"
        )

        image_bytes = b"\x89PNG\r\n\x1a\n-fake"
        await gateway.complete_multimodal(
            prompt="Describe this",
            images=[image_bytes],
            system_prompt="You see images.",
            image_mime_type="image/png",
        )

        kwargs = litellm_stub.acompletion.await_args.kwargs
        messages = kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "You see images."}

        user_msg = messages[1]
        assert user_msg["role"] == "user"
        content = user_msg["content"]
        assert isinstance(content, list)
        assert len(content) == 2

        # 图像段
        assert content[0]["type"] == "image_url"
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        assert content[0]["image_url"]["url"] == f"data:image/png;base64,{b64}"

        # 文本段
        assert content[1] == {"type": "text", "text": "Describe this"}

    @pytest.mark.asyncio
    async def test_multimodal_per_call_model_override(
        self, gateway: LLMGateway, litellm_stub
    ):
        """多模态调用支持单次切换到 vision 模型。"""
        litellm_stub.acompletion.return_value = _make_completion_response(
            model="qwen-vl-max"
        )

        await gateway.complete_multimodal(
            prompt="ocr",
            images=[b"png-bytes"],
            model="qwen-vl-max",
        )

        assert litellm_stub.acompletion.await_args.kwargs["model"] == _expected_model("qwen-vl-max")
        # 网关默认仍为 gpt-4o
        assert gateway.model == "gpt-4o"


# ─── 流式 ─────────────────────────────────────────────────────────────


class TestLLMGatewayStream:
    """``stream`` 应异步迭代 token，并在 stream=True 下调用 LiteLLM。"""

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self, gateway: LLMGateway, litellm_stub):
        """token 应按 chunk 顺序产出，跳过空 delta。"""
        litellm_stub.acompletion.return_value = _make_stream_response(
            ["Hello", ", ", "world", "!"]
        )

        tokens: list[str] = []
        async for tok in gateway.stream(prompt="hi"):
            tokens.append(tok)

        assert tokens == ["Hello", ", ", "world", "!"]
        kwargs = litellm_stub.acompletion.await_args.kwargs
        assert kwargs["stream"] is True
        assert kwargs["model"] == _expected_model("gpt-4o")


# ─── 错误映射 ─────────────────────────────────────────────────────────


class TestLLMGatewayErrorMapping:
    """异常被归一为 ``LLMGatewayError`` 并设置正确的 ``reason``。"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raised, expected_reason",
        [
            (RuntimeError("HTTP 429 rate_limit exceeded"), "rate_limit"),
            (RuntimeError("AuthenticationError 401: invalid api key"), "auth"),
            (RuntimeError("Model not found 404"), "model_unavailable"),
            (RuntimeError("Connection timeout while reading"), "timeout"),
            (RuntimeError("some unexpected boom"), "unknown"),
        ],
    )
    async def test_error_classification(
        self,
        gateway: LLMGateway,
        litellm_stub,
        raised: Exception,
        expected_reason: str,
    ):
        """常见 LiteLLM 错误信息应映射到对应 ``reason``。"""
        litellm_stub.acompletion.side_effect = raised

        with pytest.raises(LLMGatewayError) as exc_info:
            await gateway.complete(prompt="hi")

        assert exc_info.value.reason == expected_reason

    @pytest.mark.asyncio
    async def test_asyncio_timeout_maps_to_timeout_reason(
        self, gateway: LLMGateway, litellm_stub
    ):
        """asyncio.wait_for 抛 TimeoutError → reason='timeout'。"""

        async def _hang(**_kwargs):
            await asyncio.sleep(10)
            return _make_completion_response()

        litellm_stub.acompletion.side_effect = _hang
        gateway.timeout = 0.05  # 收紧到 50ms 触发 wait_for 超时

        with pytest.raises(LLMGatewayError) as exc_info:
            await gateway.complete(prompt="hi")

        assert exc_info.value.reason == "timeout"

    @pytest.mark.asyncio
    async def test_missing_litellm_raises_gateway_error(self, gateway: LLMGateway):
        """litellm 模块未安装时，应抛 ``LLMGatewayError`` 而非 ImportError。"""
        # 移除 litellm_stub fixture 注入的模块，模拟未安装
        original = sys.modules.pop("litellm", None)
        try:
            with pytest.raises(LLMGatewayError) as exc_info:
                await gateway.complete(prompt="hi")
            assert "litellm" in str(exc_info.value).lower()
        finally:
            if original is not None:
                sys.modules["litellm"] = original

    @pytest.mark.asyncio
    async def test_stream_error_classification(
        self, gateway: LLMGateway, litellm_stub
    ):
        """流式接口的错误也应被归一为 ``LLMGatewayError``。"""
        litellm_stub.acompletion.side_effect = RuntimeError("HTTP 429 rate_limit")

        with pytest.raises(LLMGatewayError) as exc_info:
            async for _ in gateway.stream(prompt="hi"):
                pass

        assert exc_info.value.reason == "rate_limit"
