"""``app.core.opensearch`` 单元测试。

任务 12.2：配置 OpenSearch 客户端和 Index 创建（chunks，IK 分词器映射）。

覆盖点：

- ``get_opensearch_client`` 是单例：多次调用返回同一实例，``OpenSearch``
  构造函数仅被调用一次，且使用 ``OPENSEARCH_HOST`` / ``OPENSEARCH_PORT``
  / ``OPENSEARCH_USER`` / ``OPENSEARCH_PASSWORD`` 配置。
- ``ensure_index_exists`` 在索引缺失时使用完整 IK 映射创建（包括
  ``ik_max_word`` analyzer 与 ``ik_smart`` search_analyzer）。
- ``ensure_index_exists`` 在索引已存在时为 no-op（幂等）。
- ``ensure_index_exists`` 在底层抛 RequestError 且错误信息为
  ``unknown analyzer`` 时降级到 ``standard`` 分词器。
- ``ensure_index_exists`` 在错误与分词器无关时向上抛出。
- 索引映射与 design.md 字段一致：``chunk_id`` / ``document_id`` /
  ``space_id`` 为 keyword，``content`` / ``title_chain`` 为 text，
  ``page_number`` / ``chunk_index`` 为 integer，``allowed_user_ids``
  为 keyword，``created_at`` 为 date。
- ``reset_opensearch_client`` 释放单例，下次调用会重新构造客户端。
- ``delete_index`` 在索引存在时调用 ``indices.delete``，不存在时静默忽略。

Validates: Requirements 4
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from opensearchpy.exceptions import RequestError

import app.core.opensearch as os_module
from app.core.opensearch import (
    INDEX_MAPPING,
    INDEX_MAPPING_FALLBACK,
    INDEX_NAME,
    delete_index,
    ensure_index_exists,
    get_opensearch_client,
    reset_opensearch_client,
)


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_singleton():
    """每个测试前后都清空模块级 OpenSearch 客户端缓存。"""
    os_module._opensearch_client = None
    yield
    os_module._opensearch_client = None


# ─── Module constants ─────────────────────────────────────────────────


class TestModuleConstants:
    """模块级常量必须与设计文档保持一致。"""

    def test_index_name_is_chunks(self):
        assert INDEX_NAME == "chunks"

    def test_index_mapping_uses_ik_analyzers(self):
        """主映射的 content 字段必须使用 ik_max_word/ik_smart。"""
        properties = INDEX_MAPPING["mappings"]["properties"]
        content = properties["content"]

        assert content["type"] == "text"
        assert content["analyzer"] == "ik_max_word"
        assert content["search_analyzer"] == "ik_smart"

    def test_index_mapping_title_chain_uses_ik_max_word(self):
        properties = INDEX_MAPPING["mappings"]["properties"]
        title_chain = properties["title_chain"]

        assert title_chain["type"] == "text"
        assert title_chain["analyzer"] == "ik_max_word"

    @pytest.mark.parametrize(
        "field_name, expected_type",
        [
            ("chunk_id", "keyword"),
            ("document_id", "keyword"),
            ("space_id", "keyword"),
            ("source_file", "keyword"),
            ("allowed_user_ids", "keyword"),
            ("page_number", "integer"),
            ("chunk_index", "integer"),
            ("created_at", "date"),
        ],
    )
    def test_index_mapping_field_types(self, field_name, expected_type):
        """每个非分析字段的类型必须严格匹配 design.md。"""
        properties = INDEX_MAPPING["mappings"]["properties"]
        assert properties[field_name]["type"] == expected_type

    def test_fallback_mapping_uses_standard_analyzer(self):
        """降级映射必须使用 OpenSearch 内置 standard 分词器，且不引用 IK。"""
        properties = INDEX_MAPPING_FALLBACK["mappings"]["properties"]
        content = properties["content"]

        assert content["type"] == "text"
        assert content["analyzer"] == "standard"
        # 降级路径下不应再设置 search_analyzer，避免引用未注册的分析器。
        assert "search_analyzer" not in content

    def test_fallback_mapping_has_same_field_set(self):
        """降级映射的字段集必须与主映射保持一致，确保下游写入兼容。"""
        main_fields = set(INDEX_MAPPING["mappings"]["properties"].keys())
        fallback_fields = set(
            INDEX_MAPPING_FALLBACK["mappings"]["properties"].keys()
        )
        assert main_fields == fallback_fields


# ─── Client singleton ──────────────────────────────────────────────────


class TestGetOpenSearchClient:
    """``get_opensearch_client`` 行为校验。"""

    @patch("app.core.opensearch.OpenSearch")
    def test_creates_client_with_host_and_port(self, mock_class):
        """构造时必须使用 settings 的 host/port。

        默认 ``OPENSEARCH_USE_SSL=False``,client 走 HTTP, 不传 http_auth (开发环境)。
        生产启用 SSL 时见下面 ``test_passes_credentials_with_ssl``。
        """
        mock_class.return_value = MagicMock()

        client = get_opensearch_client()

        assert client is mock_class.return_value
        mock_class.assert_called_once()
        kwargs = mock_class.call_args.kwargs
        assert "hosts" in kwargs
        host_entry = kwargs["hosts"][0]
        assert "host" in host_entry
        assert "port" in host_entry
        # 默认 use_ssl=False (HTTP), 此模式下不传 http_auth
        assert kwargs.get("use_ssl") is False
        # timeout 不可省略,避免请求挂死
        assert kwargs.get("timeout") is not None

    @patch("app.core.opensearch.OpenSearch")
    def test_returns_same_instance_on_repeated_calls(self, mock_class):
        """单例：多次调用复用同一客户端，构造函数只跑一次。"""
        mock_class.return_value = MagicMock()

        first = get_opensearch_client()
        second = get_opensearch_client()
        third = get_opensearch_client()

        assert first is second is third
        assert mock_class.call_count == 1

    @patch("app.core.opensearch.get_settings")
    @patch("app.core.opensearch.OpenSearch")
    def test_passes_credentials_from_settings(
        self, mock_class, mock_get_settings
    ):
        """SSL 启用时, user/password 透传到 http_auth。"""
        settings = MagicMock()
        settings.OPENSEARCH_HOST = "opensearch.example"
        settings.OPENSEARCH_PORT = 9201
        settings.OPENSEARCH_USER = "admin"
        settings.OPENSEARCH_PASSWORD = "Sup3r-Secret"
        # 显式启用 SSL 才会触发 http_auth 注入
        settings.OPENSEARCH_USE_SSL = True
        settings.OPENSEARCH_VERIFY_CERTS = False
        mock_get_settings.return_value = settings

        get_opensearch_client()

        kwargs = mock_class.call_args.kwargs
        assert kwargs["http_auth"] == ("admin", "Sup3r-Secret")
        assert kwargs["use_ssl"] is True
        assert kwargs["verify_certs"] is False
        host_entry = kwargs["hosts"][0]
        assert host_entry["host"] == "opensearch.example"
        assert host_entry["port"] == 9201


# ─── Reset helper ──────────────────────────────────────────────────────


class TestResetOpenSearchClient:
    """``reset_opensearch_client`` 必须清空单例状态。"""

    @patch("app.core.opensearch.OpenSearch")
    def test_reset_forces_new_client_on_next_call(self, mock_class):
        first_inst = MagicMock(name="first")
        second_inst = MagicMock(name="second")
        mock_class.side_effect = [first_inst, second_inst]

        first = get_opensearch_client()
        assert first is first_inst

        reset_opensearch_client()

        second = get_opensearch_client()
        assert second is second_inst
        assert mock_class.call_count == 2

    def test_reset_is_safe_when_no_client_cached(self):
        """从未调用 ``get_opensearch_client`` 时 reset 也应静默通过。"""
        os_module._opensearch_client = None
        reset_opensearch_client()  # 不应抛错
        assert os_module._opensearch_client is None


# ─── ensure_index_exists ──────────────────────────────────────────────


class TestEnsureIndexExists:
    """``ensure_index_exists`` 幂等创建 ``chunks`` 索引。"""

    @patch("app.core.opensearch.get_opensearch_client")
    def test_creates_index_when_missing(self, mock_get_client):
        client = MagicMock()
        client.indices.exists.return_value = False
        mock_get_client.return_value = client

        ensure_index_exists()

        client.indices.create.assert_called_once()

    @patch("app.core.opensearch.get_opensearch_client")
    def test_creates_index_with_correct_name(self, mock_get_client):
        client = MagicMock()
        client.indices.exists.return_value = False
        mock_get_client.return_value = client

        ensure_index_exists()

        kwargs = client.indices.create.call_args.kwargs
        assert kwargs["index"] == "chunks"

    @patch("app.core.opensearch.get_opensearch_client")
    def test_creates_index_with_full_ik_mapping(self, mock_get_client):
        """创建时必须传入完整的 IK 映射（包含所有 design.md 字段）。"""
        client = MagicMock()
        client.indices.exists.return_value = False
        mock_get_client.return_value = client

        ensure_index_exists()

        body = client.indices.create.call_args.kwargs["body"]
        properties = body["mappings"]["properties"]

        # 全部 design.md 中规定的字段必须出现
        expected_fields = {
            "chunk_id",
            "document_id",
            "space_id",
            "content",
            "title_chain",
            "source_file",
            "page_number",
            "chunk_index",
            "allowed_user_ids",
            "created_at",
        }
        assert set(properties.keys()) == expected_fields

        # IK 分词器必须被引用
        assert properties["content"]["analyzer"] == "ik_max_word"
        assert properties["content"]["search_analyzer"] == "ik_smart"
        assert properties["title_chain"]["analyzer"] == "ik_max_word"

    @patch("app.core.opensearch.get_opensearch_client")
    def test_skips_creation_when_index_exists(self, mock_get_client):
        """索引已存在时不再调用 indices.create（幂等）。"""
        client = MagicMock()
        client.indices.exists.return_value = True
        mock_get_client.return_value = client

        ensure_index_exists()

        client.indices.create.assert_not_called()

    @patch("app.core.opensearch.get_opensearch_client")
    def test_idempotent_across_repeated_calls(self, mock_get_client):
        """连续两次调用：首次创建，第二次因已存在而跳过。"""
        client = MagicMock()
        client.indices.exists.side_effect = [False, True]
        mock_get_client.return_value = client

        ensure_index_exists()
        ensure_index_exists()

        assert client.indices.create.call_count == 1

    @patch("app.core.opensearch.get_opensearch_client")
    def test_falls_back_when_ik_analyzer_unknown(self, mock_get_client):
        """RequestError 提示分词器未注册时，降级到 standard 分词器重试。"""
        client = MagicMock()
        client.indices.exists.return_value = False
        # 第一次：IK 不可用；第二次：降级映射成功
        ik_error = RequestError(
            400,
            "mapper_parsing_exception",
            {
                "error": {
                    "root_cause": [
                        {
                            "type": "mapper_parsing_exception",
                            "reason": "Unknown analyzer type [ik_max_word] for [content]",
                        }
                    ]
                }
            },
        )
        client.indices.create.side_effect = [ik_error, None]
        mock_get_client.return_value = client

        ensure_index_exists()

        assert client.indices.create.call_count == 2
        # 第二次调用必须使用降级映射
        fallback_body = client.indices.create.call_args_list[1].kwargs["body"]
        assert (
            fallback_body["mappings"]["properties"]["content"]["analyzer"]
            == "standard"
        )

    @patch("app.core.opensearch.get_opensearch_client")
    def test_falls_back_on_generic_exception_with_analyzer_message(
        self, mock_get_client
    ):
        """部分客户端在低层抛裸 Exception；只要消息提到分词器就应降级。"""
        client = MagicMock()
        client.indices.exists.return_value = False
        client.indices.create.side_effect = [
            Exception("analyzer [ik_max_word] not found"),
            None,
        ]
        mock_get_client.return_value = client

        ensure_index_exists()

        assert client.indices.create.call_count == 2

    @patch("app.core.opensearch.get_opensearch_client")
    def test_propagates_unrelated_request_error(self, mock_get_client):
        """与分词器无关的 RequestError 应直接向上抛，避免吞错。"""
        client = MagicMock()
        client.indices.exists.return_value = False
        unrelated_error = RequestError(
            400,
            "illegal_argument_exception",
            {
                "error": {
                    "root_cause": [
                        {
                            "type": "illegal_argument_exception",
                            "reason": "number_of_shards must be > 0",
                        }
                    ]
                }
            },
        )
        client.indices.create.side_effect = unrelated_error
        mock_get_client.return_value = client

        with pytest.raises(RequestError):
            ensure_index_exists()

        # 不应触发降级路径
        assert client.indices.create.call_count == 1

    @patch("app.core.opensearch.get_opensearch_client")
    def test_propagates_when_fallback_also_fails(self, mock_get_client):
        """降级路径仍失败时必须抛出，避免静默掩盖问题。"""
        client = MagicMock()
        client.indices.exists.return_value = False
        client.indices.create.side_effect = [
            Exception("Unknown analyzer type [ik_max_word]"),
            Exception("disk full"),
        ]
        mock_get_client.return_value = client

        with pytest.raises(Exception, match="disk full"):
            ensure_index_exists()


# ─── delete_index ─────────────────────────────────────────────────────


class TestDeleteIndex:
    """``delete_index`` 用于测试/重置场景。"""

    @patch("app.core.opensearch.get_opensearch_client")
    def test_calls_delete_when_index_exists(self, mock_get_client):
        client = MagicMock()
        client.indices.exists.return_value = True
        mock_get_client.return_value = client

        delete_index()

        client.indices.delete.assert_called_once_with(index="chunks")

    @patch("app.core.opensearch.get_opensearch_client")
    def test_swallows_when_index_missing(self, mock_get_client):
        """索引不存在时不应抛错，也不应调用 delete。"""
        client = MagicMock()
        client.indices.exists.return_value = False
        mock_get_client.return_value = client

        delete_index()  # 不应抛错

        client.indices.delete.assert_not_called()
