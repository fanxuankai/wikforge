"""任务 16.9：RAG API 接口端到端测试。

覆盖 ``backend/app/api/rag.py`` 中定义的三个端点：

- ``POST /api/rag/chat``：SSE 流式问答
- ``GET  /api/rag/sessions``：列出当前用户的活跃会话
- ``GET  /api/rag/sessions/{session_id}/history``：获取指定会话的历史消息

测试要点：
- 通过 ``app.dependency_overrides`` 注入 mock 的 ``get_current_user``、
  ``get_db``、``get_rag_engine``，避免触达真实的数据库 / Redis / LLM。
- ``TestClient.stream`` 用于消费 SSE 响应体。
- 鉴权失败（即未提供 ``current_user`` 依赖）走 FastAPI 默认行为：
  本测试通过覆写依赖项模拟"已登录"，并显式额外构造一个不覆写
  ``get_current_user`` 的应用来验证未鉴权场景下的 401 行为。
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth import get_current_user
from app.api.rag import (
    get_rag_engine,
    router as rag_router,
)
from app.core.database import get_db
from app.core.exceptions import register_exception_handlers
from app.services.rag_engine import NO_RELEVANT_INFO_MESSAGE


# ─── 辅助：FakeRAGEngine ────────────────────────────────────────────────


class FakeRAGEngine:
    """一个仅供 API 测试使用的简化 RAG 引擎替身。

    - ``chat`` 返回预设 token 列表，按 ``async generator`` 协议产出。
    - ``get_user_sessions`` / ``get_session`` / ``get_session_history``
      从内存字典里读取，模拟 Redis 行为。
    - 不实现任何 token 解析 / Redis 交互，使路由层的 SSE 编码、依赖装配、
      鉴权与权限拼接路径成为唯一被测对象。
    """

    def __init__(
        self,
        tokens: list[str] | None = None,
        sessions: dict[str, dict] | None = None,
        histories: dict[str, list[dict]] | None = None,
    ) -> None:
        self._tokens = tokens or ["你", "好"]
        self._sessions = sessions or {}
        self._histories = histories or {}
        # 调用记录用于断言：[(question, session_id, user_id, allowed_space_ids)]
        self.chat_calls: list[tuple[str, str | None, str, list[str]]] = []

    async def chat(
        self,
        question: str,
        session_id: str | None,
        user_id: str,
        allowed_space_ids: list[str],
        config: Any | None = None,
    ) -> AsyncGenerator[str, None]:
        self.chat_calls.append(
            (question, session_id, user_id, list(allowed_space_ids))
        )
        for tok in self._tokens:
            yield tok

    async def get_user_sessions(self, user_id: str) -> list[dict]:
        return [
            s for s in self._sessions.values() if s.get("user_id") == user_id
        ]

    async def get_session(self, session_id: str) -> dict | None:
        return self._sessions.get(session_id)

    async def get_session_history(self, session_id: str) -> list[dict]:
        return list(self._histories.get(session_id, []))


# ─── 辅助：FailingRAGEngine ─────────────────────────────────────────────


class FailingRAGEngine(FakeRAGEngine):
    """在 ``chat`` 流中抛出 LLM 错误的引擎，用来验证 SSE error 事件。"""

    def __init__(self, exc: Exception, tokens_before: list[str] | None = None):
        super().__init__()
        self._exc = exc
        self._tokens_before = tokens_before or []

    async def chat(  # type: ignore[override]
        self,
        question: str,
        session_id: str | None,
        user_id: str,
        allowed_space_ids: list[str],
        config: Any | None = None,
    ) -> AsyncGenerator[str, None]:
        for tok in self._tokens_before:
            yield tok
        raise self._exc


# ─── 辅助：构造测试 App ────────────────────────────────────────────────


def _build_app(
    engine: FakeRAGEngine,
    *,
    user_id: str | None = None,
    allowed_space_ids: list[str] | None = None,
    skip_auth_override: bool = False,
) -> tuple[FastAPI, str, list[str]]:
    """组装一个仅注册 RAG 路由 + mock 依赖的 FastAPI app。

    Args:
        engine: 注入到 ``get_rag_engine`` 的引擎替身。
        user_id: ``current_user.id`` 取值；缺省时随机生成 UUID。
        allowed_space_ids: 注入到 DB 查询返回的空间 ID 列表；
            缺省给一个随机 UUID，模拟"用户至少能访问一个空间"。
        skip_auth_override: 为 ``True`` 时不覆写 ``get_current_user``，
            用于验证未鉴权访问下的行为。

    Returns:
        ``(app, user_id, allowed_space_ids)``——便于测试用例做断言。
    """
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(rag_router)

    final_user_id = user_id or str(uuid.uuid4())
    final_allowed_space_ids = (
        allowed_space_ids
        if allowed_space_ids is not None
        else [str(uuid.uuid4())]
    )

    if not skip_auth_override:
        fake_user = MagicMock()
        fake_user.id = uuid.UUID(final_user_id)

        async def _override_user():
            return fake_user

        app.dependency_overrides[get_current_user] = _override_user

    # 模拟 ``select(Permission.resource_id)...`` 返回 allowed_space_ids
    db_session = AsyncMock()
    db_result = MagicMock()
    db_result.scalars.return_value.all.return_value = final_allowed_space_ids
    db_session.execute = AsyncMock(return_value=db_result)

    async def _override_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_rag_engine] = lambda: engine
    return app, final_user_id, final_allowed_space_ids


def _parse_sse(body: str) -> list[dict]:
    """把 SSE 响应体拆成 JSON 解码后的 data 列表。

    - 仅识别 ``data: ...\\n\\n`` 块（路由实现统一使用 ``data:`` 形式）。
    - 跳过空块。
    """
    parsed: list[dict] = []
    for block in body.strip().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        for line in block.splitlines():
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
                if payload:
                    parsed.append(json.loads(payload))
    return parsed


# ─── POST /api/rag/chat ────────────────────────────────────────────────


class TestRAGChatRoute:
    """``POST /api/rag/chat`` SSE 流式问答端点测试。"""

    def test_streams_tokens_then_done(self):
        """正常路径：依次产出 token 事件，最后产出 done 事件。"""
        engine = FakeRAGEngine(tokens=["你", "好", "！"])
        app, user_id, allowed = _build_app(engine)
        client = TestClient(app)

        with client.stream(
            "POST",
            "/api/rag/chat",
            json={"question": "你好"},
        ) as resp:
            assert resp.status_code == 200
            # SSE 媒体类型与缓存控制响应头
            assert resp.headers["content-type"].startswith("text/event-stream")
            assert resp.headers.get("cache-control") == "no-cache"
            body = resp.read().decode("utf-8")

        events = _parse_sse(body)
        # 最后一个事件必为 done，前面是若干 token
        kinds = [e["type"] for e in events]
        assert kinds == ["token", "token", "token", "done"]

        token_texts = [e["content"] for e in events if e["type"] == "token"]
        assert token_texts == ["你", "好", "！"]

        # done 事件返回完整答案
        done_event = events[-1]
        assert done_event["content"] == "你好！"

        # 引擎被以正确参数调用
        assert len(engine.chat_calls) == 1
        question, session_id, called_user_id, called_spaces = (
            engine.chat_calls[0]
        )
        assert question == "你好"
        assert session_id is None
        assert called_user_id == user_id
        assert called_spaces == allowed

    def test_passes_session_id_through(self):
        """请求中显式传 session_id 时应原样透传给引擎。"""
        engine = FakeRAGEngine(tokens=["回", "答"])
        app, _, _ = _build_app(engine)
        client = TestClient(app)

        existing_session = "sess-123"
        with client.stream(
            "POST",
            "/api/rag/chat",
            json={"question": "继续聊", "session_id": existing_session},
        ) as resp:
            assert resp.status_code == 200
            resp.read()

        _, session_id, _, _ = engine.chat_calls[0]
        assert session_id == existing_session

    def test_validates_question_min_length(self):
        """空 question 应返回 422 而非透传到引擎。"""
        engine = FakeRAGEngine()
        app, _, _ = _build_app(engine)
        client = TestClient(app)

        resp = client.post("/api/rag/chat", json={"question": ""})
        assert resp.status_code == 422
        assert engine.chat_calls == []

    def test_validates_question_max_length(self):
        """超过 2000 字符的 question 应返回 422。"""
        engine = FakeRAGEngine()
        app, _, _ = _build_app(engine)
        client = TestClient(app)

        resp = client.post(
            "/api/rag/chat", json={"question": "x" * 2001}
        )
        assert resp.status_code == 422
        assert engine.chat_calls == []

    def test_top_k_out_of_range_rejected(self):
        """``top_k`` 超出 [1, 20] 区间应返回 422，与服务层夹紧解耦。"""
        engine = FakeRAGEngine()
        app, _, _ = _build_app(engine)
        client = TestClient(app)

        # 0 / 21 都应被 Pydantic 拒绝
        for invalid in (0, 21, -1):
            resp = client.post(
                "/api/rag/chat",
                json={"question": "Hi", "top_k": invalid},
            )
            assert resp.status_code == 422, invalid

    def test_no_relevant_info_path(self):
        """检索为空时引擎产出固定提示，路由也应以 token + done 结尾。"""
        engine = FakeRAGEngine(tokens=[NO_RELEVANT_INFO_MESSAGE])
        app, _, _ = _build_app(engine)
        client = TestClient(app)

        with client.stream(
            "POST",
            "/api/rag/chat",
            json={"question": "未知问题"},
        ) as resp:
            assert resp.status_code == 200
            body = resp.read().decode("utf-8")

        events = _parse_sse(body)
        assert events[0] == {
            "type": "token",
            "content": NO_RELEVANT_INFO_MESSAGE,
        }
        assert events[-1]["type"] == "done"
        assert events[-1]["content"] == NO_RELEVANT_INFO_MESSAGE

    def test_llm_gateway_error_emits_error_event(self):
        """``LLMGatewayError`` 应被路由捕获并以 SSE error 事件返回。"""
        from app.services.llm_gateway import LLMGatewayError

        engine = FailingRAGEngine(
            exc=LLMGatewayError("auth failed", reason="auth"),
            tokens_before=["先", "导"],
        )
        app, _, _ = _build_app(engine)
        client = TestClient(app)

        with client.stream(
            "POST",
            "/api/rag/chat",
            json={"question": "x"},
        ) as resp:
            assert resp.status_code == 200
            body = resp.read().decode("utf-8")

        events = _parse_sse(body)
        kinds = [e["type"] for e in events]
        # 出错前已产出的两个 token 应保留，最后一个事件为 error
        assert kinds[:2] == ["token", "token"]
        assert kinds[-1] == "error"
        # 不应再出现 done
        assert "done" not in kinds

    def test_unexpected_error_emits_generic_error_event(self):
        """非 LLM 类型异常也应被兜底转换为 error 事件，不抛 500。"""
        engine = FailingRAGEngine(exc=RuntimeError("boom"))
        app, _, _ = _build_app(engine)
        client = TestClient(app)

        with client.stream(
            "POST", "/api/rag/chat", json={"question": "x"}
        ) as resp:
            assert resp.status_code == 200
            body = resp.read().decode("utf-8")

        events = _parse_sse(body)
        # 第一个（也是唯一）事件为 error
        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert isinstance(events[0]["content"], str)
        # 不暴露原始异常 message
        assert "boom" not in events[0]["content"]

    def test_requires_authentication(self):
        """未鉴权访问应返回 401/403，不会调到引擎。"""
        engine = FakeRAGEngine()
        app, _, _ = _build_app(engine, skip_auth_override=True)
        client = TestClient(app)

        resp = client.post("/api/rag/chat", json={"question": "Hi"})
        # 鉴权依赖失败时由 FastAPI 抛 HTTPException → 401/403
        assert resp.status_code in (401, 403)
        assert engine.chat_calls == []


# ─── GET /api/rag/sessions ─────────────────────────────────────────────


class TestListSessionsRoute:
    """``GET /api/rag/sessions`` 端点测试。"""

    def test_returns_sessions_for_current_user(self):
        """只返回 ``user_id`` 等于当前用户的会话。"""
        user_id = str(uuid.uuid4())
        other_id = str(uuid.uuid4())
        now = time.time()

        engine = FakeRAGEngine(
            sessions={
                "s1": {
                    "session_id": "s1",
                    "user_id": user_id,
                    "last_active": now,
                    "is_expired": False,
                },
                "s2": {
                    "session_id": "s2",
                    "user_id": user_id,
                    "last_active": now - 60,
                    "is_expired": False,
                },
                "s3": {
                    "session_id": "s3",
                    "user_id": other_id,
                    "last_active": now,
                    "is_expired": False,
                },
            }
        )
        app, _, _ = _build_app(engine, user_id=user_id)
        client = TestClient(app)

        resp = client.get("/api/rag/sessions")
        assert resp.status_code == 200
        body = resp.json()
        ids = sorted(s["id"] for s in body["sessions"])
        assert ids == ["s1", "s2"]
        # 字段齐全 (新 schema: id / last_active_at / preview)
        for s in body["sessions"]:
            assert "id" in s
            # last_active_at 是 ISO 8601 字符串
            assert isinstance(s["last_active_at"], str)
            assert "preview" in s

    def test_returns_empty_when_no_sessions(self):
        """无会话时应返回空数组而非 404。"""
        engine = FakeRAGEngine(sessions={})
        app, _, _ = _build_app(engine)
        client = TestClient(app)

        resp = client.get("/api/rag/sessions")
        assert resp.status_code == 200
        assert resp.json() == {"sessions": []}

    def test_engine_failure_returns_500(self):
        """引擎抛错应被路由捕获并返回 500，含中文错误提示。"""
        engine = FakeRAGEngine()
        engine.get_user_sessions = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("redis down")
        )
        app, _, _ = _build_app(engine)
        client = TestClient(app)

        resp = client.get("/api/rag/sessions")
        assert resp.status_code == 500
        # 错误体由全局异常处理器格式化，但 detail 应为中文友好提示
        body = resp.json()
        # 不同异常处理器结构略异，统一在 JSON 串里搜中文提示
        assert "会话列表" in json.dumps(body, ensure_ascii=False)

    def test_requires_authentication(self):
        """未鉴权访问应返回 401/403。"""
        engine = FakeRAGEngine()
        app, _, _ = _build_app(engine, skip_auth_override=True)
        client = TestClient(app)

        resp = client.get("/api/rag/sessions")
        assert resp.status_code in (401, 403)


# ─── GET /api/rag/sessions/{id}/history ────────────────────────────────


class TestSessionHistoryRoute:
    """``GET /api/rag/sessions/{session_id}/history`` 端点测试。"""

    def test_returns_messages_for_owner(self):
        """会话归属当前用户时返回其历史消息。"""
        user_id = str(uuid.uuid4())
        engine = FakeRAGEngine(
            sessions={
                "s1": {
                    "session_id": "s1",
                    "user_id": user_id,
                    "last_active": time.time(),
                    "is_expired": False,
                }
            },
            histories={
                "s1": [
                    {"role": "user", "content": "你好"},
                    {
                        "role": "assistant",
                        "content": "你好[1]",
                        "citations": [{"index": 1, "source_file": "a.pdf"}],
                    },
                ]
            },
        )
        app, _, _ = _build_app(engine, user_id=user_id)
        client = TestClient(app)

        resp = client.get("/api/rag/sessions/s1/history")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "s1"
        assert len(body["messages"]) == 2

        first, second = body["messages"]
        assert first["role"] == "user"
        assert first["content"] == "你好"
        assert first["citations"] is None

        assert second["role"] == "assistant"
        assert second["content"] == "你好[1]"
        assert second["citations"] == [
            {"index": 1, "source_file": "a.pdf"}
        ]

    def test_404_when_session_not_found(self):
        """未知会话返回 404。"""
        engine = FakeRAGEngine(sessions={})
        app, _, _ = _build_app(engine)
        client = TestClient(app)

        resp = client.get("/api/rag/sessions/nonexistent/history")
        assert resp.status_code == 404
        assert "会话不存在" in json.dumps(resp.json(), ensure_ascii=False)

    def test_403_when_session_belongs_to_other_user(self):
        """会话归属其他用户时拒绝访问。"""
        owner_id = str(uuid.uuid4())
        intruder_id = str(uuid.uuid4())
        engine = FakeRAGEngine(
            sessions={
                "s1": {
                    "session_id": "s1",
                    "user_id": owner_id,
                    "last_active": time.time(),
                    "is_expired": False,
                }
            },
            histories={"s1": [{"role": "user", "content": "x"}]},
        )
        app, _, _ = _build_app(engine, user_id=intruder_id)
        client = TestClient(app)

        resp = client.get("/api/rag/sessions/s1/history")
        assert resp.status_code == 403
        assert "无权访问" in json.dumps(resp.json(), ensure_ascii=False)

    def test_returns_empty_messages_for_owned_but_empty_session(self):
        """会话存在但无消息时返回空列表（不报错）。"""
        user_id = str(uuid.uuid4())
        engine = FakeRAGEngine(
            sessions={
                "s1": {
                    "session_id": "s1",
                    "user_id": user_id,
                    "last_active": time.time(),
                    "is_expired": False,
                }
            },
            histories={"s1": []},
        )
        app, _, _ = _build_app(engine, user_id=user_id)
        client = TestClient(app)

        resp = client.get("/api/rag/sessions/s1/history")
        assert resp.status_code == 200
        assert resp.json() == {"session_id": "s1", "messages": []}

    def test_engine_failure_returns_500(self):
        """``get_session_history`` 抛错时返回 500 + 中文提示。"""
        user_id = str(uuid.uuid4())
        engine = FakeRAGEngine(
            sessions={
                "s1": {
                    "session_id": "s1",
                    "user_id": user_id,
                    "last_active": time.time(),
                    "is_expired": False,
                }
            }
        )
        engine.get_session_history = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("oops")
        )
        app, _, _ = _build_app(engine, user_id=user_id)
        client = TestClient(app)

        resp = client.get("/api/rag/sessions/s1/history")
        assert resp.status_code == 500
        assert "会话历史" in json.dumps(resp.json(), ensure_ascii=False)

    def test_requires_authentication(self):
        """未鉴权访问应返回 401/403。"""
        engine = FakeRAGEngine()
        app, _, _ = _build_app(engine, skip_auth_override=True)
        client = TestClient(app)

        resp = client.get("/api/rag/sessions/s1/history")
        assert resp.status_code in (401, 403)
