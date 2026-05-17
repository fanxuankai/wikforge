"""RAG (Retrieval-Augmented Generation) API routes.

Provides:
- POST /api/rag/chat: Conversational Q&A with SSE streaming
- GET /api/rag/sessions: List user's active chat sessions
- GET /api/rag/sessions/{session_id}/history: Get session message history
"""

import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.core.database import get_db
from app.core.redis import get_redis
from app.models.permission import AccessLevel, Permission, ResourceType
from app.models.user import User
from app.services.llm_gateway import LLMGatewayError
from app.services.rag_engine import (
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    LLM_ERROR_MESSAGE,
    MAX_TOP_K,
    MIN_TOP_K,
    RAGConfig,
    RAGEngine,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["rag"])


# ─── Request/Response Schemas ──────────────────────────────────────────


class ChatRequest(BaseModel):
    """Request body for RAG chat endpoint."""

    question: str = Field(
        ..., min_length=1, max_length=2000, description="用户问题（最多 2000 字符）"
    )
    session_id: str | None = Field(
        default=None, description="会话 ID（为空则创建新会话）"
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K, ge=MIN_TOP_K, le=MAX_TOP_K,
        description="检索文档块数量（1-20，默认 5）"
    )
    similarity_threshold: float = Field(
        default=DEFAULT_SIMILARITY_THRESHOLD, ge=0.0, le=1.0,
        description="相似度阈值（0-1，默认 0.5）"
    )
    llm_timeout: float = Field(
        default=DEFAULT_LLM_TIMEOUT, ge=5.0, le=300.0,
        description="LLM 超时时间（秒，默认 60）"
    )
    model: str | None = Field(
        default=None, description="LLM 模型（为空则使用默认模型）"
    )
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="LLM 温度参数"
    )


class SessionInfo(BaseModel):
    """Session information response.

    字段命名与前端 ChatSession 对齐 (id / last_active_at / preview)。
    """

    id: str
    last_active_at: str  # ISO 8601 字符串, 前端 new Date() 直接可解析
    preview: str = ""    # 首条用户问题的截断, 用于会话列表显示


class SessionListResponse(BaseModel):
    """Response for session list endpoint."""

    sessions: list[SessionInfo]


class MessageItem(BaseModel):
    """A single message in session history."""

    role: str
    content: str
    citations: list[dict] | None = None


class SessionHistoryResponse(BaseModel):
    """Response for session history endpoint."""

    session_id: str
    messages: list[MessageItem]


# ─── Dependencies ──────────────────────────────────────────────────────


async def get_rag_engine(redis: Redis = Depends(get_redis)) -> RAGEngine:
    """Dependency to get RAGEngine instance."""
    return RAGEngine(redis_client=redis)


async def get_user_allowed_space_ids(
    user: User,
    db: AsyncSession,
) -> list[str]:
    """Get list of space IDs the user has read or write access to.

    Admin (邮箱与 INITIAL_ADMIN_EMAIL 匹配) 拥有全部空间访问权。
    """
    from app.api.auth import is_admin_user
    from app.models.space import Space

    if is_admin_user(user):
        result = await db.execute(select(Space.id))
        return [str(sid) for sid in result.scalars().all()]

    stmt = select(Permission.resource_id).where(
        Permission.user_id == user.id,
        Permission.resource_type == ResourceType.space,
        Permission.access_level.in_([AccessLevel.read, AccessLevel.write]),
    )
    result = await db.execute(stmt)
    space_ids = result.scalars().all()
    return [str(sid) for sid in space_ids]


# ─── SSE Streaming Helper ─────────────────────────────────────────────


async def _sse_stream(
    rag_engine: RAGEngine,
    question: str,
    session_id: str | None,
    user_id: str,
    allowed_space_ids: list[str],
    config: RAGConfig,
):
    """Generate SSE events from RAG engine streaming response.

    SSE format:
    - data: <token> for each token
    - event: done when streaming is complete
    - event: error on failure

    Yields:
        SSE-formatted strings
    """
    try:
        full_response = ""
        async for token in rag_engine.chat(
            question=question,
            session_id=session_id,
            user_id=user_id,
            allowed_space_ids=allowed_space_ids,
            config=config,
        ):
            full_response += token
            # SSE data event
            data = json.dumps({"type": "token", "content": token}, ensure_ascii=False)
            yield f"data: {data}\n\n"

        # Send done event
        done_data = json.dumps({"type": "done", "content": full_response}, ensure_ascii=False)
        yield f"data: {done_data}\n\n"

    except LLMGatewayError as e:
        logger.error(f"RAG chat LLM error: {e}")
        error_data = json.dumps(
            {"type": "error", "content": LLM_ERROR_MESSAGE},
            ensure_ascii=False,
        )
        yield f"data: {error_data}\n\n"

    except Exception as e:
        logger.error(f"RAG chat unexpected error: {e}", exc_info=True)
        error_data = json.dumps(
            {"type": "error", "content": "服务异常，请稍后重试。"},
            ensure_ascii=False,
        )
        yield f"data: {error_data}\n\n"


# ─── Endpoints ─────────────────────────────────────────────────────────


@router.post("/chat")
async def chat(
    body: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    """RAG conversational Q&A with SSE streaming.

    Retrieves relevant document chunks, builds context prompt,
    and streams LLM response via Server-Sent Events.

    - Retrieves top-K chunks (default 5, configurable 1-20)
    - Filters by similarity threshold (default 0.5)
    - Maintains conversation history (last 20 turns, 30 min TTL)
    - Returns "未找到相关信息" if no relevant chunks found
    - Returns error message on LLM timeout (default 60s)
    """
    # Get user's accessible spaces
    allowed_space_ids = await get_user_allowed_space_ids(current_user, db)

    # Build RAG config from request
    config = RAGConfig(
        top_k=body.top_k,
        similarity_threshold=body.similarity_threshold,
        llm_timeout=body.llm_timeout,
        model=body.model,
        temperature=body.temperature,
    )

    # Return SSE streaming response
    return StreamingResponse(
        _sse_stream(
            rag_engine=rag_engine,
            question=body.question,
            session_id=body.session_id,
            user_id=str(current_user.id),
            allowed_space_ids=allowed_space_ids,
            config=config,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    current_user: User = Depends(get_current_user),
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    """List active chat sessions for the current user.

    Returns all non-expired sessions with their metadata.
    """
    try:
        sessions = await rag_engine.get_user_sessions(str(current_user.id))

        items: list[SessionInfo] = []
        for s in sessions:
            session_id = s["session_id"]
            # last_active 是 unix timestamp, 转 ISO 8601 字符串给前端
            ts = float(s.get("last_active", 0) or 0)
            iso_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

            # 取首条 user 消息当 preview
            try:
                msgs = await rag_engine.get_session_history(session_id)
                preview = ""
                for m in msgs:
                    if m.get("role") == "user":
                        preview = (m.get("content") or "")[:60]
                        break
            except Exception:
                preview = ""

            items.append(
                SessionInfo(id=session_id, last_active_at=iso_time, preview=preview)
            )

        # 按 last_active 倒序
        items.sort(key=lambda x: x.last_active_at, reverse=True)
        return SessionListResponse(sessions=items)
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="获取会话列表失败",
        )


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    """删除一个会话 (Redis 中清空)。

    只有会话所有者能删除。404 表示会话不存在或已过期。
    """
    try:
        session = await rag_engine.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        if session.get("user_id") != str(current_user.id):
            raise HTTPException(status_code=403, detail="无权删除该会话")

        await rag_engine.delete_session(session_id)
        return None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="删除会话失败")


@router.get("/sessions/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(
    session_id: str,
    current_user: User = Depends(get_current_user),
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    """Get message history for a specific session.

    Returns all messages in the session (up to 20 turns).
    Only the session owner can access the history.
    """
    try:
        # Verify session belongs to user
        session = await rag_engine.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")

        if session["user_id"] != str(current_user.id):
            raise HTTPException(status_code=403, detail="无权访问该会话")

        messages = await rag_engine.get_session_history(session_id)
        return SessionHistoryResponse(
            session_id=session_id,
            messages=[
                MessageItem(
                    role=m.get("role", ""),
                    content=m.get("content", ""),
                    citations=m.get("citations"),
                )
                for m in messages
            ],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session history: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="获取会话历史失败",
        )
