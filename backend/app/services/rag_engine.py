"""RAG Engine: Retrieval-Augmented Generation conversational Q&A.

Implements:
- RAGEngine.chat(): Retrieve top-K chunks → build prompt → stream LLM response
- Session management via Redis (last 20 turns, TTL 30 min)
- Citation parsing from LLM output ([1], [2] format)
- Similarity threshold filtering (default 0.5)
- LLM timeout handling (default 60s)
- Session expiration (30 min inactivity)
"""

import json
import logging
import re
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.services.llm_gateway import LLMGateway, LLMGatewayError
from app.services.search_service import SearchHit, SearchService

logger = logging.getLogger(__name__)

# Constants
DEFAULT_TOP_K = 5
MIN_TOP_K = 1
MAX_TOP_K = 20
DEFAULT_SIMILARITY_THRESHOLD = 0.5
DEFAULT_LLM_TIMEOUT = 60.0
SESSION_TTL_SECONDS = 1800  # 30 minutes
MAX_CONVERSATION_TURNS = 20
NO_RELEVANT_INFO_MESSAGE = "未找到相关信息，无法回答您的问题。请尝试换一种方式提问。"
LLM_ERROR_MESSAGE = "服务暂时不可用，请稍后重试。"

# Redis key prefix
SESSION_KEY_PREFIX = "session:"


@dataclass
class RAGConfig:
    """Configuration for a RAG chat request.

    Attributes:
        top_k: Number of top chunks to retrieve (1-20, default 5)
        similarity_threshold: Minimum similarity score (0-1, default 0.5)
        llm_timeout: LLM call timeout in seconds (default 60)
        model: LLM model to use (defaults to settings)
        temperature: LLM temperature (0-2, default 0.1)
        max_tokens: Maximum tokens for LLM response (default 4096)
    """

    top_k: int = DEFAULT_TOP_K
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    llm_timeout: float = DEFAULT_LLM_TIMEOUT
    model: str | None = None
    temperature: float = 0.1
    max_tokens: int = 4096


@dataclass
class Citation:
    """A citation reference in the RAG response.

    Attributes:
        index: Citation number (1-based)
        document_id: Source document ID
        chunk_id: Source chunk ID
        source_file: Source file name
        title_chain: Title chain (e.g., "Chapter 1 > Section 2")
        chunk_index: Position index of the chunk in the document
    """

    index: int
    document_id: str
    chunk_id: str
    source_file: str
    title_chain: str
    chunk_index: int


@dataclass
class RAGResponse:
    """Complete RAG response after streaming is done.

    Attributes:
        content: Full generated text
        citations: List of citations referenced in the response
        session_id: Session ID for this conversation
    """

    content: str = ""
    citations: list[Citation] = field(default_factory=list)
    session_id: str = ""


class RAGEngine:
    """RAG conversational Q&A engine.

    Orchestrates:
    1. Retrieve relevant document chunks via SearchService
    2. Build prompt with context + conversation history
    3. Stream LLM response via LLMGateway
    4. Parse citations from LLM output
    5. Manage conversation sessions in Redis
    """

    def __init__(
        self,
        search_service: SearchService | None = None,
        llm_gateway: LLMGateway | None = None,
        redis_client=None,
    ):
        """Initialize the RAG engine.

        Args:
            search_service: Service for retrieving relevant chunks.
            llm_gateway: Gateway for LLM calls.
            redis_client: Redis client for session management.
        """
        self._search_service = search_service or SearchService()
        self._llm_gateway = llm_gateway
        self._redis = redis_client

    def _get_llm_gateway(self, config: RAGConfig) -> LLMGateway:
        """Get or create LLM gateway with config-specific settings."""
        if self._llm_gateway:
            return self._llm_gateway
        return LLMGateway(
            model=config.model,
            timeout=config.llm_timeout,
        )

    async def _get_redis(self):
        """Get Redis client (lazy initialization)."""
        if self._redis is None:
            from app.core.redis import get_redis
            self._redis = await get_redis()
        return self._redis

    # ─── Main Chat Entry Point ─────────────────────────────────────────

    async def chat(
        self,
        question: str,
        session_id: str | None,
        user_id: str,
        allowed_space_ids: list[str],
        config: RAGConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a RAG response for the given question.

        Flow:
        1. Retrieve top-K chunks from search service
        2. Filter by similarity threshold
        3. If no relevant chunks, yield "未找到相关信息" message
        4. Build prompt with context + conversation history
        5. Stream LLM response
        6. Save conversation turn to Redis

        Args:
            question: User's question
            session_id: Existing session ID (None to create new)
            user_id: Current user's ID
            allowed_space_ids: Spaces the user can access
            config: RAG configuration

        Yields:
            Tokens/chunks of the LLM response as they arrive

        Raises:
            LLMGatewayError: If LLM call fails or times out
        """
        if config is None:
            config = RAGConfig()

        # Clamp top_k to valid range
        config.top_k = max(MIN_TOP_K, min(MAX_TOP_K, config.top_k))

        # Create or validate session
        if not session_id:
            session_id = str(uuid.uuid4())

        # Check if session is expired
        is_expired = await self._is_session_expired(session_id)
        if is_expired:
            session_id = str(uuid.uuid4())

        # 1. Retrieve relevant chunks
        chunks = await self._retrieve_chunks(
            question=question,
            user_id=user_id,
            allowed_space_ids=allowed_space_ids,
            top_k=config.top_k,
        )

        # 2. Filter by similarity threshold
        relevant_chunks = [
            c for c in chunks if c.score >= config.similarity_threshold
        ]

        # 3. If no relevant chunks, return "未找到相关信息"
        if not relevant_chunks:
            no_info_msg = NO_RELEVANT_INFO_MESSAGE
            # Save the turn even for no-result responses
            await self._save_turn(session_id, user_id, question, no_info_msg, [])
            yield no_info_msg
            return

        # 4. Get conversation history
        history = await self._get_conversation_history(session_id)

        # 5. Build prompt
        prompt, system_prompt = self._build_prompt(question, relevant_chunks, history)

        # 6. Stream LLM response
        llm = self._get_llm_gateway(config)
        full_response = ""

        try:
            async for token in llm.stream(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            ):
                full_response += token
                yield token
        except LLMGatewayError as e:
            if e.reason == "timeout":
                error_msg = LLM_ERROR_MESSAGE
                # Preserve session context on error
                await self._save_turn(session_id, user_id, question, error_msg, [])
                yield error_msg
                return
            raise

        # 7. Parse citations from response
        citations = self._parse_citations(full_response, relevant_chunks)

        # 8. Save conversation turn
        await self._save_turn(
            session_id, user_id, question, full_response,
            [self._citation_to_dict(c) for c in citations]
        )

    # ─── Retrieve Chunks ───────────────────────────────────────────────

    async def _retrieve_chunks(
        self,
        question: str,
        user_id: str,
        allowed_space_ids: list[str],
        top_k: int,
    ) -> list[SearchHit]:
        """Retrieve top-K relevant chunks using the search service.

        Args:
            question: User's question
            user_id: Current user's ID
            allowed_space_ids: Accessible space IDs
            top_k: Number of chunks to retrieve

        Returns:
            List of SearchHit results
        """
        response = await self._search_service.search(
            query=question,
            user_id=user_id,
            allowed_space_ids=allowed_space_ids,
            page=1,
            page_size=top_k,
        )

        # Convert SearchResult back to SearchHit for internal use
        hits = []
        for r in response.results:
            hits.append(SearchHit(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                chunk_index=r.chunk_index,
                title_chain=r.title_chain,
                source_file=r.source_file,
                content=r.highlight,  # Use highlight as content summary
                score=r.score,
            ))
        return hits

    # ─── Prompt Building ───────────────────────────────────────────────

    def _build_prompt(
        self,
        question: str,
        chunks: list[SearchHit],
        history: list[dict],
    ) -> tuple[str, str]:
        """Build the prompt with context and conversation history.

        The system prompt instructs the LLM to:
        - Answer based on provided context
        - Use [1], [2] citation format
        - Respond in the same language as the question

        Args:
            question: User's current question
            chunks: Retrieved relevant chunks
            history: Previous conversation turns

        Returns:
            Tuple of (user_prompt, system_prompt)
        """
        system_prompt = (
            "你是一个企业知识库助手。请根据提供的参考资料回答用户的问题。\n\n"
            "规则：\n"
            "1. 仅基于提供的参考资料回答，不要编造信息。\n"
            "2. 在回答中使用 [1]、[2] 等格式标注引用来源，对应参考资料的编号。\n"
            "3. 如果参考资料不足以回答问题，请明确说明。\n"
            "4. 使用与用户问题相同的语言回答。\n"
            "5. 回答要简洁、准确、有条理。"
        )

        # Build context section
        context_parts = []
        for i, chunk in enumerate(chunks, start=1):
            source_info = f"[来源: {chunk.source_file}"
            if chunk.title_chain:
                source_info += f" > {chunk.title_chain}"
            source_info += "]"
            context_parts.append(f"[{i}] {source_info}\n{chunk.content}")

        context_section = "\n\n".join(context_parts)

        # Build conversation history section
        history_section = ""
        if history:
            history_lines = []
            for turn in history:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role == "user":
                    history_lines.append(f"用户: {content}")
                elif role == "assistant":
                    history_lines.append(f"助手: {content}")
            history_section = "\n".join(history_lines)

        # Compose user prompt
        user_prompt_parts = [f"参考资料：\n{context_section}"]
        if history_section:
            user_prompt_parts.append(f"\n对话历史：\n{history_section}")
        user_prompt_parts.append(f"\n当前问题：{question}")

        user_prompt = "\n".join(user_prompt_parts)

        return user_prompt, system_prompt

    # ─── Citation Parsing ──────────────────────────────────────────────

    def _parse_citations(
        self, response_text: str, chunks: list[SearchHit]
    ) -> list[Citation]:
        """Parse citation references from LLM response text.

        Looks for [1], [2], etc. patterns and maps them to source chunks.

        Args:
            response_text: Full LLM response text
            chunks: The chunks that were provided as context

        Returns:
            List of Citation objects for referenced chunks
        """
        # Find all citation numbers in the response
        citation_pattern = re.compile(r"\[(\d+)\]")
        matches = citation_pattern.findall(response_text)

        # Deduplicate and sort
        cited_indices = sorted(set(int(m) for m in matches))

        citations = []
        for idx in cited_indices:
            # Citation indices are 1-based, chunks list is 0-based
            chunk_pos = idx - 1
            if 0 <= chunk_pos < len(chunks):
                chunk = chunks[chunk_pos]
                citations.append(Citation(
                    index=idx,
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    source_file=chunk.source_file,
                    title_chain=chunk.title_chain,
                    chunk_index=chunk.chunk_index,
                ))

        return citations

    def _citation_to_dict(self, citation: Citation) -> dict:
        """Convert a Citation to a JSON-serializable dict."""
        return {
            "index": citation.index,
            "document_id": citation.document_id,
            "chunk_id": citation.chunk_id,
            "source_file": citation.source_file,
            "title_chain": citation.title_chain,
            "chunk_index": citation.chunk_index,
        }

    # ─── Session Management ────────────────────────────────────────────

    async def _get_session_key(self, session_id: str) -> str:
        """Get the Redis key for a session."""
        return f"{SESSION_KEY_PREFIX}{session_id}"

    async def _is_session_expired(self, session_id: str) -> bool:
        """Check if a session is expired (30 min inactivity).

        Args:
            session_id: Session ID to check

        Returns:
            True if session is expired or doesn't exist
        """
        redis = await self._get_redis()
        key = await self._get_session_key(session_id)

        session_data = await redis.hgetall(key)
        if not session_data:
            return False  # New session, not expired

        last_active = session_data.get("last_active")
        if last_active:
            elapsed = time.time() - float(last_active)
            if elapsed > SESSION_TTL_SECONDS:
                return True

        return False

    async def _get_conversation_history(self, session_id: str) -> list[dict]:
        """Get conversation history from Redis.

        Returns the last 20 turns (messages) for the session.

        Args:
            session_id: Session ID

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        redis = await self._get_redis()
        key = await self._get_session_key(session_id)

        messages_json = await redis.hget(key, "messages")
        if not messages_json:
            return []

        try:
            messages = json.loads(messages_json)
            # Return last 20 turns (each turn = user + assistant = 2 messages)
            # Max 20 turns = 40 messages
            max_messages = MAX_CONVERSATION_TURNS * 2
            return messages[-max_messages:]
        except (json.JSONDecodeError, TypeError):
            return []

    async def _save_turn(
        self,
        session_id: str,
        user_id: str,
        question: str,
        answer: str,
        citations: list[dict],
    ) -> None:
        """Save a conversation turn to Redis.

        Appends user question and assistant answer to the session's message list.
        Maintains max 20 turns (40 messages). Refreshes TTL to 30 minutes.

        Args:
            session_id: Session ID
            user_id: User ID
            question: User's question
            answer: Assistant's answer
            citations: Citation data for the answer
        """
        redis = await self._get_redis()
        key = await self._get_session_key(session_id)

        # Get existing messages
        messages_json = await redis.hget(key, "messages")
        if messages_json:
            try:
                messages = json.loads(messages_json)
            except (json.JSONDecodeError, TypeError):
                messages = []
        else:
            messages = []

        # Append new turn
        messages.append({"role": "user", "content": question})
        messages.append({
            "role": "assistant",
            "content": answer,
            "citations": citations,
        })

        # Trim to max 20 turns (40 messages)
        max_messages = MAX_CONVERSATION_TURNS * 2
        if len(messages) > max_messages:
            messages = messages[-max_messages:]

        # Save to Redis
        now = time.time()
        await redis.hset(key, mapping={
            "user_id": user_id,
            "messages": json.dumps(messages, ensure_ascii=False),
            "last_active": str(now),
            "session_id": session_id,
        })

        # Set TTL (30 minutes)
        await redis.expire(key, SESSION_TTL_SECONDS)

    # ─── Session Queries ───────────────────────────────────────────────

    async def get_session(self, session_id: str) -> dict | None:
        """Get session data from Redis.

        Args:
            session_id: Session ID

        Returns:
            Session data dict or None if not found
        """
        redis = await self._get_redis()
        key = await self._get_session_key(session_id)

        session_data = await redis.hgetall(key)
        if not session_data:
            return None

        return {
            "session_id": session_id,
            "user_id": session_data.get("user_id", ""),
            "last_active": float(session_data.get("last_active", 0)),
            "is_expired": await self._is_session_expired(session_id),
        }

    async def get_user_sessions(self, user_id: str) -> list[dict]:
        """Get all active sessions for a user.

        Scans Redis for sessions belonging to the user.
        Note: In production, consider maintaining a user→sessions index.

        Args:
            user_id: User ID

        Returns:
            List of session data dicts
        """
        redis = await self._get_redis()
        sessions = []

        # Scan for session keys
        async for key in redis.scan_iter(match=f"{SESSION_KEY_PREFIX}*"):
            session_data = await redis.hgetall(key)
            if session_data.get("user_id") == user_id:
                session_id = session_data.get("session_id", key.replace(SESSION_KEY_PREFIX, ""))
                is_expired = await self._is_session_expired(session_id)
                if not is_expired:
                    sessions.append({
                        "session_id": session_id,
                        "user_id": user_id,
                        "last_active": float(session_data.get("last_active", 0)),
                        "is_expired": False,
                    })

        return sessions

    async def get_session_history(self, session_id: str) -> list[dict]:
        """Get full message history for a session.

        Args:
            session_id: Session ID

        Returns:
            List of message dicts
        """
        return await self._get_conversation_history(session_id)

    async def delete_session(self, session_id: str) -> bool:
        """删除一个会话 (Redis hash key 整体清空)。

        Args:
            session_id: Session ID

        Returns:
            True 表示有删除发生, False 表示 key 不存在
        """
        redis = await self._get_redis()
        key = await self._get_session_key(session_id)
        deleted = await redis.delete(key)
        return bool(deleted)
