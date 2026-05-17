"""Unit tests for the RAG engine and API.

Tests cover:
- RAG Engine core logic (retrieve → prompt → stream → citations)
- Similarity threshold filtering (< 0.5 returns "未找到相关信息")
- Citation parsing from LLM output ([1], [2] format)
- Session management (Redis storage, 20 turns max, TTL 30 min)
- Session expiration (30 min inactivity)
- LLM timeout handling (60s default, error message on timeout)
- Prompt building (context + history)
- RAG API endpoints (POST /api/rag/chat, GET /api/rag/sessions)
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.rag_engine import (
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    LLM_ERROR_MESSAGE,
    MAX_CONVERSATION_TURNS,
    MAX_TOP_K,
    MIN_TOP_K,
    NO_RELEVANT_INFO_MESSAGE,
    SESSION_KEY_PREFIX,
    SESSION_TTL_SECONDS,
    Citation,
    RAGConfig,
    RAGEngine,
    RAGResponse,
)
from app.services.llm_gateway import LLMGatewayError
from app.services.search_service import SearchHit, SearchResponse, SearchResult


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_redis():
    """Create a mock Redis client that simulates hash operations."""
    redis = AsyncMock()
    _store = {}

    async def mock_hset(key, mapping=None, **kwargs):
        if key not in _store:
            _store[key] = {}
        if mapping:
            _store[key].update(mapping)
        _store[key].update(kwargs)

    async def mock_hget(key, field):
        if key in _store:
            return _store[key].get(field)
        return None

    async def mock_hgetall(key):
        return _store.get(key, {})

    async def mock_expire(key, ttl):
        pass

    async def mock_scan_iter(match=None):
        for key in _store:
            if match and not key.startswith(match.replace("*", "")):
                continue
            yield key

    redis.hset = mock_hset
    redis.hget = mock_hget
    redis.hgetall = mock_hgetall
    redis.expire = mock_expire
    redis.scan_iter = mock_scan_iter
    redis._store = _store
    return redis


@pytest.fixture
def mock_search_service():
    """Create a mock search service."""
    service = AsyncMock()
    return service


@pytest.fixture
def mock_llm_gateway():
    """Create a mock LLM gateway with streaming support."""
    gateway = AsyncMock()
    return gateway


@pytest.fixture
def rag_engine(mock_search_service, mock_llm_gateway, mock_redis):
    """Create a RAGEngine with all dependencies mocked."""
    return RAGEngine(
        search_service=mock_search_service,
        llm_gateway=mock_llm_gateway,
        redis_client=mock_redis,
    )


def make_search_result(
    chunk_id: str | None = None,
    document_id: str | None = None,
    content: str = "test content about AI",
    score: float = 0.85,
    chunk_index: int = 0,
    title_chain: str = "Chapter 1 > Section 2",
    source_file: str = "test.pdf",
) -> SearchResult:
    """Helper to create a SearchResult for testing."""
    return SearchResult(
        chunk_id=chunk_id or str(uuid.uuid4()),
        document_id=document_id or str(uuid.uuid4()),
        chunk_index=chunk_index,
        title_chain=title_chain,
        source_file=source_file,
        score=score,
        highlight=content,
    )


def make_search_response(results: list[SearchResult] | None = None) -> SearchResponse:
    """Helper to create a SearchResponse."""
    if results is None:
        results = [make_search_result()]
    return SearchResponse(
        results=results,
        total=len(results),
        page=1,
        page_size=len(results),
    )


async def collect_stream(gen: AsyncGenerator[str, None]) -> str:
    """Helper to collect all tokens from an async generator."""
    tokens = []
    async for token in gen:
        tokens.append(token)
    return "".join(tokens)


# ─── RAG Config Tests ──────────────────────────────────────────────────


class TestRAGConfig:
    """Tests for RAG configuration defaults and validation."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = RAGConfig()
        assert config.top_k == DEFAULT_TOP_K
        assert config.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD
        assert config.llm_timeout == DEFAULT_LLM_TIMEOUT
        assert config.model is None
        assert config.temperature == 0.1
        assert config.max_tokens == 4096

    def test_custom_config(self):
        """Custom config values should be preserved."""
        config = RAGConfig(
            top_k=10,
            similarity_threshold=0.7,
            llm_timeout=30.0,
            model="gpt-4o",
            temperature=0.5,
        )
        assert config.top_k == 10
        assert config.similarity_threshold == 0.7
        assert config.llm_timeout == 30.0
        assert config.model == "gpt-4o"


# ─── Citation Parsing Tests ────────────────────────────────────────────


class TestCitationParsing:
    """Tests for citation parsing from LLM output."""

    def test_parse_single_citation(self, rag_engine):
        """Should parse a single [1] citation."""
        chunks = [
            SearchHit(
                chunk_id="chunk_1", document_id="doc_1",
                source_file="report.pdf", title_chain="Intro",
                content="content", score=0.9, chunk_index=0,
            )
        ]
        response = "Based on the document [1], AI is transformative."

        citations = rag_engine._parse_citations(response, chunks)

        assert len(citations) == 1
        assert citations[0].index == 1
        assert citations[0].chunk_id == "chunk_1"
        assert citations[0].source_file == "report.pdf"

    def test_parse_multiple_citations(self, rag_engine):
        """Should parse multiple citations [1], [2], [3]."""
        chunks = [
            SearchHit(chunk_id=f"chunk_{i}", document_id=f"doc_{i}",
                      source_file=f"file_{i}.pdf", title_chain=f"Section {i}",
                      content="content", score=0.9, chunk_index=i)
            for i in range(3)
        ]
        response = "According to [1] and [2], with support from [3]."

        citations = rag_engine._parse_citations(response, chunks)

        assert len(citations) == 3
        assert citations[0].index == 1
        assert citations[1].index == 2
        assert citations[2].index == 3

    def test_parse_duplicate_citations(self, rag_engine):
        """Should deduplicate repeated citation references."""
        chunks = [
            SearchHit(chunk_id="chunk_1", document_id="doc_1",
                      source_file="report.pdf", title_chain="Intro",
                      content="content", score=0.9, chunk_index=0)
        ]
        response = "As stated in [1], and confirmed by [1] again."

        citations = rag_engine._parse_citations(response, chunks)

        assert len(citations) == 1

    def test_parse_no_citations(self, rag_engine):
        """Should return empty list when no citations in response."""
        chunks = [
            SearchHit(chunk_id="chunk_1", document_id="doc_1",
                      source_file="report.pdf", title_chain="Intro",
                      content="content", score=0.9, chunk_index=0)
        ]
        response = "This is a response without any citations."

        citations = rag_engine._parse_citations(response, chunks)

        assert len(citations) == 0

    def test_parse_out_of_range_citation(self, rag_engine):
        """Should ignore citation indices that exceed chunk count."""
        chunks = [
            SearchHit(chunk_id="chunk_1", document_id="doc_1",
                      source_file="report.pdf", title_chain="Intro",
                      content="content", score=0.9, chunk_index=0)
        ]
        response = "See [1] and [5] for details."

        citations = rag_engine._parse_citations(response, chunks)

        # Only [1] is valid, [5] is out of range
        assert len(citations) == 1
        assert citations[0].index == 1

    def test_citation_to_dict(self, rag_engine):
        """Should convert Citation to serializable dict."""
        citation = Citation(
            index=1,
            document_id="doc_1",
            chunk_id="chunk_1",
            source_file="report.pdf",
            title_chain="Chapter 1",
            chunk_index=3,
        )

        result = rag_engine._citation_to_dict(citation)

        assert result["index"] == 1
        assert result["document_id"] == "doc_1"
        assert result["chunk_id"] == "chunk_1"
        assert result["source_file"] == "report.pdf"
        assert result["title_chain"] == "Chapter 1"
        assert result["chunk_index"] == 3


# ─── Prompt Building Tests ─────────────────────────────────────────────


class TestPromptBuilding:
    """Tests for prompt construction."""

    def test_build_prompt_with_chunks(self, rag_engine):
        """Should include numbered context chunks in prompt."""
        chunks = [
            SearchHit(chunk_id="c1", document_id="d1", source_file="doc.pdf",
                      title_chain="Intro", content="AI is powerful.", score=0.9,
                      chunk_index=0),
            SearchHit(chunk_id="c2", document_id="d2", source_file="guide.pdf",
                      title_chain="Chapter 2", content="ML basics.", score=0.8,
                      chunk_index=1),
        ]

        user_prompt, system_prompt = rag_engine._build_prompt(
            "What is AI?", chunks, []
        )

        # System prompt should contain instructions
        assert "引用来源" in system_prompt
        assert "[1]" in system_prompt

        # User prompt should contain numbered context
        assert "[1]" in user_prompt
        assert "[2]" in user_prompt
        assert "AI is powerful." in user_prompt
        assert "ML basics." in user_prompt
        assert "doc.pdf" in user_prompt
        assert "What is AI?" in user_prompt

    def test_build_prompt_with_history(self, rag_engine):
        """Should include conversation history in prompt."""
        chunks = [
            SearchHit(chunk_id="c1", document_id="d1", source_file="doc.pdf",
                      title_chain="", content="content", score=0.9, chunk_index=0),
        ]
        history = [
            {"role": "user", "content": "What is ML?"},
            {"role": "assistant", "content": "ML is machine learning."},
        ]

        user_prompt, _ = rag_engine._build_prompt("Tell me more", chunks, history)

        assert "What is ML?" in user_prompt
        assert "ML is machine learning." in user_prompt
        assert "Tell me more" in user_prompt

    def test_build_prompt_without_history(self, rag_engine):
        """Should work without conversation history."""
        chunks = [
            SearchHit(chunk_id="c1", document_id="d1", source_file="doc.pdf",
                      title_chain="", content="content", score=0.9, chunk_index=0),
        ]

        user_prompt, system_prompt = rag_engine._build_prompt(
            "Question?", chunks, []
        )

        assert "对话历史" not in user_prompt
        assert "Question?" in user_prompt


# ─── Similarity Threshold Tests ────────────────────────────────────────


class TestSimilarityThreshold:
    """Tests for similarity threshold filtering."""

    @pytest.mark.asyncio
    async def test_all_below_threshold_returns_no_info(self, rag_engine, mock_search_service):
        """When all results are below threshold, should return '未找到相关信息'."""
        # All results have score < 0.5
        low_score_results = [
            make_search_result(score=0.3),
            make_search_result(score=0.2),
            make_search_result(score=0.4),
        ]
        mock_search_service.search.return_value = make_search_response(low_score_results)

        config = RAGConfig(similarity_threshold=0.5)
        result = await collect_stream(
            rag_engine.chat(
                question="test question",
                session_id=None,
                user_id="user_1",
                allowed_space_ids=["space_1"],
                config=config,
            )
        )

        assert result == NO_RELEVANT_INFO_MESSAGE

    @pytest.mark.asyncio
    async def test_some_above_threshold_proceeds(self, rag_engine, mock_search_service, mock_llm_gateway):
        """When some results are above threshold, should proceed with LLM call."""
        results = [
            make_search_result(score=0.8),
            make_search_result(score=0.3),  # Below threshold
        ]
        mock_search_service.search.return_value = make_search_response(results)

        # Mock LLM streaming
        async def mock_stream(*args, **kwargs):
            yield "Answer"
            yield " based"
            yield " on [1]."

        mock_llm_gateway.stream = mock_stream

        config = RAGConfig(similarity_threshold=0.5)
        result = await collect_stream(
            rag_engine.chat(
                question="test question",
                session_id=None,
                user_id="user_1",
                allowed_space_ids=["space_1"],
                config=config,
            )
        )

        assert "Answer" in result
        assert result != NO_RELEVANT_INFO_MESSAGE

    @pytest.mark.asyncio
    async def test_custom_threshold(self, rag_engine, mock_search_service):
        """Custom threshold should be respected."""
        results = [make_search_result(score=0.6)]
        mock_search_service.search.return_value = make_search_response(results)

        # With threshold 0.7, score 0.6 should be filtered out
        config = RAGConfig(similarity_threshold=0.7)
        result = await collect_stream(
            rag_engine.chat(
                question="test",
                session_id=None,
                user_id="user_1",
                allowed_space_ids=["space_1"],
                config=config,
            )
        )

        assert result == NO_RELEVANT_INFO_MESSAGE


# ─── LLM Timeout Tests ────────────────────────────────────────────────


class TestLLMTimeout:
    """Tests for LLM timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_returns_error_message(self, rag_engine, mock_search_service, mock_llm_gateway):
        """LLM timeout should return error message and preserve session."""
        results = [make_search_result(score=0.9)]
        mock_search_service.search.return_value = make_search_response(results)

        # Mock LLM to raise timeout
        async def mock_stream_timeout(*args, **kwargs):
            raise LLMGatewayError("Timed out after 60s", reason="timeout")
            yield  # Make it a generator  # noqa: E501

        mock_llm_gateway.stream = mock_stream_timeout

        config = RAGConfig(llm_timeout=60.0)
        result = await collect_stream(
            rag_engine.chat(
                question="test",
                session_id="session_1",
                user_id="user_1",
                allowed_space_ids=["space_1"],
                config=config,
            )
        )

        assert result == LLM_ERROR_MESSAGE

    @pytest.mark.asyncio
    async def test_non_timeout_error_propagates(self, rag_engine, mock_search_service, mock_llm_gateway):
        """Non-timeout LLM errors should propagate."""
        results = [make_search_result(score=0.9)]
        mock_search_service.search.return_value = make_search_response(results)

        # Mock LLM to raise non-timeout error
        async def mock_stream_error(*args, **kwargs):
            raise LLMGatewayError("Auth failed", reason="auth")
            yield  # noqa: E501

        mock_llm_gateway.stream = mock_stream_error

        with pytest.raises(LLMGatewayError):
            await collect_stream(
                rag_engine.chat(
                    question="test",
                    session_id=None,
                    user_id="user_1",
                    allowed_space_ids=["space_1"],
                )
            )


# ─── Session Management Tests ─────────────────────────────────────────


class TestSessionManagement:
    """Tests for Redis-based session management."""

    @pytest.mark.asyncio
    async def test_save_and_retrieve_turn(self, rag_engine, mock_redis):
        """Should save a conversation turn and retrieve it."""
        session_id = "test_session"
        await rag_engine._save_turn(
            session_id=session_id,
            user_id="user_1",
            question="What is AI?",
            answer="AI is artificial intelligence.",
            citations=[{"index": 1, "source_file": "doc.pdf"}],
        )

        history = await rag_engine._get_conversation_history(session_id)

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "What is AI?"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "AI is artificial intelligence."

    @pytest.mark.asyncio
    async def test_max_20_turns(self, rag_engine, mock_redis):
        """Should keep only the last 20 turns (40 messages)."""
        session_id = "test_session"

        # Save 25 turns
        for i in range(25):
            await rag_engine._save_turn(
                session_id=session_id,
                user_id="user_1",
                question=f"Question {i}",
                answer=f"Answer {i}",
                citations=[],
            )

        history = await rag_engine._get_conversation_history(session_id)

        # Should have max 40 messages (20 turns)
        assert len(history) <= MAX_CONVERSATION_TURNS * 2
        # Should have the most recent messages
        assert "Question 24" in history[-2]["content"]
        assert "Answer 24" in history[-1]["content"]

    @pytest.mark.asyncio
    async def test_session_expiration(self, rag_engine, mock_redis):
        """Session should be marked expired after 30 min inactivity."""
        session_id = "expired_session"
        key = f"{SESSION_KEY_PREFIX}{session_id}"

        # Set last_active to 31 minutes ago
        expired_time = str(time.time() - (SESSION_TTL_SECONDS + 60))
        mock_redis._store[key] = {
            "user_id": "user_1",
            "messages": "[]",
            "last_active": expired_time,
            "session_id": session_id,
        }

        is_expired = await rag_engine._is_session_expired(session_id)
        assert is_expired is True

    @pytest.mark.asyncio
    async def test_session_not_expired(self, rag_engine, mock_redis):
        """Session should not be expired within 30 min."""
        session_id = "active_session"
        key = f"{SESSION_KEY_PREFIX}{session_id}"

        # Set last_active to 5 minutes ago
        recent_time = str(time.time() - 300)
        mock_redis._store[key] = {
            "user_id": "user_1",
            "messages": "[]",
            "last_active": recent_time,
            "session_id": session_id,
        }

        is_expired = await rag_engine._is_session_expired(session_id)
        assert is_expired is False

    @pytest.mark.asyncio
    async def test_new_session_not_expired(self, rag_engine, mock_redis):
        """A non-existent session should not be considered expired."""
        is_expired = await rag_engine._is_session_expired("new_session")
        assert is_expired is False

    @pytest.mark.asyncio
    async def test_get_session(self, rag_engine, mock_redis):
        """Should retrieve session metadata."""
        session_id = "test_session"
        key = f"{SESSION_KEY_PREFIX}{session_id}"
        now = str(time.time())

        mock_redis._store[key] = {
            "user_id": "user_1",
            "messages": "[]",
            "last_active": now,
            "session_id": session_id,
        }

        session = await rag_engine.get_session(session_id)

        assert session is not None
        assert session["session_id"] == session_id
        assert session["user_id"] == "user_1"
        assert session["is_expired"] is False

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, rag_engine, mock_redis):
        """Should return None for non-existent session."""
        session = await rag_engine.get_session("nonexistent")
        assert session is None

    @pytest.mark.asyncio
    async def test_get_user_sessions(self, rag_engine, mock_redis):
        """Should return all active sessions for a user."""
        now = str(time.time())

        # Create 2 sessions for user_1
        mock_redis._store[f"{SESSION_KEY_PREFIX}session_1"] = {
            "user_id": "user_1",
            "messages": "[]",
            "last_active": now,
            "session_id": "session_1",
        }
        mock_redis._store[f"{SESSION_KEY_PREFIX}session_2"] = {
            "user_id": "user_1",
            "messages": "[]",
            "last_active": now,
            "session_id": "session_2",
        }
        # Create 1 session for user_2
        mock_redis._store[f"{SESSION_KEY_PREFIX}session_3"] = {
            "user_id": "user_2",
            "messages": "[]",
            "last_active": now,
            "session_id": "session_3",
        }

        sessions = await rag_engine.get_user_sessions("user_1")

        assert len(sessions) == 2
        session_ids = [s["session_id"] for s in sessions]
        assert "session_1" in session_ids
        assert "session_2" in session_ids


# ─── Full Chat Flow Tests ─────────────────────────────────────────────


class TestChatFlow:
    """Integration tests for the full RAG chat flow."""

    @pytest.mark.asyncio
    async def test_full_chat_flow(self, rag_engine, mock_search_service, mock_llm_gateway, mock_redis):
        """Full chat flow: retrieve → prompt → stream → save."""
        # Setup search results
        results = [
            make_search_result(
                chunk_id="chunk_1",
                document_id="doc_1",
                content="AI is artificial intelligence.",
                score=0.9,
                source_file="ai_guide.pdf",
                title_chain="Introduction",
            )
        ]
        mock_search_service.search.return_value = make_search_response(results)

        # Setup LLM streaming
        async def mock_stream(*args, **kwargs):
            yield "AI is "
            yield "artificial intelligence [1]."

        mock_llm_gateway.stream = mock_stream

        # Execute chat
        result = await collect_stream(
            rag_engine.chat(
                question="What is AI?",
                session_id="session_1",
                user_id="user_1",
                allowed_space_ids=["space_1"],
            )
        )

        assert result == "AI is artificial intelligence [1]."

        # Verify search was called
        mock_search_service.search.assert_called_once()

        # Verify session was saved
        history = await rag_engine._get_conversation_history("session_1")
        assert len(history) == 2
        assert history[0]["content"] == "What is AI?"
        assert history[1]["content"] == "AI is artificial intelligence [1]."

    @pytest.mark.asyncio
    async def test_chat_creates_new_session(self, rag_engine, mock_search_service, mock_llm_gateway, mock_redis):
        """Chat with no session_id should create a new session."""
        results = [make_search_result(score=0.9)]
        mock_search_service.search.return_value = make_search_response(results)

        async def mock_stream(*args, **kwargs):
            yield "Response"

        mock_llm_gateway.stream = mock_stream

        result = await collect_stream(
            rag_engine.chat(
                question="test",
                session_id=None,
                user_id="user_1",
                allowed_space_ids=["space_1"],
            )
        )

        assert result == "Response"

    @pytest.mark.asyncio
    async def test_chat_with_expired_session_creates_new(self, rag_engine, mock_search_service, mock_llm_gateway, mock_redis):
        """Chat with expired session should create a new session."""
        # Create an expired session
        expired_time = str(time.time() - (SESSION_TTL_SECONDS + 60))
        mock_redis._store[f"{SESSION_KEY_PREFIX}old_session"] = {
            "user_id": "user_1",
            "messages": json.dumps([
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
            ]),
            "last_active": expired_time,
            "session_id": "old_session",
        }

        results = [make_search_result(score=0.9)]
        mock_search_service.search.return_value = make_search_response(results)

        async def mock_stream(*args, **kwargs):
            yield "New response"

        mock_llm_gateway.stream = mock_stream

        result = await collect_stream(
            rag_engine.chat(
                question="new question",
                session_id="old_session",
                user_id="user_1",
                allowed_space_ids=["space_1"],
            )
        )

        assert result == "New response"

    @pytest.mark.asyncio
    async def test_top_k_clamping(self, rag_engine, mock_search_service, mock_llm_gateway):
        """Top-K should be clamped to valid range."""
        results = [make_search_result(score=0.9)]
        mock_search_service.search.return_value = make_search_response(results)

        async def mock_stream(*args, **kwargs):
            yield "OK"

        mock_llm_gateway.stream = mock_stream

        # top_k > MAX should be clamped
        config = RAGConfig(top_k=100)
        await collect_stream(
            rag_engine.chat(
                question="test",
                session_id=None,
                user_id="user_1",
                allowed_space_ids=["space_1"],
                config=config,
            )
        )

        # Verify search was called with clamped page_size
        call_kwargs = mock_search_service.search.call_args
        assert call_kwargs.kwargs.get("page_size", call_kwargs[1].get("page_size", 0)) <= MAX_TOP_K


# ─── RAG API Schema Tests ─────────────────────────────────────────────


class TestRAGAPISchemas:
    """Tests for RAG API request/response schemas.

    These tests import from app.api.rag which requires database drivers.
    They are skipped if asyncpg is not available in the test environment.
    """

    @pytest.fixture(autouse=True)
    def _skip_if_no_asyncpg(self):
        """Skip these tests if asyncpg is not installed."""
        pytest.importorskip("asyncpg")

    def test_chat_request_defaults(self):
        """ChatRequest should have correct defaults."""
        from app.api.rag import ChatRequest

        req = ChatRequest(question="What is AI?")
        assert req.question == "What is AI?"
        assert req.session_id is None
        assert req.top_k == DEFAULT_TOP_K
        assert req.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD
        assert req.llm_timeout == DEFAULT_LLM_TIMEOUT
        assert req.model is None
        assert req.temperature == 0.1

    def test_chat_request_custom_values(self):
        """ChatRequest should accept custom values."""
        from app.api.rag import ChatRequest

        req = ChatRequest(
            question="test",
            session_id="session_123",
            top_k=10,
            similarity_threshold=0.7,
            llm_timeout=30.0,
            model="gpt-4o",
            temperature=0.5,
        )
        assert req.top_k == 10
        assert req.similarity_threshold == 0.7
        assert req.llm_timeout == 30.0
        assert req.model == "gpt-4o"

    def test_session_info_schema(self):
        """SessionInfo holds session metadata aligned to frontend (id/last_active_at/preview)."""
        from app.api.rag import SessionInfo

        info = SessionInfo(
            id="s1",
            last_active_at="2026-05-17T15:00:00+00:00",
            preview="第一条用户问题",
        )
        assert info.id == "s1"
        assert info.preview == "第一条用户问题"

    def test_message_item_schema(self):
        """MessageItem should hold message data."""
        from app.api.rag import MessageItem

        msg = MessageItem(
            role="assistant",
            content="Hello",
            citations=[{"index": 1, "source_file": "doc.pdf"}],
        )
        assert msg.role == "assistant"
        assert msg.content == "Hello"
        assert len(msg.citations) == 1
