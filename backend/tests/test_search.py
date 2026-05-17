"""Unit tests for the composite search engine.

Tests cover:
- BM25 retriever (OpenSearch query with permission filtering)
- Dense vector retriever (Qdrant search with Pre-Filtering)
- Sparse vector retriever (Qdrant sparse search with Pre-Filtering)
- Permission filter construction (Qdrant and OpenSearch filters)
- RRF fusion algorithm (k=60, merge, dedup, top 100)
- Cross-Encoder reranking (fallback scoring)
- Timeout degradation (skip timed-out retrievers)
- Result formatting (score 0-1, highlight 200 chars)
- Search API endpoint (pagination, validation)
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.search_service import (
    DEFAULT_PAGE_SIZE,
    HIGHLIGHT_MAX_CHARS,
    MAX_PAGE_SIZE,
    RERANK_TOP_N,
    RETRIEVER_TIMEOUT,
    RRF_CANDIDATE_LIMIT,
    RRF_K,
    TOP_K_PER_RETRIEVER,
    SearchHit,
    SearchResponse,
    SearchResult,
    SearchService,
)


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    service = AsyncMock()
    service.embed_query = AsyncMock(return_value=MagicMock(
        dense_vector=[0.1] * 1024,
        sparse_indices=[1, 5, 10, 20],
        sparse_values=[0.5, 0.3, 0.8, 0.2],
    ))
    return service


@pytest.fixture
def search_service(mock_embedding_service):
    """Create a SearchService with mocked embedding service."""
    return SearchService(embedding_service=mock_embedding_service)


def make_search_hit(
    chunk_id: str | None = None,
    document_id: str | None = None,
    content: str = "test content",
    score: float = 1.0,
    chunk_index: int = 0,
    title_chain: str = "Section > Subsection",
    source_file: str = "test.pdf",
    space_id: str = "",
) -> SearchHit:
    """Helper to create a SearchHit for testing."""
    return SearchHit(
        chunk_id=chunk_id or str(uuid.uuid4()),
        document_id=document_id or str(uuid.uuid4()),
        space_id=space_id or str(uuid.uuid4()),
        chunk_index=chunk_index,
        title_chain=title_chain,
        source_file=source_file,
        content=content,
        score=score,
    )


# ─── Permission Filter Tests ──────────────────────────────────────────


class TestPermissionFilterConstruction:
    """Tests for permission filter building logic."""

    def test_build_qdrant_filter(self, search_service):
        """Qdrant filter should include user_id and space_ids."""
        user_id = str(uuid.uuid4())
        space_ids = [str(uuid.uuid4()), str(uuid.uuid4())]

        result = search_service._build_qdrant_filter(user_id, space_ids)

        assert "should" in result
        assert len(result["should"]) == 2

        # First condition: match user_id in allowed_user_ids
        assert result["should"][0]["key"] == "allowed_user_ids"
        assert result["should"][0]["match"]["value"] == user_id

        # Second condition: match space_id in allowed spaces
        assert result["should"][1]["key"] == "space_id"
        assert result["should"][1]["match"]["any"] == space_ids

    def test_build_opensearch_filter(self, search_service):
        """OpenSearch filter should use bool/should with term filters."""
        user_id = str(uuid.uuid4())
        space_ids = [str(uuid.uuid4()), str(uuid.uuid4())]

        result = search_service._build_opensearch_filter(user_id, space_ids)

        assert "bool" in result
        assert "should" in result["bool"]
        assert result["bool"]["minimum_should_match"] == 1

        should_clauses = result["bool"]["should"]
        assert len(should_clauses) == 2

        # First clause: term filter on allowed_user_ids
        assert should_clauses[0] == {"term": {"allowed_user_ids": user_id}}

        # Second clause: terms filter on space_id
        assert should_clauses[1] == {"terms": {"space_id": space_ids}}

    def test_build_qdrant_filter_empty_spaces(self, search_service):
        """Qdrant filter with empty space list should still include user_id."""
        user_id = str(uuid.uuid4())
        result = search_service._build_qdrant_filter(user_id, [])

        assert result["should"][0]["match"]["value"] == user_id
        assert result["should"][1]["match"]["any"] == []

    def test_build_opensearch_filter_single_space(self, search_service):
        """OpenSearch filter with single space should work correctly."""
        user_id = str(uuid.uuid4())
        space_id = str(uuid.uuid4())

        result = search_service._build_opensearch_filter(user_id, [space_id])

        should_clauses = result["bool"]["should"]
        assert should_clauses[1] == {"terms": {"space_id": [space_id]}}


# ─── RRF Fusion Tests ──────────────────────────────────────────────────


class TestRRFFusion:
    """Tests for the RRF fusion algorithm."""

    def test_rrf_single_retriever(self, search_service):
        """RRF with single retriever should rank by original order."""
        hits = [make_search_hit(chunk_id=f"chunk_{i}") for i in range(5)]
        results = search_service._rrf_fusion([hits])

        assert len(results) == 5
        # First result should have highest RRF score
        assert results[0].score > results[1].score

    def test_rrf_multiple_retrievers_overlap(self, search_service):
        """RRF should boost documents appearing in multiple retrievers."""
        shared_id = "shared_chunk"
        unique_id_1 = "unique_1"
        unique_id_2 = "unique_2"

        retriever_1 = [
            make_search_hit(chunk_id=shared_id, score=0.9),
            make_search_hit(chunk_id=unique_id_1, score=0.8),
        ]
        retriever_2 = [
            make_search_hit(chunk_id=shared_id, score=0.85),
            make_search_hit(chunk_id=unique_id_2, score=0.7),
        ]

        results = search_service._rrf_fusion([retriever_1, retriever_2])

        # Shared chunk should be ranked first (appears in both)
        assert results[0].chunk_id == shared_id
        # Its RRF score should be sum of both contributions
        expected_score = 1.0 / (RRF_K + 1) + 1.0 / (RRF_K + 1)
        assert abs(results[0].score - expected_score) < 1e-6

    def test_rrf_deduplication(self, search_service):
        """RRF should deduplicate chunks appearing in multiple retrievers."""
        shared_id = "shared_chunk"

        retriever_1 = [make_search_hit(chunk_id=shared_id)]
        retriever_2 = [make_search_hit(chunk_id=shared_id)]
        retriever_3 = [make_search_hit(chunk_id=shared_id)]

        results = search_service._rrf_fusion([retriever_1, retriever_2, retriever_3])

        # Should only appear once
        assert len(results) == 1
        assert results[0].chunk_id == shared_id

    def test_rrf_score_formula(self, search_service):
        """RRF score should follow formula: 1/(k + rank)."""
        hits = [make_search_hit(chunk_id=f"chunk_{i}") for i in range(3)]
        results = search_service._rrf_fusion([hits])

        # Rank 1: 1/(60+1) = 0.01639...
        expected_score_rank1 = 1.0 / (RRF_K + 1)
        assert abs(results[0].score - expected_score_rank1) < 1e-6

        # Rank 2: 1/(60+2) = 0.01613...
        expected_score_rank2 = 1.0 / (RRF_K + 2)
        assert abs(results[1].score - expected_score_rank2) < 1e-6

    def test_rrf_limit_100_candidates(self, search_service):
        """RRF should return at most 100 candidates."""
        # Create 60 unique hits per retriever (180 total unique)
        retriever_1 = [make_search_hit(chunk_id=f"r1_{i}") for i in range(60)]
        retriever_2 = [make_search_hit(chunk_id=f"r2_{i}") for i in range(60)]
        retriever_3 = [make_search_hit(chunk_id=f"r3_{i}") for i in range(60)]

        results = search_service._rrf_fusion([retriever_1, retriever_2, retriever_3])

        assert len(results) <= RRF_CANDIDATE_LIMIT

    def test_rrf_empty_results(self, search_service):
        """RRF with empty results should return empty list."""
        results = search_service._rrf_fusion([])
        assert results == []

    def test_rrf_one_empty_retriever(self, search_service):
        """RRF should handle one empty retriever gracefully."""
        hits = [make_search_hit(chunk_id=f"chunk_{i}") for i in range(3)]
        results = search_service._rrf_fusion([hits, []])

        assert len(results) == 3


# ─── Cross-Encoder Reranking Tests ─────────────────────────────────────


class TestCrossEncoderRerank:
    """Tests for Cross-Encoder reranking."""

    @pytest.mark.asyncio
    async def test_fallback_rerank_scores(self, search_service):
        """Fallback scoring should rank relevant candidates higher than irrelevant.

        新算法 (token + bigram 取最大值) 兼容中英文混合, 但具体数值跟 token-only
        不同; 这里只验证排序而不锚定具体分数。
        """
        query = "machine learning algorithms"
        candidates = [
            make_search_hit(content="machine learning is a subset of AI"),
            make_search_hit(content="algorithms for sorting data"),
            make_search_hit(content="zzzzzz qqqqqq xxxxxx"),  # 跟 query 完全无 bigram 重叠
        ]

        scores = search_service._fallback_rerank_scores(query, candidates)

        # 命中两个 token (machine + learning) 的应该高于命中一个 token (algorithms)
        assert scores[0] > scores[1]
        # 完全不相关的应该最低
        assert scores[1] > scores[2]
        # 完全无重叠的得 0
        assert scores[2] == 0.0

    @pytest.mark.asyncio
    async def test_rerank_normalizes_scores(self, search_service):
        """Reranking should normalize scores to 0-1 range."""
        candidates = [
            make_search_hit(content="relevant content about search"),
            make_search_hit(content="another relevant document"),
            make_search_hit(content="completely unrelated"),
        ]

        # Mock the cross-encoder to use fallback
        with patch.object(
            search_service,
            "_compute_cross_encoder_scores",
            side_effect=Exception("Model not available"),
        ):
            results = await search_service._cross_encoder_rerank(
                "search", candidates
            )

        # All scores should be between 0 and 1
        for r in results:
            assert 0.0 <= r.score <= 1.0

    @pytest.mark.asyncio
    async def test_rerank_empty_candidates(self, search_service):
        """Reranking empty candidates should return empty list."""
        results = await search_service._cross_encoder_rerank("query", [])
        assert results == []

    @pytest.mark.asyncio
    async def test_rerank_sorts_by_score(self, search_service):
        """Reranking should sort candidates by score descending."""
        candidates = [
            make_search_hit(content="low relevance"),
            make_search_hit(content="search engine optimization"),
            make_search_hit(content="search algorithms and data structures"),
        ]

        with patch.object(
            search_service,
            "_compute_cross_encoder_scores",
            return_value=[0.1, 0.8, 0.9],
        ):
            results = await search_service._cross_encoder_rerank(
                "search", candidates
            )

        # Should be sorted by score descending
        assert results[0].score >= results[1].score
        assert results[1].score >= results[2].score


# ─── Timeout Degradation Tests ─────────────────────────────────────────


class TestTimeoutDegradation:
    """Tests for search timeout degradation."""

    @pytest.mark.asyncio
    async def test_skip_timed_out_retriever(self, search_service):
        """Should skip retrievers that exceed 3-second timeout."""
        fast_results = [make_search_hit(chunk_id="fast_1")]

        async def slow_retriever(*args, **kwargs):
            await asyncio.sleep(10)  # Will timeout
            return [make_search_hit(chunk_id="slow_1")]

        async def fast_retriever(*args, **kwargs):
            return fast_results

        with patch.object(
            search_service, "_bm25_recall", side_effect=slow_retriever
        ), patch.object(
            search_service, "_dense_recall", side_effect=fast_retriever
        ), patch.object(
            search_service, "_sparse_recall", side_effect=fast_retriever
        ):
            results = await search_service._multi_recall(
                query="test",
                dense_vector=[0.1] * 1024,
                sparse_indices=[1, 2],
                sparse_values=[0.5, 0.3],
                qdrant_filter={},
                opensearch_filter={},
            )

        # Should have results from 2 fast retrievers, BM25 timed out
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_all_retrievers_succeed(self, search_service):
        """All retrievers returning within timeout should be included."""
        hits = [make_search_hit()]

        async def fast_retriever(*args, **kwargs):
            return hits

        with patch.object(
            search_service, "_bm25_recall", side_effect=fast_retriever
        ), patch.object(
            search_service, "_dense_recall", side_effect=fast_retriever
        ), patch.object(
            search_service, "_sparse_recall", side_effect=fast_retriever
        ):
            results = await search_service._multi_recall(
                query="test",
                dense_vector=[0.1] * 1024,
                sparse_indices=[1],
                sparse_values=[0.5],
                qdrant_filter={},
                opensearch_filter={},
            )

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_all_retrievers_fail(self, search_service):
        """If all retrievers fail, should return empty results."""

        async def failing_retriever(*args, **kwargs):
            raise RuntimeError("Connection failed")

        with patch.object(
            search_service, "_bm25_recall", side_effect=failing_retriever
        ), patch.object(
            search_service, "_dense_recall", side_effect=failing_retriever
        ), patch.object(
            search_service, "_sparse_recall", side_effect=failing_retriever
        ):
            results = await search_service._multi_recall(
                query="test",
                dense_vector=[0.1] * 1024,
                sparse_indices=[1],
                sparse_values=[0.5],
                qdrant_filter={},
                opensearch_filter={},
            )

        assert len(results) == 0


# ─── Result Formatting Tests ───────────────────────────────────────────


class TestResultFormatting:
    """Tests for search result formatting."""

    def test_format_results_score_clamped(self, search_service):
        """Scores should be clamped to 0-1 range."""
        candidates = [
            make_search_hit(content="test", score=1.5),
            make_search_hit(content="test", score=-0.5),
        ]

        results = search_service._format_results(candidates, "test")

        assert results[0].score == 1.0
        assert results[1].score == 0.0

    def test_format_results_preserves_metadata(self, search_service):
        """Formatting should preserve all metadata fields."""
        hit = make_search_hit(
            chunk_id="chunk_123",
            document_id="doc_456",
            chunk_index=5,
            title_chain="Chapter 1 > Section 2",
            source_file="report.pdf",
            content="Some content here",
            score=0.85,
        )

        results = search_service._format_results([hit], "content")

        assert len(results) == 1
        assert results[0].chunk_id == "chunk_123"
        assert results[0].document_id == "doc_456"
        assert results[0].chunk_index == 5
        assert results[0].title_chain == "Chapter 1 > Section 2"
        assert results[0].source_file == "report.pdf"
        assert results[0].score == 0.85

    def test_highlight_short_content(self, search_service):
        """Short content with matching query gets ``<mark>`` wrapping."""
        content = "Short text"
        highlight = search_service._generate_highlight(content, "text")
        # 命中关键词 'text' 会被 ``<mark>`` 包裹
        assert highlight == "Short <mark>text</mark>"

    def test_highlight_short_content_no_match(self, search_service):
        """Short content with no matching query returns content as-is."""
        content = "Short text"
        highlight = search_service._generate_highlight(content, "nonexistent")
        assert highlight == content

    def test_highlight_max_200_chars(self, search_service):
        """Highlight should not exceed 200 characters."""
        content = "x" * 500
        highlight = search_service._generate_highlight(content, "x")
        assert len(highlight) <= HIGHLIGHT_MAX_CHARS

    def test_highlight_empty_content(self, search_service):
        """Empty content should return empty highlight."""
        highlight = search_service._generate_highlight("", "query")
        assert highlight == ""

    def test_highlight_finds_relevant_section(self, search_service):
        """Highlight should find the section containing query terms."""
        content = "A" * 300 + " machine learning " + "B" * 300
        highlight = search_service._generate_highlight(content, "machine learning")

        # The highlight should contain the query terms
        assert "machine" in highlight.lower() or "learning" in highlight.lower()


# ─── Full Search Integration Tests (Mocked Backends) ──────────────────


class TestSearchIntegration:
    """Integration tests for the full search flow with mocked backends."""

    @pytest.mark.asyncio
    async def test_search_empty_spaces(self, search_service):
        """Search with no accessible spaces should return empty results."""
        response = await search_service.search(
            query="test query",
            user_id=str(uuid.uuid4()),
            allowed_space_ids=[],
            page=1,
            page_size=10,
        )

        assert response.results == []
        assert response.total == 0

    @pytest.mark.asyncio
    async def test_search_pagination(self, search_service):
        """Search should respect pagination parameters."""
        # Mock retrievers to return results
        hits = [make_search_hit(chunk_id=f"chunk_{i}") for i in range(30)]

        async def mock_retriever(*args, **kwargs):
            return hits

        with patch.object(
            search_service, "_bm25_recall", side_effect=mock_retriever
        ), patch.object(
            search_service, "_dense_recall", side_effect=mock_retriever
        ), patch.object(
            search_service, "_sparse_recall", side_effect=mock_retriever
        ), patch.object(
            search_service, "_cross_encoder_rerank",
            new_callable=lambda: AsyncMock(side_effect=lambda q, c: c),
        ):
            response = await search_service.search(
                query="test",
                user_id=str(uuid.uuid4()),
                allowed_space_ids=[str(uuid.uuid4())],
                page=1,
                page_size=5,
            )

        assert response.page == 1
        assert response.page_size == 5
        assert len(response.results) == 5

    @pytest.mark.asyncio
    async def test_search_max_page_size(self, search_service):
        """Page size should be capped at 50."""
        response = await search_service.search(
            query="test",
            user_id=str(uuid.uuid4()),
            allowed_space_ids=[],
            page=1,
            page_size=100,  # Exceeds max
        )

        assert response.page_size == MAX_PAGE_SIZE

    @pytest.mark.asyncio
    async def test_search_page_minimum(self, search_service):
        """Page number should be at least 1."""
        response = await search_service.search(
            query="test",
            user_id=str(uuid.uuid4()),
            allowed_space_ids=[],
            page=0,  # Below minimum
            page_size=10,
        )

        assert response.page == 1

    @pytest.mark.asyncio
    async def test_search_full_flow(self, search_service):
        """Full search flow should produce valid results."""
        user_id = str(uuid.uuid4())
        space_id = str(uuid.uuid4())

        bm25_hits = [
            make_search_hit(chunk_id="bm25_1", content="BM25 result about AI", score=5.0),
            make_search_hit(chunk_id="shared", content="Shared result about ML", score=4.0),
        ]
        dense_hits = [
            make_search_hit(chunk_id="dense_1", content="Dense vector result", score=0.9),
            make_search_hit(chunk_id="shared", content="Shared result about ML", score=0.85),
        ]
        sparse_hits = [
            make_search_hit(chunk_id="sparse_1", content="Sparse result", score=0.7),
        ]

        with patch.object(
            search_service, "_bm25_recall", return_value=bm25_hits
        ), patch.object(
            search_service, "_dense_recall", return_value=dense_hits
        ), patch.object(
            search_service, "_sparse_recall", return_value=sparse_hits
        ), patch.object(
            search_service, "_cross_encoder_rerank",
            new_callable=lambda: AsyncMock(side_effect=lambda q, c: c),
        ):
            response = await search_service.search(
                query="AI machine learning",
                user_id=user_id,
                allowed_space_ids=[space_id],
                page=1,
                page_size=10,
            )

        # Should have results
        assert response.total > 0
        assert len(response.results) > 0

        # All scores should be 0-1
        for r in response.results:
            assert 0.0 <= r.score <= 1.0

        # All highlights should be <= 200 chars
        for r in response.results:
            assert len(r.highlight) <= HIGHLIGHT_MAX_CHARS


# ─── Qdrant Filter Conversion Tests ───────────────────────────────────


class TestQdrantFilterConversion:
    """Tests for converting filter dict to Qdrant Filter object."""

    def test_dict_to_qdrant_filter_with_value(self, search_service):
        """Should convert value match to FieldCondition with MatchValue."""
        filter_dict = {
            "should": [
                {"key": "allowed_user_ids", "match": {"value": "user_123"}},
            ]
        }

        result = search_service._dict_to_qdrant_filter(filter_dict)

        assert result.should is not None
        assert len(result.should) == 1

    def test_dict_to_qdrant_filter_with_any(self, search_service):
        """Should convert any match to FieldCondition with MatchAny."""
        filter_dict = {
            "should": [
                {"key": "space_id", "match": {"any": ["space_1", "space_2"]}},
            ]
        }

        result = search_service._dict_to_qdrant_filter(filter_dict)

        assert result.should is not None
        assert len(result.should) == 1

    def test_dict_to_qdrant_filter_combined(self, search_service):
        """Should handle combined value and any conditions."""
        filter_dict = {
            "should": [
                {"key": "allowed_user_ids", "match": {"value": "user_123"}},
                {"key": "space_id", "match": {"any": ["space_1"]}},
            ]
        }

        result = search_service._dict_to_qdrant_filter(filter_dict)

        assert result.should is not None
        assert len(result.should) == 2


# ─── Search API Endpoint Tests ─────────────────────────────────────────


class TestSearchAPI:
    """Tests for the POST /api/search endpoint schemas and logic."""

    def test_search_response_dataclass(self):
        """SearchResponse dataclass should hold correct data."""
        results = [
            SearchResult(
                chunk_id="chunk_1",
                document_id="doc_1",
                chunk_index=0,
                title_chain="Title",
                source_file="file.pdf",
                score=0.85,
                highlight="matching text",
            )
        ]
        response = SearchResponse(
            results=results,
            total=1,
            page=1,
            page_size=10,
        )

        assert len(response.results) == 1
        assert response.total == 1
        assert response.page == 1
        assert response.page_size == 10
        assert response.results[0].score == 0.85

    def test_search_result_fields(self):
        """SearchResult should contain all required fields."""
        result = SearchResult(
            chunk_id="chunk_123",
            document_id="doc_456",
            chunk_index=3,
            title_chain="Chapter 1 > Section 2",
            source_file="report.pdf",
            score=0.92,
            highlight="relevant text snippet",
        )

        assert result.chunk_id == "chunk_123"
        assert result.document_id == "doc_456"
        assert result.chunk_index == 3
        assert result.title_chain == "Chapter 1 > Section 2"
        assert result.source_file == "report.pdf"
        assert result.score == 0.92
        assert result.highlight == "relevant text snippet"

    def test_search_response_pagination_fields(self):
        """SearchResponse should track pagination state."""
        response = SearchResponse(
            results=[],
            total=100,
            page=3,
            page_size=20,
        )

        assert response.total == 100
        assert response.page == 3
        assert response.page_size == 20
