"""Cross-Encoder 精排（BGE-Reranker，对 Top 20 候选精排）聚焦单元测试。

对应任务 14.6：实现 Cross-Encoder 精排（BGE-Reranker，对 Top 20 候选精排）。

设计依据：
- requirements.md 需求 6.3：候选集生成后，Search_Engine 须使用 Cross_Encoder
  重排序模型对候选集中前 20 个文档块进行精排，并按精排分数降序排列最终结果。
- design.md "Search Service - Cross-Encoder 精排"：
      candidates = self.rrf_fusion(recalls, k=60)
      results = await self.rerank(candidates[:20])
- 实现位于 ``app/services/search_service.py``：
  ``SearchService._cross_encoder_rerank`` + ``_compute_cross_encoder_scores``，
  使用 sentence_transformers ``CrossEncoder("BAAI/bge-reranker-base")``，
  ImportError / 模型异常时降级为关键词重叠启发式 ``_fallback_rerank_scores``。

测试范围（任务 14.6 审计要点）：
- 仅对 Top RERANK_TOP_N (=20) 候选执行精排；其余候选保留 RRF 顺序追加在末尾
- 重排后分数归一化到 [0, 1]
- 输出按重排分数降序排列
- 空候选集返回 []
- ``sentence_transformers`` 不可用时安全降级
- 降级路径使用关键词重叠启发式
"""

from __future__ import annotations

import math
import sys
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import HealthCheck, given, settings as hyp_settings, strategies as st

from app.services.search_service import (
    RERANK_TOP_N,
    RRF_CANDIDATE_LIMIT,
    SearchHit,
    SearchService,
)


# ─── Fixtures & Helpers ────────────────────────────────────────────────


@pytest.fixture
def mock_embedding_service() -> AsyncMock:
    """无副作用的 EmbeddingService（精排逻辑不直接依赖它）。"""
    service = AsyncMock()
    service.embed_query = AsyncMock(
        return_value=MagicMock(
            dense_vector=[0.0] * 1024,
            sparse_indices=[1, 2, 3],
            sparse_values=[0.1, 0.2, 0.3],
        )
    )
    return service


@pytest.fixture
def search_service(mock_embedding_service: AsyncMock) -> SearchService:
    """构造 SearchService（embedding 已 mock，精排是纯计算 + 单一外部依赖）。"""
    return SearchService(embedding_service=mock_embedding_service)


def _make_hit(
    *,
    chunk_id: str | None = None,
    content: str = "default content",
    score: float = 1.0,
) -> SearchHit:
    """构造一个 SearchHit（仅填充精排关心的字段）。"""
    return SearchHit(
        chunk_id=chunk_id or str(uuid.uuid4()),
        document_id=str(uuid.uuid4()),
        space_id=str(uuid.uuid4()),
        chunk_index=0,
        title_chain="A > B",
        source_file="file.pdf",
        content=content,
        score=score,
    )


# ─── 常量保证 ─────────────────────────────────────────────────────────


class TestRerankConstants:
    """需求 6.3 显式约定 Top 20，须在常量层固定。"""

    def test_rerank_top_n_equals_20(self) -> None:
        """RERANK_TOP_N 必须严格等于 20（需求 6.3）。"""
        assert RERANK_TOP_N == 20


# ─── 边界 ─────────────────────────────────────────────────────────────


class TestRerankBoundary:
    """空候选集与单候选场景。"""

    @pytest.mark.asyncio
    async def test_empty_candidates_returns_empty(
        self, search_service: SearchService
    ) -> None:
        """空候选集应直接返回 []，不应触发任何模型调用。"""
        with patch.object(
            search_service, "_compute_cross_encoder_scores"
        ) as mocked:
            results = await search_service._cross_encoder_rerank("query", [])

        assert results == []
        mocked.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_candidate_score_in_range(
        self, search_service: SearchService
    ) -> None:
        """单候选场景下不会触发除零，分数应被夹到 [0, 1]。"""
        candidate = _make_hit(content="only one")
        with patch.object(
            search_service,
            "_compute_cross_encoder_scores",
            AsyncMock(return_value=[2.5]),
        ):
            results = await search_service._cross_encoder_rerank(
                "any", [candidate]
            )

        assert len(results) == 1
        assert 0.0 <= results[0].score <= 1.0


# ─── 排序 ─────────────────────────────────────────────────────────────


class TestRerankOrdering:
    """精排后必须按分数降序排列。"""

    @pytest.mark.asyncio
    async def test_sorted_descending_by_reranked_score(
        self, search_service: SearchService
    ) -> None:
        """精排返回顺序应严格非升序（需求 6.3）。"""
        candidates = [
            _make_hit(chunk_id="low"),
            _make_hit(chunk_id="high"),
            _make_hit(chunk_id="mid"),
        ]
        # 故意用与输入顺序不一致的分数，验证确实进行了重排
        with patch.object(
            search_service,
            "_compute_cross_encoder_scores",
            AsyncMock(return_value=[0.1, 0.9, 0.5]),
        ):
            results = await search_service._cross_encoder_rerank(
                "q", candidates
            )

        assert [h.chunk_id for h in results] == ["high", "mid", "low"]
        for prev, nxt in zip(results, results[1:]):
            assert prev.score >= nxt.score

    @pytest.mark.asyncio
    async def test_reorders_relative_to_rrf_order(
        self, search_service: SearchService
    ) -> None:
        """当精排分数与 RRF 输入顺序不一致时，输出顺序应跟随精排。"""
        # RRF 给出的顺序是 a, b, c；精排却给 c 最高分
        a, b, c = (
            _make_hit(chunk_id="a"),
            _make_hit(chunk_id="b"),
            _make_hit(chunk_id="c"),
        )
        with patch.object(
            search_service,
            "_compute_cross_encoder_scores",
            AsyncMock(return_value=[0.1, 0.4, 0.9]),
        ):
            results = await search_service._cross_encoder_rerank(
                "q", [a, b, c]
            )

        assert [h.chunk_id for h in results] == ["c", "b", "a"]


# ─── 分数归一化 ───────────────────────────────────────────────────────


class TestRerankNormalization:
    """精排分数应归一化到 [0, 1]。"""

    @pytest.mark.asyncio
    async def test_scores_clamped_to_unit_interval(
        self, search_service: SearchService
    ) -> None:
        """无论原始分数范围如何，最终分数都应在 [0, 1]。"""
        candidates = [_make_hit(chunk_id=f"c{i}") for i in range(5)]
        # 选择一组横跨负值与大正值的原始分数，验证 min-max 归一化
        raw_scores = [-3.0, 0.0, 1.5, 4.2, 10.0]
        with patch.object(
            search_service,
            "_compute_cross_encoder_scores",
            AsyncMock(return_value=raw_scores),
        ):
            results = await search_service._cross_encoder_rerank(
                "q", candidates
            )

        for h in results:
            assert 0.0 <= h.score <= 1.0
        # 最高分对应原始最大值，归一化后应为 1.0
        assert math.isclose(results[0].score, 1.0, abs_tol=1e-9)
        # 最低分对应原始最小值，归一化后应为 0.0
        assert math.isclose(results[-1].score, 0.0, abs_tol=1e-9)

    @pytest.mark.asyncio
    async def test_constant_scores_do_not_divide_by_zero(
        self, search_service: SearchService
    ) -> None:
        """所有候选分数相同时不应抛 ZeroDivisionError，全部分数都应 ∈ [0, 1]。"""
        candidates = [_make_hit(chunk_id=f"c{i}") for i in range(3)]
        with patch.object(
            search_service,
            "_compute_cross_encoder_scores",
            AsyncMock(return_value=[0.42, 0.42, 0.42]),
        ):
            results = await search_service._cross_encoder_rerank(
                "q", candidates
            )

        for h in results:
            assert 0.0 <= h.score <= 1.0


# ─── 仅对 Top 20 精排 ─────────────────────────────────────────────────


class TestRerankTopNSlicing:
    """``search()`` 应仅将前 RERANK_TOP_N (=20) 候选送入 ``_cross_encoder_rerank``。"""

    @pytest.mark.asyncio
    async def test_search_passes_only_top_n_to_reranker(
        self, search_service: SearchService
    ) -> None:
        """RRF 候选超过 20 时，精排只接收前 20 条；其余按原顺序追加。

        Validates: Requirements 6.3
        """
        # 构造 30 条互不重叠的候选，触发 RRF 后会保留全部 30 条排序
        retriever_hits = [
            _make_hit(chunk_id=f"c{i:02d}", content=f"content {i}")
            for i in range(30)
        ]

        async def _fake_retriever(*_args, **_kwargs):
            return retriever_hits

        # 精排桩：原样返回，便于断言传入数量
        rerank_stub = AsyncMock(side_effect=lambda _q, c: list(c))

        with patch.object(
            search_service, "_bm25_recall", side_effect=_fake_retriever
        ), patch.object(
            search_service, "_dense_recall", side_effect=_fake_retriever
        ), patch.object(
            search_service, "_sparse_recall", side_effect=_fake_retriever
        ), patch.object(
            search_service, "_cross_encoder_rerank", new=rerank_stub
        ):
            response = await search_service.search(
                query="test",
                user_id=str(uuid.uuid4()),
                allowed_space_ids=[str(uuid.uuid4())],
                page=1,
                page_size=50,
            )

        # 精排只被调用一次
        assert rerank_stub.await_count == 1
        # 第二个位置参数（candidates）长度应为 RERANK_TOP_N
        passed_candidates = rerank_stub.await_args.args[1]
        assert len(passed_candidates) == RERANK_TOP_N

        # 总结果应保留所有 RRF 候选（精排 + 未精排尾部）
        # 由于只有 30 条不同 chunk_id，且 RRF 上限是 100，应保留全部 30 条
        assert response.total == 30


# ─── 降级到关键词重叠 ─────────────────────────────────────────────────


class TestRerankFallback:
    """``sentence_transformers`` 不可用 / 模型异常时的降级路径。"""

    def test_fallback_uses_keyword_overlap(
        self, search_service: SearchService
    ) -> None:
        """``_fallback_rerank_scores`` 基于关键词与字符 bigram 重叠打分,
        排序应优先 token 命中率高的候选。
        """
        query = "machine learning algorithms"
        candidates = [
            _make_hit(content="Machine learning is a subset of AI"),  # 命中 machine + learning
            _make_hit(content="Algorithms for sorting data"),         # 命中 algorithms
            _make_hit(content="zzzzzz qqqqq xxxxxx"),                 # 完全不重叠 (无 token / bigram)
        ]

        scores = search_service._fallback_rerank_scores(query, candidates)

        assert len(scores) == 3
        # 排序: 第一条 > 第二条 > 第三条
        assert scores[0] > scores[1]
        assert scores[1] > scores[2]
        assert scores[2] == 0.0

    def test_fallback_is_case_insensitive(
        self, search_service: SearchService
    ) -> None:
        """大小写差异不影响关键词重叠匹配。"""
        scores = search_service._fallback_rerank_scores(
            "Search Engine",
            [
                _make_hit(content="search engine optimization"),
                _make_hit(content="SEARCH ENGINE algorithms"),
                _make_hit(content="zzzzzz qqqqq"),
            ],
        )
        # 前两条都完全命中 token, 应等于 1.0
        assert math.isclose(scores[0], 1.0, rel_tol=1e-9)
        assert math.isclose(scores[1], 1.0, rel_tol=1e-9)
        # 第三条完全无 token / bigram 重叠
        assert scores[2] == 0.0

    def test_fallback_empty_query_returns_zero(
        self, search_service: SearchService
    ) -> None:
        """空查询不会抛除零，所有候选得分为 0。"""
        scores = search_service._fallback_rerank_scores(
            "",
            [_make_hit(content="any content"), _make_hit(content="other")],
        )
        assert scores == [0.0, 0.0]

    @pytest.mark.asyncio
    async def test_compute_falls_back_when_sentence_transformers_missing(
        self, search_service: SearchService
    ) -> None:
        """``sentence_transformers`` 未安装（ImportError）时应安全降级到关键词重叠。

        Validates: Requirements 6.3
        """
        # 通过把 sentence_transformers 注入 sys.modules 中并触发 ImportError
        # 来精确模拟 ``from sentence_transformers import CrossEncoder`` 失败。
        # 当前环境中该包未安装，但即便未来被引入，这里的 patch 仍能复现失败路径。
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with patch.object(
                search_service,
                "_fallback_rerank_scores",
                wraps=search_service._fallback_rerank_scores,
            ) as fallback_spy:
                query = "alpha beta"
                candidates = [
                    _make_hit(content="alpha gamma"),
                    _make_hit(content="delta epsilon"),
                ]
                scores = await search_service._compute_cross_encoder_scores(
                    query, candidates
                )

        # 必须降级到关键词重叠
        fallback_spy.assert_called_once_with(query, candidates)
        # 第一条候选与 query 有 token 重叠 (alpha), 应该 > 第二条
        assert scores[0] > scores[1]

    @pytest.mark.asyncio
    async def test_rerank_recovers_when_compute_raises(
        self, search_service: SearchService
    ) -> None:
        """``_compute_cross_encoder_scores`` 抛异常时，整体仍应返回归一化结果。

        即便底层模型异常，``_cross_encoder_rerank`` 也不应抛错，而是降级到
        基于 RRF 分数的归一化输出（参见实现的 except 分支）。
        """
        candidates = [
            _make_hit(chunk_id="x", score=0.04),
            _make_hit(chunk_id="y", score=0.02),
            _make_hit(chunk_id="z", score=0.01),
        ]
        with patch.object(
            search_service,
            "_compute_cross_encoder_scores",
            AsyncMock(side_effect=RuntimeError("model down")),
        ):
            results = await search_service._cross_encoder_rerank(
                "q", candidates
            )

        assert len(results) == 3
        for h in results:
            assert 0.0 <= h.score <= 1.0
        # 最高 RRF 分数的候选归一化后应等于 1.0
        assert math.isclose(
            max(h.score for h in results), 1.0, abs_tol=1e-9
        )


# ─── Hypothesis 属性测试 ──────────────────────────────────────────────


@st.composite
def _rerank_inputs(draw) -> tuple[list[SearchHit], list[float]]:
    """生成 (候选列表, 与之等长的原始 Cross-Encoder 分数)。"""
    n = draw(st.integers(min_value=1, max_value=RERANK_TOP_N))
    candidates = [_make_hit(chunk_id=f"c{i}") for i in range(n)]
    scores = draw(
        st.lists(
            st.floats(
                min_value=-1e3,
                max_value=1e3,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=n,
            max_size=n,
        )
    )
    return candidates, scores


class TestRerankProperties:
    """覆盖输入空间，验证精排归一化与排序的不变量。

    Validates: Requirements 6.3
    """

    @given(inputs=_rerank_inputs())
    @hyp_settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @pytest.mark.asyncio
    async def test_property_normalized_and_sorted(
        self,
        search_service: SearchService,
        inputs: tuple[list[SearchHit], list[float]],
    ) -> None:
        """属性：所有重排分数 ∈ [0, 1] 且按降序排列。

        Validates: Requirements 6.3
        """
        candidates, raw_scores = inputs
        with patch.object(
            search_service,
            "_compute_cross_encoder_scores",
            AsyncMock(return_value=raw_scores),
        ):
            results = await search_service._cross_encoder_rerank(
                "q", candidates
            )

        # 1) 长度守恒
        assert len(results) == len(candidates)

        # 2) 所有分数在 [0, 1] 区间
        for h in results:
            assert 0.0 <= h.score <= 1.0

        # 3) 严格非升序（允许并列）
        for prev, nxt in zip(results, results[1:]):
            assert prev.score >= nxt.score - 1e-12
