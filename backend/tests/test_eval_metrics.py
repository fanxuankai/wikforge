"""检索质量指标单元测试（任务 25.7）。

覆盖 ``backend/eval/metrics.py`` 中 Recall@K / MRR / NDCG@K 的核心契约：
- 边界值（K=1、空 ground truth、空检索结果）
- 已知公式手算结果
- 聚合函数正确平均
"""

from __future__ import annotations

import math

import pytest

from eval.metrics import (
    EvalSample,
    dcg_at_k,
    evaluate,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


# ─── Recall@K ─────────────────────────────────────────────────────────


class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = ["a", "b", "c"]
        relevant = ["a", "b", "c"]
        assert recall_at_k(retrieved, relevant, 3) == 1.0

    def test_partial_recall(self):
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = ["a", "b", "c"]
        assert recall_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)

    def test_no_recall(self):
        assert recall_at_k(["x", "y"], ["a", "b"], 5) == 0.0

    def test_empty_relevant_returns_zero(self):
        assert recall_at_k(["a"], [], 1) == 0.0

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            recall_at_k(["a"], ["a"], 0)

    def test_recall_at_k_truncates(self):
        # k=1 时只能命中第一个，relevant 有 2 个，recall=0.5
        assert recall_at_k(["a", "b"], ["a", "b"], 1) == 0.5


# ─── MRR ──────────────────────────────────────────────────────────────


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "b"], ["a"]) == 1.0

    def test_second_position(self):
        assert reciprocal_rank(["x", "a"], ["a"]) == 0.5

    def test_no_hit(self):
        assert reciprocal_rank(["x", "y"], ["a"]) == 0.0

    def test_first_hit_wins_when_multiple(self):
        # 多个相关项时只看第一个命中位置
        assert reciprocal_rank(["x", "b", "a"], ["a", "b"]) == 0.5


# ─── DCG / NDCG ──────────────────────────────────────────────────────


class TestNDCG:
    def test_perfect_order_gives_ndcg_1(self):
        # retrieved 与理想排序一致 → ndcg = 1
        grades = {"a": 3.0, "b": 2.0, "c": 1.0}
        assert ndcg_at_k(["a", "b", "c"], grades, 3) == pytest.approx(1.0)

    def test_reversed_order_below_1(self):
        grades = {"a": 3.0, "b": 2.0, "c": 1.0}
        ndcg = ndcg_at_k(["c", "b", "a"], grades, 3)
        assert 0.0 < ndcg < 1.0

    def test_no_relevant_returns_zero(self):
        assert ndcg_at_k(["a", "b"], {}, 3) == 0.0

    def test_dcg_formula_matches_manual(self):
        # 手算：rel=[3,2,1] → DCG = 3/log2(2) + 2/log2(3) + 1/log2(4)
        grades = {"a": 3.0, "b": 2.0, "c": 1.0}
        expected = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
        assert dcg_at_k(["a", "b", "c"], grades, 3) == pytest.approx(expected)

    def test_ndcg_truncates_to_k(self):
        grades = {"a": 1.0, "b": 1.0, "c": 1.0}
        # k=1：检索到的第一个是 a，与理想等价 → 1.0
        assert ndcg_at_k(["a", "b", "c"], grades, 1) == pytest.approx(1.0)


# ─── evaluate（聚合） ─────────────────────────────────────────────────


class TestEvaluate:
    def test_average_across_samples(self):
        samples = [
            EvalSample(query_id="1", query="q1", relevant_chunk_ids=["a"]),
            EvalSample(query_id="2", query="q2", relevant_chunk_ids=["b"]),
        ]

        # 第一条命中第 1 位，第二条命中第 2 位
        def retrieve(q: str) -> list[str]:
            if q == "q1":
                return ["a", "x"]
            return ["x", "b"]

        result = evaluate(samples, retrieve, k_values=(1, 2))
        assert result.num_samples == 2
        # MRR = (1/1 + 1/2) / 2 = 0.75
        assert result.mrr == pytest.approx(0.75)
        # Recall@1: q1 命中, q2 未命中 → 0.5
        assert result.recall_at_k[1] == pytest.approx(0.5)
        # Recall@2: 两条都命中 → 1.0
        assert result.recall_at_k[2] == pytest.approx(1.0)
        # NDCG@1: q1=1, q2=0 → 平均 0.5
        assert result.ndcg_at_k[1] == pytest.approx(0.5)
        # NDCG@2: q1 第 1 位最优 → 1.0；q2 第 2 位 → 1/log2(3) / (1/log2(2)) = 1/log2(3)
        expected_q2 = (1 / math.log2(3)) / (1 / math.log2(2))
        assert result.ndcg_at_k[2] == pytest.approx((1.0 + expected_q2) / 2)

    def test_empty_samples_returns_zero_result(self):
        result = evaluate([], lambda _q: [])
        assert result.num_samples == 0
        assert result.mrr == 0.0
        assert result.recall_at_k == {}
