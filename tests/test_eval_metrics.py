"""Tests for retrieval-quality metrics.

Pure functions over ranked id lists — no index, no I/O. These are the
scoring half of the LongMemEval harness; getting them wrong would make
every benchmark number meaningless, so they are pinned by hand-computed
expectations rather than by golden files.
"""
from __future__ import annotations

import math

import pytest

import eval_metrics


class TestRecallAnyAtK:
    def test_gold_at_first_rank_scores_one(self):
        assert eval_metrics.recall_any_at_k(["a", "b", "c"], {"a"}, k=5) == 1.0

    def test_gold_outside_cutoff_scores_zero(self):
        # gold is at rank 3; k=2 must not see it
        assert eval_metrics.recall_any_at_k(["a", "b", "gold"], {"gold"}, k=2) == 0.0

    def test_any_one_of_several_gold_sessions_suffices(self):
        # recall_ANY@k: one hit is a hit, even when 3 gold sessions exist
        assert eval_metrics.recall_any_at_k(["x", "g2"], {"g1", "g2", "g3"}, k=5) == 1.0

    def test_no_gold_in_ranking_scores_zero(self):
        assert eval_metrics.recall_any_at_k(["a", "b"], {"gold"}, k=5) == 0.0

    def test_empty_ranking_scores_zero(self):
        assert eval_metrics.recall_any_at_k([], {"gold"}, k=5) == 0.0

    def test_empty_gold_set_is_rejected(self):
        # A question with no gold session is a dataset bug, not a 0.0 score —
        # silently scoring it zero would drag the mean down invisibly.
        with pytest.raises(ValueError):
            eval_metrics.recall_any_at_k(["a"], set(), k=5)


class TestReciprocalRank:
    def test_first_hit_at_rank_one(self):
        assert eval_metrics.reciprocal_rank(["gold", "b"], {"gold"}) == 1.0

    def test_first_hit_at_rank_three(self):
        assert eval_metrics.reciprocal_rank(["a", "b", "gold"], {"gold"}) == pytest.approx(1 / 3)

    def test_uses_earliest_gold_when_several_present(self):
        assert eval_metrics.reciprocal_rank(["a", "g2", "g1"], {"g1", "g2"}) == pytest.approx(0.5)

    def test_no_hit_scores_zero(self):
        assert eval_metrics.reciprocal_rank(["a", "b"], {"gold"}) == 0.0

    def test_scans_the_whole_ranking_not_just_top_k(self):
        # MRR is conventionally computed over the full ranked list.
        ranking = [f"d{i}" for i in range(50)] + ["gold"]
        assert eval_metrics.reciprocal_rank(ranking, {"gold"}) == pytest.approx(1 / 51)


class TestNdcgAtK:
    def test_single_gold_at_rank_one_is_perfect(self):
        assert eval_metrics.ndcg_at_k(["gold", "b", "c"], {"gold"}, k=5) == pytest.approx(1.0)

    def test_single_gold_at_rank_three(self):
        # DCG = 1/log2(3+1) = 0.5 ; IDCG = 1/log2(1+1) = 1.0
        assert eval_metrics.ndcg_at_k(["a", "b", "gold"], {"gold"}, k=5) == pytest.approx(0.5)

    def test_two_gold_ranked_first_is_perfect(self):
        assert eval_metrics.ndcg_at_k(["g1", "g2", "x"], {"g1", "g2"}, k=5) == pytest.approx(1.0)

    def test_two_gold_split_by_a_miss(self):
        # DCG  = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.63093 = 1.63093
        expected = 1.5 / (1.0 + 1.0 / math.log2(3))
        assert eval_metrics.ndcg_at_k(["g1", "x", "g2"], {"g1", "g2"}, k=5) == pytest.approx(expected)

    def test_ideal_ranking_is_capped_at_k(self):
        # 3 gold but k=1: the best achievable is one hit, so a hit at rank 1
        # must score 1.0 rather than being punished for an unreachable ideal.
        assert eval_metrics.ndcg_at_k(["g1", "g2", "g3"], {"g1", "g2", "g3"}, k=1) == pytest.approx(1.0)

    def test_no_hit_scores_zero(self):
        assert eval_metrics.ndcg_at_k(["a", "b"], {"gold"}, k=5) == 0.0


class TestAggregate:
    def test_averages_each_metric_across_questions(self):
        per_question = [
            {"recall@5": 1.0, "ndcg@10": 1.0, "rr": 1.0},
            {"recall@5": 0.0, "ndcg@10": 0.5, "rr": 0.5},
        ]
        summary = eval_metrics.aggregate(per_question)
        assert summary["recall@5"] == pytest.approx(0.5)
        assert summary["ndcg@10"] == pytest.approx(0.75)
        assert summary["rr"] == pytest.approx(0.75)

    def test_reports_the_question_count(self):
        summary = eval_metrics.aggregate([{"recall@5": 1.0}, {"recall@5": 0.0}])
        assert summary["questions"] == 2

    def test_empty_input_yields_zero_questions_and_no_metrics(self):
        assert eval_metrics.aggregate([]) == {"questions": 0}


class TestScoreQuestion:
    def test_produces_every_configured_cutoff(self):
        scores = eval_metrics.score_question(
            ranked_ids=["a", "gold", "c"],
            gold_ids={"gold"},
            k_values=(1, 5),
        )
        assert scores["recall@1"] == 0.0
        assert scores["recall@5"] == 1.0
        assert scores["rr"] == pytest.approx(0.5)
        assert "ndcg@10" in scores


class TestAggregateRejectsRaggedInput:
    """Skipping absent keys silently shrinks a metric's denominator.

    `[{"recall@5": 1.0}, {}]` used to report questions=2 and recall@5=1.0 —
    the unscored question vanished from the mean while still being counted.
    A partial or resumed run would look complete and better than it was.
    """

    def test_differing_metric_keys_raise(self):
        with pytest.raises(ValueError, match="same metrics"):
            eval_metrics.aggregate([{"recall@5": 1.0}, {"ndcg@10": 0.5}])

    def test_a_question_missing_a_metric_raises(self):
        with pytest.raises(ValueError, match="same metrics"):
            eval_metrics.aggregate([{"recall@5": 1.0}, {}])

    def test_uniform_keys_still_aggregate(self):
        summary = eval_metrics.aggregate([{"recall@5": 1.0}, {"recall@5": 0.0}])
        assert summary["recall@5"] == pytest.approx(0.5)


class TestInvalidCutoffs:
    def test_zero_k_is_rejected(self):
        with pytest.raises(ValueError, match="k must be"):
            eval_metrics.recall_any_at_k(["a"], {"a"}, k=0)

    def test_negative_k_is_rejected(self):
        # ranked_ids[:-3] would silently mean "all but the last three".
        with pytest.raises(ValueError, match="k must be"):
            eval_metrics.recall_any_at_k(["a", "b"], {"a"}, k=-3)

    def test_negative_k_rejected_for_ndcg(self):
        with pytest.raises(ValueError, match="k must be"):
            eval_metrics.ndcg_at_k(["a"], {"a"}, k=-1)


class TestDuplicateRankedIds:
    def test_ndcg_never_exceeds_one_when_a_gold_id_repeats(self):
        # A malformed ranking listing the same gold twice must not be able to
        # accumulate gain twice and score above a perfect ranking.
        score = eval_metrics.ndcg_at_k(["gold", "gold", "gold"], {"gold"}, k=5)
        assert score <= 1.0

    def test_recall_is_unaffected_by_repeats(self):
        assert eval_metrics.recall_any_at_k(["gold", "gold"], {"gold"}, k=5) == 1.0
