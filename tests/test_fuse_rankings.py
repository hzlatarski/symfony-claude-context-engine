"""Tests for the extracted RRF fusion primitive.

``hybrid_search.search_articles`` had the fusion math inlined, which meant
the benchmark could only measure it by copying it — and a copy drifts.
These tests pin the pure function that both now share.
"""
from __future__ import annotations

import pytest

from hybrid_search import _RRF_K, fuse_rankings


def _doc(doc_id: str, **extra) -> dict:
    return {"id": doc_id, **extra}


class TestFusionMath:
    def test_document_in_both_lists_accumulates_both_contributions(self):
        fused = fuse_rankings([[_doc("a")], [_doc("a")]], limit=5)
        assert fused[0]["rrf_score"] == pytest.approx(2 * (1 / (_RRF_K + 1)))

    def test_document_in_one_list_gets_a_single_contribution(self):
        fused = fuse_rankings([[_doc("a")], [_doc("b")]], limit=5)
        by_id = {d["id"]: d["rrf_score"] for d in fused}
        assert by_id["a"] == pytest.approx(1 / (_RRF_K + 1))
        assert by_id["b"] == pytest.approx(1 / (_RRF_K + 1))

    def test_rank_position_discounts_the_contribution(self):
        fused = fuse_rankings([[_doc("first"), _doc("second")]], limit=5)
        by_id = {d["id"]: d["rrf_score"] for d in fused}
        assert by_id["first"] == pytest.approx(1 / (_RRF_K + 1))
        assert by_id["second"] == pytest.approx(1 / (_RRF_K + 2))

    def test_agreement_across_lists_beats_a_single_top_rank(self):
        # "b" is 2nd in both lists, "a" is 1st in one and absent from the other.
        # RRF's whole point: corroboration outranks a lone strong opinion.
        fused = fuse_rankings(
            [[_doc("a"), _doc("b")], [_doc("c"), _doc("b")]],
            limit=5,
        )
        assert fused[0]["id"] == "b"


class TestOrderingAndShape:
    def test_sorts_by_descending_fused_score(self):
        fused = fuse_rankings(
            [[_doc("x"), _doc("y"), _doc("z")], [_doc("z")]],
            limit=5,
        )
        scores = [d["rrf_score"] for d in fused]
        assert scores == sorted(scores, reverse=True)

    def test_truncates_to_the_limit(self):
        fused = fuse_rankings([[_doc(f"d{i}") for i in range(10)]], limit=3)
        assert len(fused) == 3

    def test_deduplicates_by_id(self):
        fused = fuse_rankings([[_doc("a")], [_doc("a")], [_doc("a")]], limit=5)
        assert len(fused) == 1

    def test_keeps_the_first_seen_payload_for_a_duplicate_id(self):
        fused = fuse_rankings(
            [[_doc("a", source="vector")], [_doc("a", source="bm25")]],
            limit=5,
        )
        assert fused[0]["source"] == "vector"


class TestDegenerateInputs:
    def test_no_lists_fuses_to_nothing(self):
        assert fuse_rankings([], limit=5) == []

    def test_all_empty_lists_fuse_to_nothing(self):
        assert fuse_rankings([[], []], limit=5) == []

    def test_tolerates_more_than_two_streams(self):
        fused = fuse_rankings([[_doc("a")], [_doc("a")], [_doc("a")]], limit=5)
        assert fused[0]["rrf_score"] == pytest.approx(3 * (1 / (_RRF_K + 1)))


class TestDuplicateIdsWithinOneStream:
    """RRF assumes one rank per document per ranking.

    A stream that lists the same id twice would otherwise cast two votes,
    outranking documents that genuinely appeared in both streams.
    """

    def test_repeat_within_a_stream_counts_once(self):
        fused = fuse_rankings([[_doc("a"), _doc("a")]], limit=5)
        assert fused[0]["rrf_score"] == pytest.approx(1 / (_RRF_K + 1))

    def test_repeat_uses_the_best_rank(self):
        fused = fuse_rankings([[_doc("x"), _doc("a"), _doc("a")]], limit=5)
        by_id = {d["id"]: d["rrf_score"] for d in fused}
        assert by_id["a"] == pytest.approx(1 / (_RRF_K + 2))

    def test_a_duplicated_single_stream_cannot_outrank_genuine_agreement(self):
        # "dup" is listed twice by one stream; "both" appears once in each.
        fused = fuse_rankings(
            [[_doc("dup"), _doc("dup"), _doc("both")], [_doc("both")]],
            limit=5,
        )
        assert fused[0]["id"] == "both"
