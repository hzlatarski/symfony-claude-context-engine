"""Tests for the hybrid RRF fusion layer.

Uses stub backends rather than real Chroma + BM25 so the test suite stays
fast and deterministic. The point of these tests is fusion correctness,
not backend behavior — the backends have their own test files.
"""
from __future__ import annotations

import pytest


def _make_result(slug: str, zone: str = "observed", text: str = "") -> dict:
    return {
        "id": f"{slug}::{zone}",
        "slug": slug,
        "text": text or f"text for {slug}",
        "metadata": {"slug": slug, "zone": zone, "type": "fact", "confidence": 0.9},
        "distance": 0.1,
    }


@pytest.fixture
def stub_backends(monkeypatch):
    """Replace vector_store and bm25_store with in-memory stubs.

    Each stub returns a fixed list per-test so fusion math is checkable
    without a live index. The fixture yields a ``(set_vec, set_bm25)``
    helper pair the test uses to stage results.
    """
    import bm25_store
    import hybrid_search
    import vector_store

    vec_state: list[dict] = []
    bm25_state: list[dict] = []

    def fake_vec(**kwargs):
        return list(vec_state)

    def fake_bm25(**kwargs):
        return list(bm25_state)

    monkeypatch.setattr(vector_store, "search_articles", fake_vec)
    monkeypatch.setattr(bm25_store, "search_articles", fake_bm25)
    monkeypatch.setattr(hybrid_search.vector_store, "search_articles", fake_vec)
    monkeypatch.setattr(hybrid_search.bm25_store, "search_articles", fake_bm25)

    def set_vec(results):
        vec_state.clear()
        vec_state.extend(results)

    def set_bm25(results):
        bm25_state.clear()
        bm25_state.extend(results)

    return set_vec, set_bm25


class TestRRFFusion:
    def test_both_backends_empty(self, stub_backends):
        from hybrid_search import search_articles
        assert search_articles("anything", limit=5) == []

    def test_only_vector_has_results(self, stub_backends):
        set_vec, _ = stub_backends
        set_vec([_make_result("concepts/a"), _make_result("concepts/b")])

        from hybrid_search import search_articles
        results = search_articles("query", limit=5)
        slugs = [r["slug"] for r in results]
        assert slugs == ["concepts/a", "concepts/b"]
        assert results[0]["rrf_score"] > results[1]["rrf_score"]

    def test_only_bm25_has_results(self, stub_backends):
        _, set_bm25 = stub_backends
        set_bm25([_make_result("concepts/x")])

        from hybrid_search import search_articles
        results = search_articles("query", limit=5)
        assert [r["slug"] for r in results] == ["concepts/x"]

    def test_both_agreeing_boosts_shared_result(self, stub_backends):
        """Documents ranked by both paths beat documents ranked by one."""
        set_vec, set_bm25 = stub_backends
        set_vec([
            _make_result("concepts/shared"),   # rank 1 vec
            _make_result("concepts/vec-only"), # rank 2 vec
        ])
        set_bm25([
            _make_result("concepts/bm25-only"),  # rank 1 bm25
            _make_result("concepts/shared"),     # rank 2 bm25
        ])

        from hybrid_search import search_articles
        results = search_articles("query", limit=5)
        # "shared" should win: 1/(60+1) + 1/(60+2) ≈ 0.0326
        # "vec-only":           1/(60+2)            ≈ 0.0161
        # "bm25-only":          1/(60+1)            ≈ 0.0164
        assert results[0]["slug"] == "concepts/shared"
        assert results[0]["rrf_score"] == pytest.approx(
            1.0 / 61 + 1.0 / 62, rel=1e-6
        )

    def test_limit_caps_output(self, stub_backends):
        set_vec, set_bm25 = stub_backends
        set_vec([_make_result(f"concepts/a{i}") for i in range(10)])
        set_bm25([_make_result(f"concepts/b{i}") for i in range(10)])

        from hybrid_search import search_articles
        results = search_articles("query", limit=3)
        assert len(results) == 3

    def test_dedup_by_composite_id(self, stub_backends):
        set_vec, set_bm25 = stub_backends
        r1 = _make_result("concepts/same")
        r2 = _make_result("concepts/same")  # same id
        set_vec([r1])
        set_bm25([r2])

        from hybrid_search import search_articles
        results = search_articles("query", limit=5)
        ids = [r["id"] for r in results]
        assert ids.count("concepts/same::observed") == 1

    def test_filters_passed_through_to_backends(self, stub_backends, monkeypatch):
        """Backends must receive the same filter args the caller passed."""
        import bm25_store
        import hybrid_search
        import vector_store

        calls = {"vec": None, "bm25": None}

        def capture_vec(**kwargs):
            calls["vec"] = kwargs
            return []

        def capture_bm25(**kwargs):
            calls["bm25"] = kwargs
            return []

        monkeypatch.setattr(vector_store, "search_articles", capture_vec)
        monkeypatch.setattr(bm25_store, "search_articles", capture_bm25)
        monkeypatch.setattr(hybrid_search.vector_store, "search_articles", capture_vec)
        monkeypatch.setattr(hybrid_search.bm25_store, "search_articles", capture_bm25)

        hybrid_search.search_articles(
            "x", limit=4, type_filter="fact", min_confidence=0.7,
            zone_filter="observed", include_quarantined=True,
        )
        for side in ("vec", "bm25"):
            assert calls[side]["type_filter"] == "fact"
            assert calls[side]["min_confidence"] == 0.7
            assert calls[side]["zone_filter"] == "observed"
            assert calls[side]["include_quarantined"] is True
            # Pool must be larger than limit so RRF has candidates to fuse.
            assert calls[side]["limit"] >= 4
