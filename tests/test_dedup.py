"""Tests for near-duplicate detection and stale-vector hygiene (dedup.py)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts import dedup


class TestNormalize:
    def test_rows_become_unit_length(self):
        m = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
        out = dedup._normalize(m)
        assert np.allclose(np.linalg.norm(out, axis=1), [1.0, 1.0])

    def test_zero_vector_does_not_divide_by_zero(self):
        m = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        out = dedup._normalize(m)
        assert np.isfinite(out).all(), "a zero vector must not produce inf/nan"
        # And it must not come out looking similar to everything else.
        assert float(out[0] @ out[1]) == pytest.approx(0.0, abs=1e-6)


class TestFindNearDuplicates:
    """The sweep itself, with the vector store and the filesystem stubbed."""

    def _stub(self, monkeypatch, vectors: dict[str, list[float]], on_disk: set[str]):
        ids = [f"{slug}::observed" for slug in vectors]
        embeddings = [vectors[s] for s in vectors]

        class FakeCollection:
            def get(self, include=None, where=None):
                return {"ids": list(ids), "embeddings": list(embeddings)}

        monkeypatch.setattr(dedup.vs, "_articles_collection", lambda: FakeCollection())
        monkeypatch.setattr(dedup, "article_exists", lambda slug: slug in on_disk)

    def test_identical_vectors_are_reported_as_identical(self, monkeypatch):
        self._stub(
            monkeypatch,
            {"concepts/a": [1.0, 0.0], "concepts/b": [1.0, 0.0]},
            on_disk={"concepts/a", "concepts/b"},
        )
        pairs = dedup.find_near_duplicates(threshold=0.88)
        assert len(pairs) == 1
        assert pairs[0].similarity == pytest.approx(1.0)
        assert pairs[0].identical is True

    def test_orthogonal_vectors_are_not_duplicates(self, monkeypatch):
        self._stub(
            monkeypatch,
            {"concepts/a": [1.0, 0.0], "concepts/b": [0.0, 1.0]},
            on_disk={"concepts/a", "concepts/b"},
        )
        assert dedup.find_near_duplicates(threshold=0.88) == []

    def test_magnitude_does_not_affect_similarity(self, monkeypatch):
        # Same direction, very different magnitude: still a duplicate.
        self._stub(
            monkeypatch,
            {"concepts/a": [1.0, 0.0], "concepts/b": [50.0, 0.0]},
            on_disk={"concepts/a", "concepts/b"},
        )
        pairs = dedup.find_near_duplicates(threshold=0.88)
        assert len(pairs) == 1

    def test_each_unordered_pair_reported_once_and_never_self_paired(self, monkeypatch):
        self._stub(
            monkeypatch,
            {
                "concepts/a": [1.0, 0.0],
                "concepts/b": [1.0, 0.0],
                "concepts/c": [1.0, 0.0],
            },
            on_disk={"concepts/a", "concepts/b", "concepts/c"},
        )
        pairs = dedup.find_near_duplicates(threshold=0.88)
        # 3 articles, all mutually identical -> exactly 3 pairs (3 choose 2)
        assert len(pairs) == 3
        keys = {frozenset((p.slug_a, p.slug_b)) for p in pairs}
        assert len(keys) == 3
        assert all(p.slug_a != p.slug_b for p in pairs)

    def test_ghost_vectors_cannot_manufacture_pairs(self, monkeypatch):
        """A slug with no file must not be reported as a duplicate.

        This is a real regression: two of the top 'duplicate' pairs in the live
        KB were against CSRF articles that had already been deleted from disk.
        """
        self._stub(
            monkeypatch,
            {"concepts/real": [1.0, 0.0], "concepts/ghost": [1.0, 0.0]},
            on_disk={"concepts/real"},  # ghost has no file
        )
        assert dedup.find_near_duplicates(threshold=0.88) == []

    def test_results_sorted_by_similarity_descending(self, monkeypatch):
        self._stub(
            monkeypatch,
            {
                "concepts/a": [1.0, 0.0],
                "concepts/b": [1.0, 0.0],          # 1.00 vs a
                "concepts/c": [0.94, 0.34],        # ~0.94 vs a
            },
            on_disk={"concepts/a", "concepts/b", "concepts/c"},
        )
        pairs = dedup.find_near_duplicates(threshold=0.88)
        sims = [p.similarity for p in pairs]
        assert sims == sorted(sims, reverse=True)

    def test_empty_index_returns_no_pairs(self, monkeypatch):
        class EmptyCollection:
            def get(self, include=None, where=None):
                return {"ids": [], "embeddings": []}

        monkeypatch.setattr(dedup.vs, "_articles_collection", lambda: EmptyCollection())
        assert dedup.find_near_duplicates() == []

    def test_similarity_is_a_plain_float(self, monkeypatch):
        """np.float32 is not JSON-serializable; --json must not blow up."""
        import json
        self._stub(
            monkeypatch,
            {"concepts/a": [1.0, 0.0], "concepts/b": [1.0, 0.0]},
            on_disk={"concepts/a", "concepts/b"},
        )
        pair = dedup.find_near_duplicates(threshold=0.88)[0]
        assert type(pair.similarity) is float
        json.dumps({"s": pair.similarity})  # must not raise


class TestStaleVectors:
    def test_finds_indexed_slugs_with_no_file(self, monkeypatch):
        class FakeCollection:
            def get(self, include=None, where=None):
                return {"ids": [
                    "concepts/real::observed",
                    "concepts/real::synthesized",
                    "concepts/ghost::observed",
                ]}

        monkeypatch.setattr(dedup.vs, "_articles_collection", lambda: FakeCollection())
        monkeypatch.setattr(dedup, "article_exists", lambda slug: slug == "concepts/real")

        assert dedup.find_stale_vectors() == ["concepts/ghost"]

    def test_prune_deletes_each_stale_slug_once(self, monkeypatch):
        deleted: list[str] = []
        monkeypatch.setattr(dedup, "find_stale_vectors", lambda: ["concepts/x", "concepts/y"])
        monkeypatch.setattr(dedup.vs, "delete_article", lambda slug: deleted.append(slug))

        removed = dedup.prune_stale_vectors()

        assert removed == ["concepts/x", "concepts/y"]
        assert deleted == ["concepts/x", "concepts/y"]


class TestPreflightBlock:
    def test_empty_candidates_render_nothing(self):
        assert dedup.format_preflight_block([]) == ""

    def test_block_names_the_candidate_slugs(self):
        block = dedup.format_preflight_block([
            {"slug": "concepts/foo", "title": "Foo", "similarity": 0.91},
            {"slug": "concepts/bar", "title": "", "similarity": 0.77},
        ])
        assert "concepts/foo" in block
        assert "concepts/bar" in block
        assert "0.91" in block
        # It must steer toward UPDATE, which is the whole point of the block.
        assert "UPDATING" in block

    def test_missing_similarity_does_not_crash_the_renderer(self):
        block = dedup.format_preflight_block([
            {"slug": "concepts/foo", "title": "Foo", "similarity": None},
        ])
        assert "concepts/foo" in block


class TestSimilarToText:
    def test_blank_text_short_circuits_without_querying(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("must not query the vector store for empty text")

        monkeypatch.setattr(dedup.vs, "search_articles", boom)
        assert dedup.similar_to_text("   ") == []

    def test_converts_distance_to_similarity_and_applies_floor(self, monkeypatch):
        monkeypatch.setattr(dedup.vs, "search_articles", lambda **k: [
            {"slug": "concepts/close", "metadata": {"title": "Close"}, "distance": 0.10},
            {"slug": "concepts/far", "metadata": {"title": "Far"}, "distance": 0.80},
        ])
        out = dedup.similar_to_text("anything", limit=5, threshold=0.55)

        assert [c["slug"] for c in out] == ["concepts/close"]
        assert out[0]["similarity"] == pytest.approx(0.90)


class TestChunkedQuery:
    """The embedder truncates at ~256 tokens; a whole daily log must be chunked."""

    def test_short_text_is_one_chunk(self):
        assert dedup._chunk_for_query("hello world") == ["hello world"]

    def test_blank_text_yields_no_chunks(self):
        assert dedup._chunk_for_query("   \n  ") == []

    def test_long_text_is_split(self):
        text = "\n\n".join(f"Paragraph {i}. " * 20 for i in range(20))
        chunks = dedup._chunk_for_query(text, chunk_chars=900)
        assert len(chunks) > 1
        assert all(len(c) <= 1200 for c in chunks)

    def test_chunk_count_is_bounded(self):
        text = "x " * 100_000
        assert len(dedup._chunk_for_query(text)) <= dedup.MAX_QUERY_CHUNKS

    def test_late_content_still_reaches_the_vector_store(self, monkeypatch):
        """Regression: passing the whole log as one query embedded only its head.

        A concept discussed only at the END of a long daily log must still
        produce a preflight candidate.
        """
        seen: list[str] = []

        def fake_search(query, **kwargs):
            seen.append(query)
            if "LATE_TOPIC" in query:
                return [{"slug": "concepts/late", "metadata": {"title": "Late"}, "distance": 0.1}]
            return [{"slug": "concepts/early", "metadata": {"title": "Early"}, "distance": 0.2}]

        monkeypatch.setattr(dedup.vs, "search_articles", fake_search)

        log = ("Early filler content. " * 200) + "\n\nLATE_TOPIC appears only here."
        out = dedup.similar_to_text(log, limit=5)

        assert len(seen) > 1, "long source must be chunked into multiple queries"
        assert any("LATE_TOPIC" in q for q in seen), "tail of the source was never embedded"
        assert "concepts/late" in [c["slug"] for c in out]

    def test_duplicate_hits_across_chunks_keep_the_best_similarity(self, monkeypatch):
        calls = {"n": 0}

        def fake_search(query, **kwargs):
            calls["n"] += 1
            # same article, weak hit first then a strong one
            distance = 0.40 if calls["n"] == 1 else 0.05
            return [{"slug": "concepts/x", "metadata": {"title": "X"}, "distance": distance}]

        monkeypatch.setattr(dedup.vs, "search_articles", fake_search)

        out = dedup.similar_to_text("y" * 3000, limit=5)

        assert len(out) == 1, "the same article must not be listed twice"
        assert out[0]["similarity"] == pytest.approx(0.95)
