"""Tests for query.py's retrieval pipeline.

The LLM synthesis step is not exercised here — that requires a real
Claude subprocess and live cost. Instead, we lock down the two pure
functions that decide which articles get sent to the LLM:

- ``select_relevant_articles`` — should call hybrid_search, dedupe by
  slug (so an article whose two zones both score high doesn't eat two
  budget slots), and respect the top_k cap.
- ``build_retrieved_context`` — should read each selected article's
  full file content and concatenate with the legacy ``## <slug>``
  separator, gracefully skipping articles that vanish between
  selection and read.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def patch_kb(tmp_path, monkeypatch):
    """Point query.py / utils / config at an isolated knowledge tree."""
    import config
    import query
    import utils

    knowledge_dir = tmp_path / "knowledge"
    concepts_dir = knowledge_dir / "concepts"
    concepts_dir.mkdir(parents=True)
    (knowledge_dir / "connections").mkdir()
    (knowledge_dir / "qa").mkdir()

    for mod in (config, utils, query):
        monkeypatch.setattr(mod, "KNOWLEDGE_DIR", knowledge_dir, raising=False)
        monkeypatch.setattr(mod, "CONCEPTS_DIR", concepts_dir, raising=False)
        monkeypatch.setattr(
            mod, "CONNECTIONS_DIR", knowledge_dir / "connections", raising=False
        )
        monkeypatch.setattr(mod, "QA_DIR", knowledge_dir / "qa", raising=False)

    return knowledge_dir


class TestSelectRelevantArticles:
    def test_dedupes_by_slug(self, patch_kb, monkeypatch):
        """Two zones for the same slug must collapse into one selection."""
        import hybrid_search
        import query

        def fake_search(question, limit, **kwargs):
            return [
                {"id": "concepts/a::observed", "slug": "concepts/a", "rrf_score": 0.05},
                {"id": "concepts/a::synthesized", "slug": "concepts/a", "rrf_score": 0.04},
                {"id": "concepts/b::observed", "slug": "concepts/b", "rrf_score": 0.03},
            ]

        monkeypatch.setattr(hybrid_search, "search_articles", fake_search)
        monkeypatch.setattr(query.hybrid_search, "search_articles", fake_search, raising=False) if hasattr(query, "hybrid_search") else None

        # query imports hybrid_search lazily inside the function
        import sys
        sys.modules["hybrid_search"] = hybrid_search

        selected = query.select_relevant_articles("question", top_k=8)
        slugs = [r["slug"] for r in selected]
        assert slugs == ["concepts/a", "concepts/b"]
        assert len(selected) == 2  # not 3 — concepts/a deduped

    def test_respects_top_k_cap(self, patch_kb, monkeypatch):
        import hybrid_search
        import query

        def fake_search(question, limit, **kwargs):
            return [
                {"id": f"concepts/a{i}::observed", "slug": f"concepts/a{i}", "rrf_score": 0.05 - i * 0.001}
                for i in range(20)
            ]

        monkeypatch.setattr(hybrid_search, "search_articles", fake_search)
        import sys
        sys.modules["hybrid_search"] = hybrid_search

        selected = query.select_relevant_articles("question", top_k=5)
        assert len(selected) == 5
        # First five distinct slugs
        assert [r["slug"] for r in selected] == [f"concepts/a{i}" for i in range(5)]


class TestBuildRetrievedContext:
    def test_concatenates_article_bodies_with_separator(self, patch_kb):
        import query

        (patch_kb / "concepts" / "alpha.md").write_text(
            "alpha body content", encoding="utf-8"
        )
        (patch_kb / "concepts" / "beta.md").write_text(
            "beta body content", encoding="utf-8"
        )

        selected = [
            {"slug": "concepts/alpha"},
            {"slug": "concepts/beta"},
        ]
        out = query.build_retrieved_context(selected)

        assert "## concepts/alpha" in out
        assert "alpha body content" in out
        assert "## concepts/beta" in out
        assert "beta body content" in out
        # alpha must come before beta
        assert out.index("alpha body content") < out.index("beta body content")

    def test_silently_skips_missing_articles(self, patch_kb):
        import query

        (patch_kb / "concepts" / "real.md").write_text("real content", encoding="utf-8")

        selected = [
            {"slug": "concepts/real"},
            {"slug": "concepts/vanished"},  # never written
        ]
        out = query.build_retrieved_context(selected)

        assert "real content" in out
        assert "concepts/vanished" not in out

    def test_empty_selection_returns_empty_string(self, patch_kb):
        import query
        assert query.build_retrieved_context([]) == ""
