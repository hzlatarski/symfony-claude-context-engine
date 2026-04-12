"""Tests for utils.embed_article_file — covers the legacy fallback path.

The helper is pure glue over already-tested vector_store + compile_truth
functions, so the smoke test in reindex.py exercises the happy path. The
interesting edge case worth locking down is the fallback branch: legacy
articles without a ``## Truth`` header must still reach the vector store
via ``extract_fallback_truth``.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Give each test a fresh ChromaDB backend rooted at tmp_path."""
    import config
    import vector_store

    monkeypatch.setattr(config, "CHROMA_DB_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", tmp_path / "knowledge")
    (tmp_path / "knowledge" / "concepts").mkdir(parents=True)
    (tmp_path / "knowledge" / "connections").mkdir()
    (tmp_path / "knowledge" / "qa").mkdir()

    # Force a fresh Chroma backend — see test_vector_store for rationale.
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient._identifier_to_system = {}
    except (ImportError, AttributeError):
        pass
    vector_store._client = None
    return vector_store


class TestEmbedArticleFallback:
    def test_legacy_article_without_truth_header_uses_fallback(self, tmp_path, isolated_store, monkeypatch):
        """A legacy article without ## Truth must still embed via fallback.

        Regression guard: the naive path calls extract_zones, gets empty
        observed+synthesized, then silently skips upsert_article. That
        would drop ~60% of legacy articles from the vector store.
        """
        import utils
        # utils.KNOWLEDGE_DIR is captured at import — patch the live attribute.
        monkeypatch.setattr(utils, "KNOWLEDGE_DIR", tmp_path / "knowledge")
        monkeypatch.setattr(utils, "CONTRADICTIONS_FILE", tmp_path / "contradictions.json")

        article = tmp_path / "knowledge" / "concepts" / "legacy-example.md"
        article.write_text(
            "---\n"
            "title: Legacy Example\n"
            "confidence: 0.7\n"
            "---\n"
            "\n"
            "This is a legacy article with no ## Truth header.\n"
            "\n"
            "## Key Points\n"
            "\n"
            "- Important fact one about stimulus controllers\n"
            "- Important fact two about database safety\n",
            encoding="utf-8",
        )

        embedded = utils.embed_article_file(article)
        assert embedded is True

        results = isolated_store.search_articles("legacy article stimulus", limit=5)
        assert any(r["slug"] == "concepts/legacy-example" for r in results)

    def test_zone_article_does_not_use_fallback(self, tmp_path, isolated_store, monkeypatch):
        """New-format article with ## Truth should use extract_zones directly."""
        import utils
        monkeypatch.setattr(utils, "KNOWLEDGE_DIR", tmp_path / "knowledge")
        monkeypatch.setattr(utils, "CONTRADICTIONS_FILE", tmp_path / "contradictions.json")

        article = tmp_path / "knowledge" / "concepts" / "new-example.md"
        article.write_text(
            "---\n"
            "title: New Example\n"
            "confidence: 0.9\n"
            "---\n"
            "\n"
            "## Truth\n"
            "\n"
            "### Observed\n"
            "\n"
            "- Stimulus controllers use kebab-case [src:daily/2026-04-12.md]\n"
            "\n"
            "### Synthesized\n"
            "\n"
            "- The kebab-case convention stems from HTML data-attribute norms\n",
            encoding="utf-8",
        )

        embedded = utils.embed_article_file(article)
        assert embedded is True

        obs_results = isolated_store.search_articles(
            "stimulus controller naming", limit=5, zone_filter="observed"
        )
        assert any(r["slug"] == "concepts/new-example" for r in obs_results)

        syn_results = isolated_store.search_articles(
            "HTML data attribute norms", limit=5, zone_filter="synthesized"
        )
        assert any(r["slug"] == "concepts/new-example" for r in syn_results)

    def test_missing_file_returns_false_without_crashing(self, tmp_path, isolated_store, monkeypatch):
        import utils
        monkeypatch.setattr(utils, "KNOWLEDGE_DIR", tmp_path / "knowledge")
        monkeypatch.setattr(utils, "CONTRADICTIONS_FILE", tmp_path / "contradictions.json")

        ghost = tmp_path / "knowledge" / "concepts" / "never-existed.md"
        assert utils.embed_article_file(ghost) is False
