"""Tests for knowledge_mcp_server — targets the pure _impl functions.

The FastMCP transport layer is covered by FastMCP's own test suite and
by test_mcp_server.py's launch regression test. Here we only verify
our tool *implementations* behave correctly against a seeded ChromaDB
store, and that the MCP server imports cleanly (a regression guard
against the sys.path bootstrap bug that bit mcp_server.py).
"""
from __future__ import annotations

import pytest


@pytest.fixture
def seeded_store(tmp_path, monkeypatch):
    import config
    import vector_store

    monkeypatch.setattr(config, "CHROMA_DB_DIR", tmp_path / "chroma")
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient._identifier_to_system = {}
    except (ImportError, AttributeError):
        pass
    vector_store._client = None

    vector_store.upsert_article(
        slug="concepts/stimulus-naming",
        title="Stimulus Naming",
        zone="observed",
        text="Stimulus controller identifiers use kebab-case; filenames use underscores.",
        metadata={
            "type": "fact",
            "confidence": 0.9,
            "quarantined": False,
            "updated": "2026-04-12",
        },
    )
    vector_store.upsert_article(
        slug="concepts/no-destructive-migrations",
        title="No Destructive Migrations",
        zone="observed",
        text="Never drop tables or columns in database migrations — data loss risk.",
        metadata={
            "type": "preference",
            "confidence": 0.95,
            "quarantined": False,
            "updated": "2026-04-12",
        },
    )
    vector_store.upsert_article(
        slug="concepts/low-confidence-plan",
        title="Low Confidence Plan",
        zone="observed",
        text="Tentative plan to rework notification pipeline, may change.",
        metadata={
            "type": "decision",
            "confidence": 0.3,
            "quarantined": False,
            "updated": "2026-04-12",
        },
    )
    vector_store.upsert_article(
        slug="concepts/stimulus-naming",
        title="Stimulus Naming",
        zone="synthesized",
        text="The kebab-case convention mirrors HTML data-attribute naming norms.",
        metadata={
            "type": "fact",
            "confidence": 0.9,
            "quarantined": False,
            "updated": "2026-04-12",
        },
    )
    return vector_store


class TestSearchKnowledgeImpl:
    def test_returns_matching_articles(self, seeded_store):
        from knowledge_mcp_server import _search_knowledge_impl
        results = _search_knowledge_impl("stimulus naming convention", limit=3)
        assert len(results) >= 1
        assert results[0]["slug"] == "concepts/stimulus-naming"

    def test_type_filter(self, seeded_store):
        from knowledge_mcp_server import _search_knowledge_impl
        results = _search_knowledge_impl("migration safety rule", limit=5, type_filter="preference")
        slugs = {r["slug"] for r in results}
        assert "concepts/no-destructive-migrations" in slugs
        assert "concepts/stimulus-naming" not in slugs

    def test_min_confidence_filter(self, seeded_store):
        from knowledge_mcp_server import _search_knowledge_impl
        results = _search_knowledge_impl("plan", limit=5, min_confidence=0.5)
        slugs = {r["slug"] for r in results}
        assert "concepts/low-confidence-plan" not in slugs

    def test_zone_filter_observed(self, seeded_store):
        from knowledge_mcp_server import _search_knowledge_impl
        results = _search_knowledge_impl(
            "stimulus naming", limit=5, zone_filter="observed"
        )
        assert all(r["metadata"]["zone"] == "observed" for r in results)

    def test_zone_filter_synthesized(self, seeded_store):
        from knowledge_mcp_server import _search_knowledge_impl
        results = _search_knowledge_impl(
            "kebab-case HTML data attribute", limit=5, zone_filter="synthesized"
        )
        assert any(r["metadata"]["zone"] == "synthesized" for r in results)

    def test_invalid_type_filter_raises(self, seeded_store):
        from knowledge_mcp_server import _search_knowledge_impl
        with pytest.raises(ValueError, match="type_filter"):
            _search_knowledge_impl("anything", type_filter="banana")

    def test_invalid_zone_filter_raises(self, seeded_store):
        from knowledge_mcp_server import _search_knowledge_impl
        with pytest.raises(ValueError, match="zone_filter"):
            _search_knowledge_impl("anything", zone_filter="unknown")

    def test_invalid_mode_raises(self, seeded_store):
        from knowledge_mcp_server import _search_knowledge_impl
        with pytest.raises(ValueError, match="mode"):
            _search_knowledge_impl("anything", mode="weird")

    def test_mode_dispatch_routes_to_backends(self, seeded_store, monkeypatch):
        """Each mode value must dispatch to its corresponding backend."""
        import bm25_store
        import hybrid_search
        import knowledge_mcp_server
        import vector_store

        calls = []

        def make_spy(name):
            def spy(**kwargs):
                calls.append(name)
                return []
            return spy

        monkeypatch.setattr(vector_store, "search_articles", make_spy("vector"))
        monkeypatch.setattr(bm25_store, "search_articles", make_spy("bm25"))
        monkeypatch.setattr(hybrid_search, "search_articles", make_spy("hybrid"))
        monkeypatch.setattr(
            knowledge_mcp_server, "vector_store", vector_store, raising=False
        )
        monkeypatch.setattr(
            knowledge_mcp_server, "bm25_store", bm25_store, raising=False
        )
        monkeypatch.setattr(
            knowledge_mcp_server, "hybrid_search", hybrid_search, raising=False
        )

        knowledge_mcp_server._search_knowledge_impl("q", mode="vector")
        knowledge_mcp_server._search_knowledge_impl("q", mode="bm25")
        knowledge_mcp_server._search_knowledge_impl("q", mode="hybrid")
        assert calls == ["vector", "bm25", "hybrid"]

    def test_quarantined_excluded_by_default(self, seeded_store):
        seeded_store.upsert_article(
            slug="concepts/quarantined-one",
            title="Quarantined",
            zone="observed",
            text="this article is under contradiction quarantine",
            metadata={
                "type": "fact",
                "confidence": 0.9,
                "quarantined": True,
                "updated": "2026-04-12",
            },
        )
        from knowledge_mcp_server import _search_knowledge_impl
        results = _search_knowledge_impl("quarantine contradiction", limit=10)
        slugs = {r["slug"] for r in results}
        assert "concepts/quarantined-one" not in slugs

    def test_quarantined_included_when_opted_in(self, seeded_store):
        seeded_store.upsert_article(
            slug="concepts/quarantined-two",
            title="Quarantined",
            zone="observed",
            text="this article is under contradiction quarantine",
            metadata={
                "type": "fact",
                "confidence": 0.9,
                "quarantined": True,
                "updated": "2026-04-12",
            },
        )
        from knowledge_mcp_server import _search_knowledge_impl
        results = _search_knowledge_impl(
            "quarantine contradiction", limit=10, include_quarantined=True
        )
        slugs = {r["slug"] for r in results}
        assert "concepts/quarantined-two" in slugs


class TestSlimSearchShape:
    """Search tools return token-efficient snippets, not full bodies.

    Borrowed from claude-mem's search → get_observations split: the search
    response carries just enough to pick interesting slugs; full bodies come
    from a follow-up ``get_article`` / ``get_articles`` call.
    """

    def test_search_returns_snippet_not_full_text(self, seeded_store):
        long_body = (
            "Stimulus controller identifiers use kebab-case; filenames use underscores. "
            * 30
        )
        seeded_store.upsert_article(
            slug="concepts/long-stimulus-article",
            title="Long Stimulus Article",
            zone="observed",
            text=long_body,
            metadata={
                "type": "fact",
                "confidence": 0.9,
                "quarantined": False,
                "updated": "2026-04-12",
            },
        )

        from knowledge_mcp_server import SNIPPET_CHARS, _search_knowledge_impl
        results = _search_knowledge_impl("long stimulus article kebab underscore", limit=5)
        hit = next(r for r in results if r["slug"] == "concepts/long-stimulus-article")

        # No full-text field on the slim shape — agents must call get_article
        assert "text" not in hit
        # Snippet is truncated; the ellipsis marker means "body was cut"
        assert "snippet" in hit
        assert len(hit["snippet"]) <= SNIPPET_CHARS + 1  # +1 for the "…" glyph
        assert hit["snippet"].endswith("…")

    def test_search_includes_title_from_metadata(self, seeded_store):
        from knowledge_mcp_server import _search_knowledge_impl
        results = _search_knowledge_impl("stimulus naming convention", limit=3)
        hit = next(r for r in results if r["slug"] == "concepts/stimulus-naming")
        # upsert_article stores the title kwarg in metadata; slim shape surfaces it
        assert hit["title"] == "Stimulus Naming"

    def test_search_title_falls_back_to_slug(self, seeded_store, monkeypatch):
        """If no title in metadata, the slim shape falls back to the slug."""
        import knowledge_mcp_server

        def fake_backend(**kwargs):
            return [{
                "id": "concepts/no-title",
                "slug": "concepts/no-title",
                "text": "body",
                "metadata": {"type": "fact"},  # no title key
                "distance": 0.1,
            }]

        monkeypatch.setattr(
            knowledge_mcp_server.hybrid_search, "search_articles", fake_backend
        )
        results = knowledge_mcp_server._search_knowledge_impl("anything")
        assert results[0]["title"] == "concepts/no-title"

    def test_short_body_snippet_has_no_ellipsis(self, seeded_store, monkeypatch):
        import knowledge_mcp_server

        def fake_backend(**kwargs):
            return [{
                "id": "concepts/short",
                "slug": "concepts/short",
                "text": "short body",
                "metadata": {"type": "fact", "title": "Short"},
                "distance": 0.1,
            }]

        monkeypatch.setattr(
            knowledge_mcp_server.hybrid_search, "search_articles", fake_backend
        )
        results = knowledge_mcp_server._search_knowledge_impl("anything")
        assert results[0]["snippet"] == "short body"
        assert not results[0]["snippet"].endswith("…")

    def test_slim_metadata_drops_large_keys(self, seeded_store, monkeypatch):
        """Only the load-bearing metadata flags survive the slim projection."""
        import knowledge_mcp_server

        def fake_backend(**kwargs):
            return [{
                "id": "concepts/fat",
                "slug": "concepts/fat",
                "text": "some content",
                "metadata": {
                    "type": "fact",
                    "confidence": 0.9,
                    "zone": "observed",
                    "quarantined": False,
                    "updated": "2026-04-12",
                    "title": "Fat",
                    "huge_blob": "x" * 5000,  # must NOT leak into the response
                    "internal_debug_info": {"chroma": "stuff"},
                },
                "distance": 0.1,
            }]

        monkeypatch.setattr(
            knowledge_mcp_server.hybrid_search, "search_articles", fake_backend
        )
        results = knowledge_mcp_server._search_knowledge_impl("anything")
        md = results[0]["metadata"]
        assert set(md.keys()) <= {"type", "confidence", "zone", "quarantined", "updated"}
        assert "huge_blob" not in md
        assert "internal_debug_info" not in md


class TestGetArticlesImpl:
    """Batch fetch for full article bodies — companion to the slim search."""

    def test_batch_fetches_multiple_articles(self, tmp_path, monkeypatch):
        import config

        monkeypatch.setattr(config, "KNOWLEDGE_DIR", tmp_path)
        (tmp_path / "concepts").mkdir()
        (tmp_path / "concepts" / "foo.md").write_text(
            "---\ntitle: Foo\ntype: fact\nconfidence: 0.8\n---\n\nfoo body\n",
            encoding="utf-8",
        )
        (tmp_path / "concepts" / "bar.md").write_text(
            "---\ntitle: Bar\ntype: decision\nconfidence: 0.6\n---\n\nbar body\n",
            encoding="utf-8",
        )

        from knowledge_mcp_server import _get_articles_impl
        results = _get_articles_impl(["concepts/foo", "concepts/bar"])

        assert len(results) == 2
        # Order preserved from input list
        assert [r["slug"] for r in results] == ["concepts/foo", "concepts/bar"]
        assert "foo body" in results[0]["content"]
        assert "bar body" in results[1]["content"]
        assert results[0]["frontmatter"]["title"] == "Foo"
        assert results[1]["frontmatter"]["type"] == "decision"

    def test_missing_slug_returns_error_marker_not_exception(self, tmp_path, monkeypatch):
        """A single bad slug must not abort the whole batch."""
        import config

        monkeypatch.setattr(config, "KNOWLEDGE_DIR", tmp_path)
        (tmp_path / "concepts").mkdir()
        (tmp_path / "concepts" / "real.md").write_text(
            "---\ntitle: Real\ntype: fact\n---\n\nreal body\n",
            encoding="utf-8",
        )

        from knowledge_mcp_server import _get_articles_impl
        results = _get_articles_impl([
            "concepts/real",
            "concepts/ghost",
            "concepts/also-ghost",
        ])

        assert len(results) == 3
        assert results[0]["slug"] == "concepts/real"
        assert "real body" in results[0]["content"]
        assert results[1] == {"slug": "concepts/ghost", "error": "not_found"}
        assert results[2] == {"slug": "concepts/also-ghost", "error": "not_found"}

    def test_empty_slug_list_returns_empty_list(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, "KNOWLEDGE_DIR", tmp_path)

        from knowledge_mcp_server import _get_articles_impl
        assert _get_articles_impl([]) == []


class TestSearchRawDailyImpl:
    def test_returns_chunks(self, seeded_store):
        seeded_store.upsert_chunk(
            chunk_id="daily/2026-04-10.md#session-14-08",
            source_file="daily/2026-04-10.md",
            text="## Session (14:08)\n\nDiscussed framework A vs framework B tradeoffs.",
            metadata={"section": "Session (14:08)", "date": "2026-04-10"},
        )
        from knowledge_mcp_server import _search_raw_daily_impl
        results = _search_raw_daily_impl("framework A vs framework B", limit=3)
        assert len(results) >= 1
        assert results[0]["id"] == "daily/2026-04-10.md#session-14-08"

    def test_date_range_filter(self, seeded_store):
        seeded_store.upsert_chunk(
            chunk_id="daily/2026-04-01.md#old",
            source_file="daily/2026-04-01.md",
            text="## Old\n\nstale content from the beginning of the month",
            metadata={"section": "Old", "date": "2026-04-01"},
        )
        seeded_store.upsert_chunk(
            chunk_id="daily/2026-04-11.md#new",
            source_file="daily/2026-04-11.md",
            text="## New\n\nrecent content from last week",
            metadata={"section": "New", "date": "2026-04-11"},
        )
        from knowledge_mcp_server import _search_raw_daily_impl
        results = _search_raw_daily_impl("content", limit=5, date_from="2026-04-10")
        ids = {r["id"] for r in results}
        assert "daily/2026-04-11.md#new" in ids
        assert "daily/2026-04-01.md#old" not in ids


class TestGetArticleImpl:
    def test_reads_article_by_slug(self, tmp_path, monkeypatch):
        import config

        monkeypatch.setattr(config, "KNOWLEDGE_DIR", tmp_path)
        (tmp_path / "concepts").mkdir()
        (tmp_path / "concepts" / "foo.md").write_text(
            "---\ntitle: Foo\ntype: fact\nconfidence: 0.8\n---\n\n## Truth\n\nsome content\n",
            encoding="utf-8",
        )

        from knowledge_mcp_server import _get_article_impl
        result = _get_article_impl("concepts/foo")
        assert result["slug"] == "concepts/foo"
        assert "some content" in result["content"]
        assert result["frontmatter"]["type"] == "fact"
        assert result["frontmatter"]["title"] == "Foo"
        assert result["frontmatter"]["confidence"] == 0.8

    def test_missing_article_raises(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, "KNOWLEDGE_DIR", tmp_path)

        from knowledge_mcp_server import _get_article_impl
        with pytest.raises(FileNotFoundError):
            _get_article_impl("concepts/does-not-exist")


class TestListContradictionsImpl:
    def test_reads_quarantine_file(self, tmp_path, monkeypatch):
        import json
        import utils

        qfile = tmp_path / "contradictions.json"
        qfile.write_text(
            json.dumps({
                "quarantined": ["concepts/bad-a", "concepts/bad-b"],
                "updated": "2026-04-12T00:00:00+00:00",
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr(utils, "CONTRADICTIONS_FILE", qfile)

        from knowledge_mcp_server import _list_contradictions_impl
        result = _list_contradictions_impl()
        assert result["count"] == 2
        assert sorted(result["quarantined"]) == ["concepts/bad-a", "concepts/bad-b"]

    def test_empty_quarantine_returns_empty_list(self, tmp_path, monkeypatch):
        import utils
        monkeypatch.setattr(utils, "CONTRADICTIONS_FILE", tmp_path / "contradictions.json")

        from knowledge_mcp_server import _list_contradictions_impl
        result = _list_contradictions_impl()
        assert result == {"quarantined": [], "count": 0}


class TestKnowledgeMcpServerBoot:
    def test_make_server_returns_fastmcp_instance(self):
        from knowledge_mcp_server import _make_server
        server = _make_server()
        assert server is not None
        assert hasattr(server, "run")
