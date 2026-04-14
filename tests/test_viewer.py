"""Smoke tests for scripts/viewer.py — the FastAPI read-only dashboard.

Every route gets exercised via FastAPI's TestClient against an isolated
temp knowledge directory seeded with a handful of articles, a daily log,
and a tool drawer. No ChromaDB interaction — ``_chroma_stats`` is
monkeypatched to return zeros so the tests don't need the ONNX model
cached on disk.

Tests here assert the *shape* of rendered HTML (status code, presence of
key strings, filter wiring), not pixel-perfect markup. Template churn
shouldn't break tests unless it removes user-visible concepts.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def seeded_kb(tmp_path, monkeypatch):
    """Build a temp knowledge dir + temp daily dir with realistic fixtures.

    Layout::

        tmp_path/
          knowledge/
            concepts/
              stimulus-naming.md     (fact, conf 0.9)
              migrations-policy.md   (preference, conf 0.95)
              low-conf-plan.md       (decision, conf 0.3)
              quarantined-one.md     (fact, conf 0.8, quarantined)
            contradictions.json      (lists quarantined-one)
            daily/
              2026-04-14.md          (raw daily log)
              2026-04-14.tools.jsonl (tool drawer with a few events)

    The fixture monkeypatches ``config.KNOWLEDGE_DIR``, ``config.DAILY_DIR``,
    and ``config.SCRIPTS_DIR`` so the viewer app reads from this temp tree,
    and patches ``viewer._chroma_stats`` so Chroma isn't touched.
    """
    import config
    import utils

    kb = tmp_path / "knowledge"
    concepts = kb / "concepts"
    concepts.mkdir(parents=True)
    daily = kb / "daily"
    daily.mkdir()

    (concepts / "stimulus-naming.md").write_text(
        "---\n"
        "title: Stimulus Naming\n"
        "type: fact\n"
        "confidence: 0.9\n"
        "updated: 2026-04-12\n"
        "sources:\n"
        "  - daily/2026-04-01.md\n"
        "---\n\n"
        "## Truth\n\n"
        "- Stimulus controllers use kebab-case identifiers.\n"
        "- See [[concepts/migrations-policy]] for related guidance.\n",
        encoding="utf-8",
    )
    (concepts / "migrations-policy.md").write_text(
        "---\n"
        "title: Migrations Policy\n"
        "type: preference\n"
        "confidence: 0.95\n"
        "updated: 2026-04-13\n"
        "---\n\n"
        "## Truth\n\n"
        "- Never drop tables or columns in migrations.\n",
        encoding="utf-8",
    )
    (concepts / "low-conf-plan.md").write_text(
        "---\n"
        "title: Tentative Rewrite Plan\n"
        "type: decision\n"
        "confidence: 0.3\n"
        "updated: 2026-04-10\n"
        "---\n\n"
        "## Truth\n\n- Tentative plan; may change.\n",
        encoding="utf-8",
    )
    (concepts / "quarantined-one.md").write_text(
        "---\n"
        "title: Quarantined Article\n"
        "type: fact\n"
        "confidence: 0.8\n"
        "quarantined: true\n"
        "updated: 2026-04-11\n"
        "---\n\n"
        "## Truth\n\n- this one is flagged.\n",
        encoding="utf-8",
    )

    (kb / "contradictions.json").write_text(
        '{"quarantined": ["concepts/quarantined-one"], "updated": "2026-04-14T00:00:00+00:00"}',
        encoding="utf-8",
    )

    (daily / "2026-04-14.md").write_text(
        "# Daily Log: 2026-04-14\n\n## Sessions\n\n### Session (09:00)\n\nWorked on X.\n",
        encoding="utf-8",
    )
    (daily / "2026-04-14.tools.jsonl").write_text(
        '{"ts":"2026-04-14T09:00:01+02:00","session_id":"sess-1","tool":"Edit","input":{"file_path":"src/Foo.php"},"result_size":123,"ok":true}\n'
        '{"ts":"2026-04-14T09:00:05+02:00","session_id":"sess-1","tool":"Bash","input":{"command":"php bin/phpunit"},"result_size":4000,"ok":true}\n'
        '{"ts":"2026-04-14T09:00:20+02:00","session_id":"sess-1","tool":"Bash","input":{"command":"false"},"result_size":0,"ok":false}\n',
        encoding="utf-8",
    )

    scripts_state = tmp_path / "scripts_state"
    scripts_state.mkdir()
    (scripts_state / "last-flush.json").write_text(
        '{"flush_costs": [{"session_id":"sess-1","timestamp":0,"cost_usd":0.012,"result":"saved"}]}',
        encoding="utf-8",
    )
    (scripts_state / "state.json").write_text(
        '{"ingested": {"README.md": {"hash":"abc"}, "AGENTS.md": {"hash":"def"}}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "KNOWLEDGE_DIR", kb)
    monkeypatch.setattr(config, "DAILY_DIR", daily)
    monkeypatch.setattr(config, "SCRIPTS_DIR", scripts_state)
    monkeypatch.setattr(config, "STATE_FILE", scripts_state / "state.json")
    monkeypatch.setattr(utils, "CONTRADICTIONS_FILE", kb / "contradictions.json")

    import viewer
    monkeypatch.setattr(viewer, "_chroma_stats", lambda: {"articles": 4, "daily_chunks": 1})

    return kb


@pytest.fixture
def client(seeded_kb):
    """Return a TestClient bound to a fresh app reading from seeded_kb."""
    import viewer
    app = viewer.create_app(knowledge_dir=seeded_kb)
    return TestClient(app)


class TestLoaders:
    """The pure data-loading helpers are testable without FastAPI."""

    def test_load_all_articles_summarizes_fixtures(self, seeded_kb):
        from viewer import load_all_articles
        articles = load_all_articles(seeded_kb)
        slugs = {a.slug for a in articles}
        assert "concepts/stimulus-naming" in slugs
        assert "concepts/migrations-policy" in slugs
        assert "concepts/low-conf-plan" in slugs
        assert "concepts/quarantined-one" in slugs

        by_slug = {a.slug: a for a in articles}
        stim = by_slug["concepts/stimulus-naming"]
        assert stim.title == "Stimulus Naming"
        assert stim.type == "fact"
        assert stim.confidence == 0.9
        assert stim.source_count == 1
        assert not stim.quarantined
        assert "kebab-case" in stim.excerpt

    def test_quarantined_flag_preserved(self, seeded_kb):
        from viewer import load_all_articles
        articles = load_all_articles(seeded_kb)
        qua = next(a for a in articles if a.slug == "concepts/quarantined-one")
        assert qua.quarantined is True

    def test_filter_type(self, seeded_kb):
        from viewer import filter_articles, load_all_articles
        articles = load_all_articles(seeded_kb)
        facts = filter_articles(articles, type_filter="fact", quarantine="hide")
        assert {a.slug for a in facts} == {"concepts/stimulus-naming"}

    def test_filter_min_confidence(self, seeded_kb):
        from viewer import filter_articles, load_all_articles
        articles = load_all_articles(seeded_kb)
        high = filter_articles(articles, min_confidence=0.9)
        slugs = {a.slug for a in high}
        assert "concepts/stimulus-naming" in slugs
        assert "concepts/migrations-policy" in slugs
        assert "concepts/low-conf-plan" not in slugs

    def test_filter_quarantine_hide_vs_only_vs_all(self, seeded_kb):
        from viewer import filter_articles, load_all_articles
        articles = load_all_articles(seeded_kb)
        assert len(filter_articles(articles, quarantine="hide")) == 3
        assert len(filter_articles(articles, quarantine="only")) == 1
        assert len(filter_articles(articles, quarantine="all")) == 4

    def test_filter_search_matches_title_and_slug(self, seeded_kb):
        from viewer import filter_articles, load_all_articles
        articles = load_all_articles(seeded_kb)
        hits = filter_articles(articles, search="stimulus")
        assert {a.slug for a in hits} == {"concepts/stimulus-naming"}

    def test_load_tool_drawer_parses_jsonl(self, seeded_kb):
        from viewer import load_tool_drawer
        events = load_tool_drawer(seeded_kb / "daily", "2026-04-14")
        assert len(events) == 3
        assert events[0]["tool"] == "Edit"
        assert events[-1]["ok"] is False

    def test_summarize_tool_events(self, seeded_kb):
        from viewer import load_tool_drawer, summarize_tool_events
        events = load_tool_drawer(seeded_kb / "daily", "2026-04-14")
        summary = summarize_tool_events(events)
        assert summary["total"] == 3
        assert summary["errors"] == 1
        assert summary["sessions"] == 1
        by_tool = dict(summary["by_tool"])
        assert by_tool["Bash"] == 2
        assert by_tool["Edit"] == 1

    def test_wikilinks_rewritten(self):
        from viewer import _rewrite_wikilinks
        text = "see [[concepts/foo]] and [[concepts/bar|the bar]]"
        out = _rewrite_wikilinks(text)
        assert "[concepts/foo](/articles/concepts/foo)" in out
        assert "[the bar](/articles/concepts/bar)" in out


class TestRoutes:
    def test_index_ok(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.text
        # Key stats visible
        assert "Overview" in body
        # 4 articles fixture
        assert "4" in body
        # Quarantine count
        assert "Contradictions" in body
        # Memory composition chip row surfaces each type
        assert "fact" in body
        assert "preference" in body

    def test_articles_list_all(self, client):
        resp = client.get("/articles")
        assert resp.status_code == 200
        body = resp.text
        # Hides quarantined by default
        assert "Stimulus Naming" in body
        assert "Migrations Policy" in body
        assert "Tentative Rewrite Plan" in body
        assert "Quarantined Article" not in body

    def test_articles_filter_by_type(self, client):
        resp = client.get("/articles?type=preference")
        assert resp.status_code == 200
        body = resp.text
        assert "Migrations Policy" in body
        assert "Stimulus Naming" not in body

    def test_articles_filter_min_confidence(self, client):
        resp = client.get("/articles?min_confidence=0.9")
        assert resp.status_code == 200
        body = resp.text
        assert "Stimulus Naming" in body
        assert "Tentative Rewrite Plan" not in body

    def test_articles_quarantine_only(self, client):
        resp = client.get("/articles?quarantine=only")
        assert resp.status_code == 200
        body = resp.text
        assert "Quarantined Article" in body
        assert "Stimulus Naming" not in body

    def test_articles_search_query(self, client):
        resp = client.get("/articles?q=kebab")
        assert resp.status_code == 200
        body = resp.text
        assert "Stimulus Naming" in body
        assert "Migrations Policy" not in body

    def test_articles_invalid_type_falls_back_to_all(self, client):
        """Unknown type should be ignored, not 500."""
        resp = client.get("/articles?type=banana")
        assert resp.status_code == 200
        body = resp.text
        assert "Stimulus Naming" in body

    def test_article_detail_renders_markdown(self, client):
        resp = client.get("/articles/concepts/stimulus-naming")
        assert resp.status_code == 200
        body = resp.text
        assert "Stimulus Naming" in body
        # Frontmatter rendered as badges
        assert "fact" in body
        # Markdown body rendered (not raw)
        assert "kebab-case" in body
        # Wikilink rewritten into an anchor
        assert "/articles/concepts/migrations-policy" in body

    def test_article_detail_missing_returns_404(self, client):
        resp = client.get("/articles/concepts/does-not-exist")
        assert resp.status_code == 404

    def test_article_detail_trailing_md_redirects(self, client):
        resp = client.get(
            "/articles/concepts/stimulus-naming.md", follow_redirects=False,
        )
        assert resp.status_code == 302
        assert resp.headers["location"].endswith("/articles/concepts/stimulus-naming")

    def test_daily_list(self, client):
        resp = client.get("/daily")
        assert resp.status_code == 200
        assert "2026-04-14" in resp.text

    def test_daily_detail(self, client):
        resp = client.get("/daily/2026-04-14")
        assert resp.status_code == 200
        body = resp.text
        assert "Worked on X" in body
        # Link to drawer visible when drawer exists
        assert "/tools/2026-04-14" in body

    def test_daily_detail_missing_404(self, client):
        resp = client.get("/daily/1999-01-01")
        assert resp.status_code == 404

    def test_tools_list(self, client):
        resp = client.get("/tools")
        assert resp.status_code == 200
        body = resp.text
        assert "2026-04-14" in body
        # Counts visible
        assert "3" in body  # total events

    def test_tools_detail(self, client):
        resp = client.get("/tools/2026-04-14")
        assert resp.status_code == 200
        body = resp.text
        assert "Edit" in body
        assert "Bash" in body
        assert "src/Foo.php" in body
        # Error event visible
        assert "err" in body

    def test_tools_detail_missing_404(self, client):
        resp = client.get("/tools/1999-01-01")
        assert resp.status_code == 404

    def test_contradictions(self, client):
        resp = client.get("/contradictions")
        assert resp.status_code == 200
        body = resp.text
        assert "Quarantined Article" in body
        assert "quarantined" in body

    def test_stats(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        body = resp.text
        # Chroma counts surfaced (via monkeypatched _chroma_stats)
        assert "Articles in Chroma" in body
        # Flush cost table visible
        assert "0.0120" in body or "0.012" in body


class TestBootAppFactory:
    """The app factory must be callable without raising even when state is sparse."""

    def test_create_app_with_empty_dir(self, tmp_path, monkeypatch):
        import config
        kb = tmp_path / "empty_kb"
        kb.mkdir()
        monkeypatch.setattr(config, "KNOWLEDGE_DIR", kb)
        monkeypatch.setattr(config, "DAILY_DIR", kb / "daily")
        monkeypatch.setattr(config, "SCRIPTS_DIR", tmp_path / "nonexistent")
        monkeypatch.setattr(config, "STATE_FILE", tmp_path / "nonexistent" / "state.json")

        import viewer
        monkeypatch.setattr(viewer, "_chroma_stats", lambda: {"articles": 0, "daily_chunks": 0})

        app = viewer.create_app(knowledge_dir=kb)
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Overview" in resp.text
