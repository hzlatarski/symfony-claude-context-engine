"""Tests for the BM25 keyword index.

Covers the three moving parts that can go wrong without the rest of the
pipeline noticing:

1. Tokenization — splits identifier-shaped strings correctly so real
   code tokens hit the index.
2. Ranking — retrieves the right article when the query matches a
   literal identifier that vector search would miss.
3. Filters — type / confidence / zone / quarantine gates behave the
   same way ``vector_store.search_articles`` does.

Index invalidation via the ``state.json`` mtime sentinel is covered in
a separate test so a file-watching regression surfaces clearly.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _write_article(
    root: Path,
    slug: str,
    *,
    title: str,
    observed: str,
    synthesized: str = "",
    type_: str = "fact",
    confidence: float = 0.9,
    pinned: bool = False,
) -> None:
    path = root / f"{slug}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter_lines = [
        "---",
        f"title: {title}",
        f"type: {type_}",
        f"confidence: {confidence}",
        f"pinned: {str(pinned).lower()}",
        "updated: 2026-04-12",
        "---",
        "",
        "## Truth",
        "",
        "### Observed",
        "",
        observed,
        "",
    ]
    if synthesized:
        frontmatter_lines.extend([
            "### Synthesized",
            "",
            synthesized,
            "",
        ])
    path.write_text("\n".join(frontmatter_lines), encoding="utf-8")


@pytest.fixture
def kb(tmp_path, monkeypatch):
    """Redirect the knowledge tree, state file, and quarantine to tmp_path.

    Monkeypatches ``config`` and legacy path copies in ``utils`` so every
    reader sees the temporary tree instead of the real one.
    """
    import bm25_store
    import config
    import utils

    knowledge_dir = tmp_path / "knowledge"
    concepts_dir = knowledge_dir / "concepts"
    connections_dir = knowledge_dir / "connections"
    qa_dir = knowledge_dir / "qa"
    state_file = tmp_path / "state.json"

    for mod in (config, utils):
        monkeypatch.setattr(mod, "KNOWLEDGE_DIR", knowledge_dir, raising=False)
        monkeypatch.setattr(mod, "CONCEPTS_DIR", concepts_dir, raising=False)
        monkeypatch.setattr(mod, "CONNECTIONS_DIR", connections_dir, raising=False)
        monkeypatch.setattr(mod, "QA_DIR", qa_dir, raising=False)
        monkeypatch.setattr(mod, "STATE_FILE", state_file, raising=False)

    monkeypatch.setattr(
        utils, "CONTRADICTIONS_FILE", knowledge_dir / "contradictions.json"
    )
    bm25_store.invalidate()
    yield concepts_dir
    bm25_store.invalidate()


class TestTokenize:
    def test_splits_camel_case(self):
        from bm25_store import tokenize
        tokens = tokenize("HybridEmailValidationService")
        assert "hybrid" in tokens
        assert "email" in tokens
        assert "validation" in tokens
        assert "service" in tokens

    def test_splits_snake_and_kebab(self):
        from bm25_store import tokenize
        tokens = tokenize("stimulus_naming-convention")
        assert "stimulus" in tokens
        assert "naming" in tokens
        assert "convention" in tokens

    def test_drops_single_chars(self):
        from bm25_store import tokenize
        tokens = tokenize("a b cd ef")
        assert tokens == ["cd", "ef"]

    def test_lowercases_everything(self):
        from bm25_store import tokenize
        tokens = tokenize("FOO Bar")
        assert tokens == ["foo", "bar"]

    def test_handles_empty_and_whitespace(self):
        from bm25_store import tokenize
        assert tokenize("") == []
        assert tokenize("   \n\t") == []


class TestSearchBasics:
    def test_literal_identifier_match_beats_fuzzy(self, kb):
        from bm25_store import search_articles
        _write_article(
            kb, "email-validation",
            title="Email Validation",
            observed="HybridEmailValidationService runs on form submit.",
        )
        _write_article(
            kb, "unrelated-thing",
            title="Unrelated",
            observed="Generic article about other topics entirely.",
        )

        results = search_articles("HybridEmailValidationService", limit=5)
        assert len(results) >= 1
        assert results[0]["slug"] == "concepts/email-validation"

    def test_empty_query_returns_empty(self, kb):
        from bm25_store import search_articles
        _write_article(kb, "foo", title="Foo", observed="some text about foo")
        assert search_articles("", limit=5) == []
        assert search_articles("   ", limit=5) == []

    def test_unknown_term_returns_empty(self, kb):
        from bm25_store import search_articles
        _write_article(kb, "foo", title="Foo", observed="some text about foo")
        results = search_articles("zzzzzz-never-seen-term", limit=5)
        assert results == []

    def test_result_shape_matches_vector_store(self, kb):
        from bm25_store import search_articles
        _write_article(
            kb, "shape-test",
            title="Shape Test",
            observed="content about shapes for testing",
        )
        results = search_articles("shapes testing", limit=5)
        assert len(results) >= 1
        r = results[0]
        assert set(r.keys()) >= {"id", "slug", "text", "metadata", "score", "distance"}
        assert r["distance"] == -r["score"]
        assert r["metadata"]["zone"] in {"observed", "synthesized"}


class TestFilters:
    def test_type_filter(self, kb):
        from bm25_store import search_articles
        _write_article(
            kb, "fact-article",
            title="Fact", observed="fact about migration rules",
            type_="fact",
        )
        _write_article(
            kb, "pref-article",
            title="Pref", observed="preference about migration rules",
            type_="preference",
        )
        results = search_articles("migration rules", limit=5, type_filter="preference")
        slugs = {r["slug"] for r in results}
        assert "concepts/pref-article" in slugs
        assert "concepts/fact-article" not in slugs

    def test_min_confidence_filter(self, kb):
        from bm25_store import search_articles
        _write_article(
            kb, "firm", title="Firm",
            observed="firm decision about payments",
            confidence=0.9,
        )
        _write_article(
            kb, "tentative", title="Tentative",
            observed="tentative decision about payments",
            confidence=0.3,
        )
        results = search_articles("decision payments", limit=5, min_confidence=0.5)
        slugs = {r["slug"] for r in results}
        assert "concepts/firm" in slugs
        assert "concepts/tentative" not in slugs

    def test_zone_filter(self, kb):
        from bm25_store import search_articles
        _write_article(
            kb, "zoned",
            title="Zoned",
            observed="raw observation about routing",
            synthesized="inference derived from routing patterns",
        )
        results = search_articles("routing", limit=5, zone_filter="synthesized")
        assert len(results) >= 1
        assert all(r["metadata"]["zone"] == "synthesized" for r in results)


class TestInvalidation:
    def test_sentinel_mtime_triggers_rebuild(self, kb):
        import os
        import time

        import bm25_store

        _write_article(kb, "first", title="First", observed="alpha beta gamma")
        assert len(bm25_store.search_articles("alpha", limit=5)) == 1

        _write_article(kb, "second", title="Second", observed="alpha delta epsilon")
        # Touch state.json so the sentinel mtime advances. time.sleep(0)
        # is not enough on Windows where FS mtime resolution is coarse —
        # bump the mtime explicitly instead.
        import config
        state = config.STATE_FILE
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text("{}", encoding="utf-8")
        future = time.time() + 10
        os.utime(state, (future, future))

        results = bm25_store.search_articles("alpha", limit=5)
        slugs = {r["slug"] for r in results}
        assert "concepts/first" in slugs
        assert "concepts/second" in slugs

    def test_invalidate_clears_state(self, kb):
        import bm25_store

        _write_article(kb, "foo", title="Foo", observed="about foo")
        bm25_store.search_articles("foo", limit=5)
        assert bm25_store.stats()["built"] is True

        bm25_store.invalidate()
        assert bm25_store._index is None
        assert bm25_store._docs == []


class TestTitleIsIndexed:
    """The title is deliberately folded into the BM25 token stream.

    ``_iter_article_zones`` tokenizes ``f"{title} {zone_text}"`` while the
    stored ``text`` is the zone body alone. Anything that rebuilds the
    index from ``text`` silently drops title terms from search — a
    regression no other test in this file would notice, because every
    other fixture happens to repeat title words in the body.
    """

    def test_query_matching_only_the_title_still_retrieves(self, kb):
        import bm25_store

        _write_article(
            kb,
            "doctrine-uuid-param-binding",
            title="Doctrine UUID parameter binding",
            observed="Bind the entity identifier explicitly on foreign key comparisons.",
        )
        _write_article(
            kb,
            "unrelated",
            title="Tailwind purge behaviour",
            observed="Utility classes are removed at build time.",
        )
        bm25_store.invalidate()

        # "doctrine" and "uuid" appear ONLY in the title, never in the body.
        results = bm25_store.search_articles("doctrine uuid", limit=5)
        assert [r["slug"] for r in results][:1] == ["concepts/doctrine-uuid-param-binding"]

    def test_body_only_terms_still_retrieve(self, kb):
        import bm25_store

        _write_article(
            kb,
            "pooling",
            title="Database notes",
            observed="Postgres connection pooling exhausted the available slots.",
        )
        bm25_store.invalidate()

        results = bm25_store.search_articles("pooling exhausted", limit=5)
        assert [r["slug"] for r in results] == ["concepts/pooling"]
