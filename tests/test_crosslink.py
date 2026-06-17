"""Tests for the cross-linker / wikilink backfill pass.

The cross-linker scans article bodies for UNLINKED prose mentions of other
articles' titles/aliases and appends ``[[<slug>]]`` bullets to a
``### Related Concepts`` section (under ``## Truth``). It is a pure-Python,
zero-LLM pass. We test the pure functions against synthetic strings and
tmp_path — no dependence on live project state, network, or Chroma.
"""
from pathlib import Path

from scripts import crosslink


# ── build_title_index ──────────────────────────────────────────────────

def test_build_title_index_maps_titles_and_aliases_lowercased():
    articles = [
        {
            "slug": "concepts/foo",
            "title": "Foo Bar Baz",
            "aliases": ["FooBarAlias", "another name"],
            "body": "",
        },
        {
            "slug": "concepts/qux",
            "title": "Qux Widget",
            "aliases": [],
            "body": "",
        },
    ]
    index = crosslink.build_title_index(articles)
    assert index["foo bar baz"] == "concepts/foo"
    assert index["foobaralias"] == "concepts/foo"
    assert index["another name"] == "concepts/foo"
    assert index["qux widget"] == "concepts/qux"


def test_build_title_index_skips_short_titles():
    articles = [
        {"slug": "concepts/a", "title": "Foo", "aliases": ["xy"], "body": ""},
        {"slug": "concepts/b", "title": "Long Enough Title", "aliases": [], "body": ""},
    ]
    index = crosslink.build_title_index(articles)
    # "Foo" (3 chars) and "xy" (2 chars) are <= 3 → skipped
    assert "foo" not in index
    assert "xy" not in index
    assert "long enough title" in index


# ── existing_links ─────────────────────────────────────────────────────

def test_existing_links_extracts_wikilink_slugs():
    body = "See [[concepts/foo]] and [[concepts/bar]]{depends_on} here."
    assert crosslink.existing_links(body) == {"concepts/foo", "concepts/bar"}


def test_existing_links_empty_when_none():
    assert crosslink.existing_links("plain text no links") == set()


# ── masking (code fences, inline code, frontmatter, links) ─────────────

def test_mask_removes_fenced_code_blocks():
    body = "before\n```\nmention Foo Bar Baz inside code\n```\nafter"
    masked = crosslink.mask_non_prose(body)
    assert "Foo Bar Baz" not in masked
    assert "before" in masked
    assert "after" in masked


def test_mask_removes_inline_code():
    body = "use `Foo Bar Baz` in code but mention Foo Bar Baz in prose"
    masked = crosslink.mask_non_prose(body)
    # The inline-code occurrence is gone; the prose occurrence survives.
    assert masked.count("Foo Bar Baz") == 1


def test_mask_removes_frontmatter():
    body = "---\ntitle: Foo Bar Baz\naliases: [Foo Bar Baz]\n---\nprose Other Title here"
    masked = crosslink.mask_non_prose(body)
    assert "Foo Bar Baz" not in masked
    assert "Other Title" in masked


def test_mask_removes_existing_wikilinks_and_md_links():
    body = "see [[concepts/foo]] and [Foo Bar Baz](http://x) but Other Title in prose"
    masked = crosslink.mask_non_prose(body)
    assert "concepts/foo" not in masked
    # The display text of the markdown link is masked too.
    assert "Foo Bar Baz" not in masked
    assert "Other Title" in masked


# ── find_missing_links ─────────────────────────────────────────────────

def _index():
    articles = [
        {"slug": "concepts/foo", "title": "Foo Bar Baz", "aliases": ["FooBarAlias"], "body": ""},
        {"slug": "concepts/qux", "title": "Qux Widget", "aliases": [], "body": ""},
        {"slug": "concepts/self", "title": "Self Article", "aliases": [], "body": ""},
    ]
    return crosslink.build_title_index(articles)


def test_find_missing_links_detects_prose_mention():
    index = _index()
    body = "This article talks about Foo Bar Baz at length."
    assert crosslink.find_missing_links(body, "concepts/me", index) == ["concepts/foo"]


def test_find_missing_links_matches_alias():
    index = _index()
    body = "We rely on FooBarAlias for the widget."
    assert crosslink.find_missing_links(body, "concepts/me", index) == ["concepts/foo"]


def test_find_missing_links_excludes_self():
    index = _index()
    body = "This is the Self Article describing itself."
    assert crosslink.find_missing_links(body, "concepts/self", index) == []


def test_find_missing_links_dedupes_existing_links():
    index = _index()
    body = "Already linked [[concepts/foo]] but also names Foo Bar Baz again."
    assert crosslink.find_missing_links(body, "concepts/me", index) == []


def test_find_missing_links_whole_word_only():
    index = _index()
    # "Qux Widget" appears only as a substring of a larger token → no match.
    body = "The QuxWidgetXYZ component does things."
    assert crosslink.find_missing_links(body, "concepts/me", index) == []


def test_find_missing_links_ignores_code_and_frontmatter():
    index = _index()
    body = (
        "---\ntitle: Me\n---\n"
        "```\nFoo Bar Baz in code\n```\n"
        "and `Qux Widget` inline."
    )
    assert crosslink.find_missing_links(body, "concepts/me", index) == []


def test_find_missing_links_dedupes_repeated_mentions():
    index = _index()
    body = "Foo Bar Baz here, Foo Bar Baz there, Foo Bar Baz everywhere."
    assert crosslink.find_missing_links(body, "concepts/me", index) == ["concepts/foo"]


def test_find_missing_links_is_case_insensitive():
    index = _index()
    body = "We discuss foo bar baz in lowercase."
    assert crosslink.find_missing_links(body, "concepts/me", index) == ["concepts/foo"]


def test_find_missing_links_orders_by_first_appearance():
    index = _index()
    body = "First Qux Widget then later Foo Bar Baz."
    result = crosslink.find_missing_links(body, "concepts/me", index)
    assert result == ["concepts/qux", "concepts/foo"]


# ── add_related_links ──────────────────────────────────────────────────

ARTICLE_NO_SECTION = """---
title: Me
---

# Me

## Truth

### Observed

- some fact

### Synthesized

- some inference

---

## Timeline

### 2026-01-01 | source

- event
"""


def test_add_related_links_creates_section_when_absent():
    out = crosslink.add_related_links(ARTICLE_NO_SECTION, ["concepts/foo", "concepts/qux"])
    assert "### Related Concepts" in out
    assert "- [[concepts/foo]] — auto-linked mention" in out
    assert "- [[concepts/qux]] — auto-linked mention" in out
    # The new section must sit inside ## Truth, before the timeline separator.
    truth_idx = out.index("## Truth")
    rel_idx = out.index("### Related Concepts")
    timeline_idx = out.index("## Timeline")
    assert truth_idx < rel_idx < timeline_idx


ARTICLE_WITH_SECTION = """---
title: Me
---

# Me

## Truth

### Observed

- some fact

### Related Concepts

- [[concepts/existing]] — auto-linked mention

---

## Timeline

- event
"""


def test_add_related_links_appends_to_existing_section():
    out = crosslink.add_related_links(ARTICLE_WITH_SECTION, ["concepts/foo"])
    assert "- [[concepts/existing]] — auto-linked mention" in out
    assert "- [[concepts/foo]] — auto-linked mention" in out
    # Only one Related Concepts heading.
    assert out.count("### Related Concepts") == 1


def test_add_related_links_is_idempotent():
    once = crosslink.add_related_links(ARTICLE_WITH_SECTION, ["concepts/foo"])
    twice = crosslink.add_related_links(once, ["concepts/foo"])
    assert once == twice
    assert twice.count("- [[concepts/foo]] — auto-linked mention") == 1


def test_add_related_links_no_slugs_returns_unchanged():
    assert crosslink.add_related_links(ARTICLE_WITH_SECTION, []) == ARTICLE_WITH_SECTION


def test_add_related_links_no_timeline_appends_at_end():
    text = "---\ntitle: Me\n---\n\n## Truth\n\n### Observed\n\n- fact\n"
    out = crosslink.add_related_links(text, ["concepts/foo"])
    assert "### Related Concepts" in out
    assert "- [[concepts/foo]] — auto-linked mention" in out


# ── end-to-end on a tmp article set ────────────────────────────────────

def test_load_articles_reads_frontmatter(tmp_path):
    concepts = tmp_path / "concepts"
    concepts.mkdir()
    (concepts / "foo.md").write_text(
        "---\ntitle: Foo Bar Baz\naliases: [FooBarAlias]\n---\n## Truth\n\nbody",
        encoding="utf-8",
    )
    arts = crosslink.load_articles(tmp_path)
    by_slug = {a["slug"]: a for a in arts}
    assert "concepts/foo" in by_slug
    assert by_slug["concepts/foo"]["title"] == "Foo Bar Baz"
    assert by_slug["concepts/foo"]["aliases"] == ["FooBarAlias"]
