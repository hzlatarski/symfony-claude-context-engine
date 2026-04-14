"""Tests for the wikilink-scope guard on lint checks.

Regression test for the bug where ``check_broken_links`` and
``check_missing_backlinks`` flagged ``[[sources/...]]`` citations as
broken — they're external file references, not wiki articles. Without
the scope guard, the ``broken_link`` fixer would then "repair" them by
wrapping them in HTML comments, destroying the source provenance.
"""
from __future__ import annotations


def _patch_kb(tmp_path, monkeypatch):
    import config
    import lint
    import utils

    for mod in (config, utils, lint):
        monkeypatch.setattr(mod, "KNOWLEDGE_DIR", tmp_path, raising=False)
        monkeypatch.setattr(mod, "CONCEPTS_DIR", tmp_path / "concepts", raising=False)
        monkeypatch.setattr(mod, "CONNECTIONS_DIR", tmp_path / "connections", raising=False)
        monkeypatch.setattr(mod, "QA_DIR", tmp_path / "qa", raising=False)
        monkeypatch.setattr(mod, "DAILY_DIR", tmp_path / "daily", raising=False)

    for d in ("concepts", "connections", "qa", "daily"):
        (tmp_path / d).mkdir(exist_ok=True)


class TestBrokenLinksIgnoresNonWikiPrefixes:
    def test_sources_prefix_not_flagged(self, tmp_path, monkeypatch):
        _patch_kb(tmp_path, monkeypatch)
        import lint

        (tmp_path / "concepts" / "article.md").write_text(
            "---\ntitle: A\ntype: fact\n---\n\n## Truth\n\n"
            "- Cited from [[sources/design-specs/2026-03-26-cost-design.md]]\n",
            encoding="utf-8",
        )

        issues = lint.check_broken_links()
        # Source citation must NOT be flagged as broken
        assert not any(
            "sources/design-specs" in issue["detail"] for issue in issues
        )

    def test_concepts_prefix_still_flagged_when_missing(self, tmp_path, monkeypatch):
        _patch_kb(tmp_path, monkeypatch)
        import lint

        (tmp_path / "concepts" / "article.md").write_text(
            "---\ntitle: A\ntype: fact\n---\n\n## Truth\n\n"
            "- See [[concepts/does-not-exist]]\n",
            encoding="utf-8",
        )

        issues = lint.check_broken_links()
        slugs = [issue["broken_target"] for issue in issues if issue["check"] == "broken_link"]
        assert "concepts/does-not-exist" in slugs


class TestExtractWikilinksSkipsCommentedLinks:
    """Regression: a wikilink wrapped in an HTML comment is a 'fixed'
    sentinel from broken_link's comment-out fallback. It must not be
    re-extracted, otherwise lint flags the same dead link forever.
    """

    def test_commented_wikilink_not_extracted(self):
        from utils import extract_wikilinks
        content = (
            "Live link: [[concepts/alive]]\n"
            "Dead link: <!-- BROKEN-LINK: [[concepts/dead]] -->\n"
        )
        links = extract_wikilinks(content)
        assert "concepts/alive" in links
        assert "concepts/dead" not in links

    def test_multiline_comment_skipped(self):
        from utils import extract_wikilinks
        content = (
            "<!--\n"
            "[[concepts/multi-line-dead]]\n"
            "-->\n"
            "[[concepts/live]]\n"
        )
        links = extract_wikilinks(content)
        assert "concepts/live" in links
        assert "concepts/multi-line-dead" not in links


class TestMissingBacklinksIgnoresNonWikiPrefixes:
    def test_sources_prefix_does_not_create_backlink_demand(self, tmp_path, monkeypatch):
        _patch_kb(tmp_path, monkeypatch)
        import lint

        (tmp_path / "concepts" / "article.md").write_text(
            "---\ntitle: A\ntype: fact\n---\n\n## Truth\n\n"
            "- Cited from [[sources/design-specs/foo.md]]\n",
            encoding="utf-8",
        )

        issues = lint.check_missing_backlinks()
        # Sources don't get backlink expectations
        assert all(
            issue.get("target_slug", "").startswith(("concepts/", "connections/", "qa/"))
            for issue in issues
        )
