"""Tests for self-healing lint fixers.

Each test builds a minimal fake knowledge tree, synthesizes a lint
issue dict shaped exactly like ``lint.py`` produces, runs the fixer,
and verifies both the on-disk change and the audit trail entry in
``knowledge/log.md``. The suite intentionally avoids calling the lint
check functions themselves — fixers are tested in isolation so a
regression in one layer doesn't cascade into the other.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def kb(tmp_path, monkeypatch):
    """Redirect config + utils + lint_fixes at an isolated knowledge tree."""
    import config
    import lint_fixes
    import utils

    knowledge_dir = tmp_path / "knowledge"
    concepts_dir = knowledge_dir / "concepts"
    connections_dir = knowledge_dir / "connections"
    qa_dir = knowledge_dir / "qa"
    daily_dir = knowledge_dir / "daily"
    log_file = knowledge_dir / "log.md"
    state_file = tmp_path / "state.json"

    for d in (concepts_dir, connections_dir, qa_dir, daily_dir):
        d.mkdir(parents=True, exist_ok=True)

    for mod in (config, utils, lint_fixes):
        monkeypatch.setattr(mod, "KNOWLEDGE_DIR", knowledge_dir, raising=False)
        monkeypatch.setattr(mod, "CONCEPTS_DIR", concepts_dir, raising=False)
        monkeypatch.setattr(mod, "CONNECTIONS_DIR", connections_dir, raising=False)
        monkeypatch.setattr(mod, "QA_DIR", qa_dir, raising=False)
        monkeypatch.setattr(mod, "DAILY_DIR", daily_dir, raising=False)
        monkeypatch.setattr(mod, "LOG_FILE", log_file, raising=False)
        monkeypatch.setattr(mod, "STATE_FILE", state_file, raising=False)

    monkeypatch.setattr(
        utils, "CONTRADICTIONS_FILE", knowledge_dir / "contradictions.json"
    )
    return knowledge_dir


def _write_article(kb: Path, slug: str, body: str, frontmatter: str = "") -> Path:
    path = kb / f"{slug}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    default_fm = (
        "---\n"
        "title: Test Article\n"
        "type: fact\n"
        "confidence: 0.9\n"
        "updated: 2026-04-12\n"
        "---\n\n"
    )
    path.write_text((frontmatter or default_fm) + body, encoding="utf-8")
    return path


class TestMissingBacklinkFix:
    def test_appends_to_existing_related_concepts(self, kb):
        from lint_fixes import fix_missing_backlink

        _write_article(
            kb, "concepts/source",
            "## Truth\n\nsome text\n\n### Related Concepts\n\n- [[concepts/target]] - linked\n",
        )
        _write_article(
            kb, "concepts/target",
            "## Truth\n\nother text\n\n### Related Concepts\n\n- [[concepts/unrelated]] - existing\n",
        )

        fixed = fix_missing_backlink({
            "check": "missing_backlink",
            "source_slug": "concepts/source",
            "target_slug": "concepts/target",
        })
        assert fixed is True

        target_content = (kb / "concepts" / "target.md").read_text()
        assert "[[concepts/source]]" in target_content
        assert "[[concepts/unrelated]]" in target_content  # existing preserved

    def test_creates_related_concepts_section_when_missing(self, kb):
        from lint_fixes import fix_missing_backlink

        _write_article(
            kb, "concepts/target",
            "## Truth\n\njust some text\n\n---\n\n## Timeline\n\n- event\n",
        )

        fixed = fix_missing_backlink({
            "check": "missing_backlink",
            "source_slug": "concepts/source",
            "target_slug": "concepts/target",
        })
        assert fixed is True

        target_content = (kb / "concepts" / "target.md").read_text()
        assert "### Related Concepts" in target_content
        assert "[[concepts/source]]" in target_content
        # Frontmatter must still be intact at the top of the file.
        assert target_content.startswith("---\ntitle:")
        # Related Concepts must land inside the body AFTER the Truth
        # header but BEFORE the Truth/Timeline horizontal rule.
        related_idx = target_content.index("### Related Concepts")
        truth_idx = target_content.index("## Truth")
        timeline_idx = target_content.index("## Timeline")
        assert truth_idx < related_idx < timeline_idx

    def test_idempotent_when_backlink_already_present(self, kb):
        from lint_fixes import fix_missing_backlink

        _write_article(
            kb, "concepts/target",
            "## Truth\n\n[[concepts/source]] already referenced here\n",
        )

        fixed = fix_missing_backlink({
            "check": "missing_backlink",
            "source_slug": "concepts/source",
            "target_slug": "concepts/target",
        })
        assert fixed is False

    def test_returns_false_when_target_missing(self, kb):
        from lint_fixes import fix_missing_backlink
        fixed = fix_missing_backlink({
            "check": "missing_backlink",
            "source_slug": "concepts/source",
            "target_slug": "concepts/nonexistent",
        })
        assert fixed is False


class TestBrokenLinkReEntrancy:
    """Regression tests for the bug where re-running broken_link fixer
    against an already-wrapped link produced nested HTML comments like
    ``<!-- BROKEN-LINK: <!-- BROKEN-LINK: [[...]] --> -->``. The fixer
    must detect a prior wrap and refuse to act on it.
    """

    def test_already_wrapped_link_is_not_double_wrapped(self, kb):
        from lint_fixes import fix_broken_link

        # Source already has a previously-wrapped broken link
        _write_article(
            kb, "concepts/source",
            "## Truth\n\n"
            "- Cite: <!-- BROKEN-LINK: [[concepts/missing-thing]] --> something\n",
        )

        fixed = fix_broken_link({
            "check": "broken_link",
            "file": "concepts/source.md",
            "broken_target": "concepts/missing-thing",
        })
        assert fixed is False  # nothing to do, re-entrancy blocked

        content = (kb / "concepts" / "source.md").read_text()
        # No nested wrapper allowed
        assert "BROKEN-LINK: <!-- BROKEN-LINK" not in content
        # Original wrapper stays
        assert "<!-- BROKEN-LINK: [[concepts/missing-thing]] -->" in content

    def test_partial_wrap_still_fixes_unwrapped_occurrence(self, kb):
        """If one occurrence is already wrapped but another isn't, only
        the unwrapped one should get wrapped on this run.
        """
        from lint_fixes import fix_broken_link

        _write_article(
            kb, "concepts/source",
            "## Truth\n\n"
            "First mention: <!-- BROKEN-LINK: [[concepts/x]] -->\n"
            "Second mention bare: [[concepts/x]]\n",
        )

        fixed = fix_broken_link({
            "check": "broken_link",
            "file": "concepts/source.md",
            "broken_target": "concepts/x",
        })
        assert fixed is True

        content = (kb / "concepts" / "source.md").read_text()
        # Should now have exactly two wrapped occurrences, no nested wrapper.
        assert content.count("<!-- BROKEN-LINK: [[concepts/x]] -->") == 2
        assert "BROKEN-LINK: <!-- BROKEN-LINK" not in content


class TestBrokenSourceAnchorReEntrancy:
    def test_already_wrapped_anchor_is_not_double_wrapped(self, kb):
        from lint_fixes import fix_broken_source_anchor

        _write_article(
            kb, "concepts/source",
            "## Truth\n\n- fact <!-- BROKEN-SRC: [src:daily/missing.md] -->\n",
        )

        fixed = fix_broken_source_anchor({
            "check": "broken_source_anchor",
            "file": "concepts/source.md",
            "broken_anchor": "daily/missing.md",
        })
        assert fixed is False
        content = (kb / "concepts" / "source.md").read_text()
        assert "BROKEN-SRC: <!-- BROKEN-SRC" not in content


class TestBrokenLinkFix:
    def test_fuzzy_match_rewrites_link(self, kb):
        from lint_fixes import fix_broken_link

        _write_article(kb, "concepts/target-slug", "## Truth\n\ncontent\n")
        # Source references the target with a one-char typo
        _write_article(
            kb, "concepts/source",
            "## Truth\n\nSee [[concepts/targett-slug]] for details\n",
        )

        fixed = fix_broken_link({
            "check": "broken_link",
            "file": "concepts/source.md",
            "broken_target": "concepts/targett-slug",
        })
        assert fixed is True

        src = (kb / "concepts" / "source.md").read_text()
        assert "[[concepts/target-slug]]" in src
        assert "targett-slug" not in src

    def test_no_match_comments_out(self, kb):
        from lint_fixes import fix_broken_link

        _write_article(kb, "concepts/existing", "## Truth\n\ncontent\n")
        _write_article(
            kb, "concepts/source",
            "## Truth\n\nSee [[concepts/completely-unrelated-thing]]\n",
        )

        fixed = fix_broken_link({
            "check": "broken_link",
            "file": "concepts/source.md",
            "broken_target": "concepts/completely-unrelated-thing",
        })
        assert fixed is True

        src = (kb / "concepts" / "source.md").read_text()
        assert "<!-- BROKEN-LINK: [[concepts/completely-unrelated-thing]] -->" in src
        assert "[[concepts/completely-unrelated-thing]]" not in src.replace(
            "<!-- BROKEN-LINK: [[concepts/completely-unrelated-thing]] -->", ""
        )


class TestStaleArticleFix:
    def test_clears_cached_hash(self, kb):
        import utils

        from lint_fixes import fix_stale_article

        # Seed state.json with an ingested daily log
        utils.save_state({
            "ingested_daily": {
                "2026-04-12.md": {"hash": "old-hash-value", "compiled_at": "2026-04-12"},
            },
            "total_cost": 0.0,
        })

        fixed = fix_stale_article({
            "check": "stale_article",
            "daily_name": "2026-04-12.md",
        })
        assert fixed is True

        state = utils.load_state()
        assert state["ingested_daily"]["2026-04-12.md"]["hash"] == ""

    def test_returns_false_when_daily_not_ingested(self, kb):
        import utils

        from lint_fixes import fix_stale_article
        utils.save_state({"ingested_daily": {}})
        fixed = fix_stale_article({
            "check": "stale_article",
            "daily_name": "2026-04-99.md",
        })
        assert fixed is False


class TestBrokenSourceAnchorFix:
    def test_fuzzy_match_rewrites_anchor(self, kb):
        from lint_fixes import fix_broken_source_anchor

        # Create a real daily file the anchor SHOULD point at
        (kb / "daily" / "2026-04-12.md").write_text("daily content", encoding="utf-8")

        _write_article(
            kb, "concepts/source",
            "## Truth\n\n- fact [src:daily/2026-04-2.md]\n",
        )

        fixed = fix_broken_source_anchor({
            "check": "broken_source_anchor",
            "file": "concepts/source.md",
            "broken_anchor": "daily/2026-04-2.md",
        })
        assert fixed is True

        src = (kb / "concepts" / "source.md").read_text()
        assert "[src:daily/2026-04-12.md]" in src

    def test_no_match_comments_out(self, kb):
        from lint_fixes import fix_broken_source_anchor

        _write_article(
            kb, "concepts/source",
            "## Truth\n\n- fact [src:daily/never-existed-xyz.md]\n",
        )

        fixed = fix_broken_source_anchor({
            "check": "broken_source_anchor",
            "file": "concepts/source.md",
            "broken_anchor": "daily/never-existed-xyz.md",
        })
        assert fixed is True

        src = (kb / "concepts" / "source.md").read_text()
        assert "<!-- BROKEN-SRC: [src:daily/never-existed-xyz.md] -->" in src


class TestMissingMemoryTypeFix:
    def test_infers_from_feedback_prefix(self, kb):
        from lint_fixes import fix_missing_memory_type

        # No type: field in frontmatter
        _write_article(
            kb, "concepts/feedback_tailwind_rebuild",
            "some content",
            frontmatter="---\ntitle: Tailwind Rebuild\nconfidence: 0.9\n---\n\n",
        )

        fixed = fix_missing_memory_type({
            "check": "missing_memory_type",
            "file": "concepts/feedback_tailwind_rebuild.md",
        })
        assert fixed is True

        content = (kb / "concepts" / "feedback_tailwind_rebuild.md").read_text()
        assert "type: preference" in content

    def test_skips_when_no_confident_prefix(self, kb):
        from lint_fixes import fix_missing_memory_type

        _write_article(
            kb, "concepts/random-article",
            "some content",
            frontmatter="---\ntitle: Random\nconfidence: 0.9\n---\n\n",
        )

        fixed = fix_missing_memory_type({
            "check": "missing_memory_type",
            "file": "concepts/random-article.md",
        })
        assert fixed is False


class TestAuditTrail:
    def test_successful_fix_appends_to_log_md(self, kb):
        from lint_fixes import fix_missing_backlink

        _write_article(
            kb, "concepts/target",
            "## Truth\n\njust text\n",
        )

        fix_missing_backlink({
            "check": "missing_backlink",
            "source_slug": "concepts/source",
            "target_slug": "concepts/target",
        })

        log_content = (kb / "log.md").read_text()
        assert "lint-fix | concepts/target" in log_content
        assert "Check: missing_backlink" in log_content
        assert "[[concepts/source]]" in log_content


class TestFixRegistryDispatch:
    def test_apply_fixes_counts_successes_and_attempts(self, kb):
        from lint_fixes import apply_fixes

        _write_article(kb, "concepts/target", "## Truth\n\ntext\n")

        issues = [
            # Fixable + succeeds
            {
                "check": "missing_backlink",
                "source_slug": "concepts/source",
                "target_slug": "concepts/target",
            },
            # Fixable but will fail (target missing)
            {
                "check": "missing_backlink",
                "source_slug": "concepts/x",
                "target_slug": "concepts/nonexistent",
            },
            # Not in registry — should be skipped, not counted as attempted
            {"check": "sparse_article", "file": "concepts/x.md"},
        ]

        fixed, attempted = apply_fixes(issues)
        assert attempted == 2
        assert fixed == 1

    def test_only_checks_filter_narrows_scope(self, kb):
        """--fix-only equivalent: apply_fixes(only_checks={...}) skips others."""
        import utils

        from lint_fixes import apply_fixes

        _write_article(kb, "concepts/target", "## Truth\n\ntext\n")
        utils.save_state({
            "ingested_daily": {
                "2026-04-12.md": {"hash": "old", "compiled_at": "2026-04-12"},
            },
        })

        issues = [
            {
                "check": "missing_backlink",
                "source_slug": "concepts/source",
                "target_slug": "concepts/target",
            },
            {"check": "stale_article", "daily_name": "2026-04-12.md"},
        ]

        # Scope to stale_article only — the backlink must NOT be applied.
        fixed, attempted = apply_fixes(issues, only_checks={"stale_article"})
        assert attempted == 1
        assert fixed == 1

        target_content = (kb / "concepts" / "target.md").read_text()
        assert "[[concepts/source]]" not in target_content
        # And the state hash must have been cleared by the stale fixer.
        assert utils.load_state()["ingested_daily"]["2026-04-12.md"]["hash"] == ""

    def test_only_checks_empty_set_skips_everything(self, kb):
        from lint_fixes import apply_fixes

        _write_article(kb, "concepts/target", "## Truth\n\ntext\n")
        issues = [{
            "check": "missing_backlink",
            "source_slug": "concepts/source",
            "target_slug": "concepts/target",
        }]
        fixed, attempted = apply_fixes(issues, only_checks=set())
        assert attempted == 0
        assert fixed == 0
