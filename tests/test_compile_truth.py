"""Pytest suite for compile_truth.py scoring functions."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from scripts.compile_truth import score_confidence


class TestConfidenceDecay:
    """Confidence should decay over time based on updated: date."""

    def test_fresh_article_keeps_full_confidence(self):
        today = date(2026, 4, 12)
        updated = today.isoformat()
        result = score_confidence(0.9, source_count=1, updated=updated, today=today)
        assert 0.88 <= result <= 1.0, f"Fresh article should keep ~full confidence, got {result}"

    def test_90_day_old_article_decays_to_half(self):
        today = date(2026, 4, 12)
        updated = (today - timedelta(days=90)).isoformat()
        result = score_confidence(0.9, source_count=1, updated=updated, today=today)
        # Half-life 90 days -> 0.9 * 0.5 = 0.45 (plus source boost)
        assert 0.40 <= result <= 0.55, f"90-day-old should be ~halved, got {result}"

    def test_old_article_never_goes_below_floor(self):
        today = date(2026, 4, 12)
        updated = (today - timedelta(days=3650)).isoformat()  # 10 years
        result = score_confidence(0.9, source_count=1, updated=updated, today=today)
        assert result >= 0.05, f"Confidence floor violated, got {result}"

    def test_multiple_sources_boost_survives_decay(self):
        today = date(2026, 4, 12)
        updated = (today - timedelta(days=90)).isoformat()
        result = score_confidence(0.9, source_count=5, updated=updated, today=today)
        # Source boost applies after decay: 0.45 + 0.20 = 0.65
        assert 0.55 <= result <= 0.75, f"Source boost should survive decay, got {result}"

    def test_none_updated_uses_conservative_baseline(self):
        today = date(2026, 4, 12)
        result = score_confidence(0.9, source_count=1, updated=None, today=today)
        # Unknown date: treat as moderately old (30 days) to avoid rewarding undated articles
        assert 0.55 <= result <= 0.85, f"Undated article got {result}"

    def test_future_dated_article_falls_back_to_undated_baseline(self):
        today = date(2026, 4, 12)
        updated = (today + timedelta(days=365)).isoformat()
        result = score_confidence(0.9, source_count=1, updated=updated, today=today)
        # Future date is treated as a data error; same 30-day baseline as None
        assert 0.55 <= result <= 0.85, f"Future-dated article got {result}"


import json


class TestQuarantine:
    """Contradicted articles should be excluded from compiled truth."""

    def test_load_contradictions_missing_file_returns_empty(self, tmp_path):
        from scripts.utils import load_contradictions
        assert load_contradictions(tmp_path / "contradictions.json") == set()

    def test_save_and_load_roundtrip(self, tmp_path):
        from scripts.utils import save_contradictions, load_contradictions
        path = tmp_path / "contradictions.json"
        slugs = {"concepts/foo", "concepts/bar"}
        save_contradictions(slugs, path)
        assert load_contradictions(path) == slugs

    def test_save_is_idempotent_and_sorted(self, tmp_path):
        from scripts.utils import save_contradictions
        path = tmp_path / "contradictions.json"
        save_contradictions({"concepts/z", "concepts/a"}, path)
        data = json.loads(path.read_text())
        assert data["quarantined"] == ["concepts/a", "concepts/z"]
        assert "updated" in data

    def test_load_contradictions_raises_on_corrupted_json(self, tmp_path):
        from scripts.utils import load_contradictions
        path = tmp_path / "contradictions.json"
        path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(RuntimeError, match="Corrupted quarantine"):
            load_contradictions(path)

    def test_load_contradictions_rejects_non_list_quarantined(self, tmp_path):
        import json as _json
        from scripts.utils import load_contradictions
        path = tmp_path / "contradictions.json"
        path.write_text(_json.dumps({"quarantined": "concepts/foo"}), encoding="utf-8")
        with pytest.raises(RuntimeError, match="schema"):
            load_contradictions(path)

    def test_load_contradictions_rejects_non_string_slug(self, tmp_path):
        import json as _json
        from scripts.utils import load_contradictions
        path = tmp_path / "contradictions.json"
        path.write_text(_json.dumps({"quarantined": ["concepts/ok", 42]}), encoding="utf-8")
        with pytest.raises(RuntimeError, match="schema"):
            load_contradictions(path)


class TestCompileTruthQuarantine:
    """compile_truth should skip articles that are in contradictions.json."""

    def test_quarantined_slug_excluded_from_output(self, tmp_path, monkeypatch):
        # Set up a minimal knowledge dir
        knowledge = tmp_path / "knowledge"
        (knowledge / "concepts").mkdir(parents=True)
        (knowledge / "connections").mkdir()

        article_a = knowledge / "concepts" / "good.md"
        article_a.write_text(
            "---\ntitle: Good\nupdated: 2026-04-12\nconfidence: 0.9\n---\n\n"
            "## Truth\n\nThis article is fine.\n"
        )
        article_b = knowledge / "concepts" / "contradicted.md"
        article_b.write_text(
            "---\ntitle: Contradicted\nupdated: 2026-04-12\nconfidence: 0.9\n---\n\n"
            "## Truth\n\nThis one is in quarantine.\n"
        )

        contradictions = knowledge / "contradictions.json"
        contradictions.write_text(
            json.dumps({"quarantined": ["concepts/contradicted"], "updated": "2026-04-12T00:00:00+00:00"})
        )

        import sys
        from scripts import compile_truth, config, utils
        monkeypatch.setattr(config, "KNOWLEDGE_DIR", knowledge)
        monkeypatch.setattr(config, "CONCEPTS_DIR", knowledge / "concepts")
        monkeypatch.setattr(config, "CONNECTIONS_DIR", knowledge / "connections")
        monkeypatch.setattr(compile_truth, "KNOWLEDGE_DIR", knowledge)
        monkeypatch.setattr(compile_truth, "CONCEPTS_DIR", knowledge / "concepts")
        monkeypatch.setattr(compile_truth, "CONNECTIONS_DIR", knowledge / "connections")
        monkeypatch.setattr(compile_truth, "COMPILED_TRUTH_FILE", knowledge / "compiled-truth.md")
        monkeypatch.setattr(utils, "CONTRADICTIONS_FILE", contradictions)

        # pyproject.toml has pythonpath = [".", "scripts"], so scripts/ is
        # importable both as a package (scripts.utils) AND as top-level (utils).
        # compile_truth.py uses bare-form imports (`from utils import ...`),
        # so the module object it reads from is sys.modules["utils"], NOT
        # scripts.utils. The monkeypatch above only patches scripts.utils;
        # we MUST also patch the bare-form copy or the test reads unpatched
        # CONTRADICTIONS_FILE and fails. This fallback is load-bearing.
        if "utils" in sys.modules and sys.modules["utils"] is not utils:
            monkeypatch.setattr(sys.modules["utils"], "CONTRADICTIONS_FILE", contradictions)
        if "config" in sys.modules and sys.modules["config"] is not config:
            monkeypatch.setattr(sys.modules["config"], "KNOWLEDGE_DIR", knowledge)
            monkeypatch.setattr(sys.modules["config"], "CONCEPTS_DIR", knowledge / "concepts")
            monkeypatch.setattr(sys.modules["config"], "CONNECTIONS_DIR", knowledge / "connections")
        if "compile_truth" in sys.modules and sys.modules["compile_truth"] is not compile_truth:
            bare_ct = sys.modules["compile_truth"]
            monkeypatch.setattr(bare_ct, "KNOWLEDGE_DIR", knowledge)
            monkeypatch.setattr(bare_ct, "CONCEPTS_DIR", knowledge / "concepts")
            monkeypatch.setattr(bare_ct, "CONNECTIONS_DIR", knowledge / "connections")
            monkeypatch.setattr(bare_ct, "COMPILED_TRUTH_FILE", knowledge / "compiled-truth.md")

        included, total, _ = compile_truth.compile_truth(budget=100_000)
        output = (knowledge / "compiled-truth.md").read_text()

        assert "## concepts/good" in output
        # The quarantined slug may appear in the QUARANTINED banner, but must
        # NOT appear as a section heading (## concepts/contradicted).
        assert "## concepts/contradicted" not in output
        assert "This one is in quarantine." not in output
        assert "QUARANTINED" in output  # banner explaining the exclusion


class TestLintResolve:
    """lint.py --resolve should clear the quarantine file."""

    def test_resolve_clears_existing_quarantine(self, tmp_path, monkeypatch):
        import json as _json
        from scripts import utils
        path = tmp_path / "contradictions.json"
        path.write_text(
            _json.dumps({"quarantined": ["concepts/foo", "concepts/bar"], "updated": "2026-04-12T00:00:00+00:00"})
        )
        monkeypatch.setattr(utils, "CONTRADICTIONS_FILE", path)

        # Also patch bare-module copy if separate (same reason as
        # TestCompileTruthQuarantine)
        import sys as _sys
        if "utils" in _sys.modules and _sys.modules["utils"] is not utils:
            monkeypatch.setattr(_sys.modules["utils"], "CONTRADICTIONS_FILE", path)

        assert utils.load_contradictions(path) == {"concepts/foo", "concepts/bar"}

        utils.save_contradictions(set(), path)

        assert utils.load_contradictions(path) == set()
        data = _json.loads(path.read_text())
        assert data["quarantined"] == []


class TestZoneExtraction:
    def test_article_with_both_zones(self):
        from scripts.compile_truth import extract_zones
        content = """## Truth

### Observed

- Fact A from source
- Fact B from source

### Synthesized

- Inferred pattern X
"""
        zones = extract_zones(content)
        assert "Fact A" in zones.observed
        assert "Fact B" in zones.observed
        assert "Inferred pattern X" in zones.synthesized

    def test_article_without_subsections_is_all_observed(self):
        from scripts.compile_truth import extract_zones
        content = """## Truth

This is legacy truth with no subsections.

### Key Points

- Key point 1
- Key point 2
"""
        zones = extract_zones(content)
        assert "legacy truth" in zones.observed
        assert "Key point 1" in zones.observed
        assert zones.synthesized == ""

    def test_article_with_only_synthesized(self):
        from scripts.compile_truth import extract_zones
        content = """## Truth

### Synthesized

- Pure inference
"""
        zones = extract_zones(content)
        assert zones.observed == ""
        assert "Pure inference" in zones.synthesized

    def test_zones_with_no_truth_section_returns_empty(self):
        from scripts.compile_truth import extract_zones
        content = """## Something Else

Not a Truth section at all.
"""
        zones = extract_zones(content)
        assert zones.observed == ""
        assert zones.synthesized == ""


def test_compile_truth_with_clusters_includes_communities_section(tmp_path, monkeypatch):
    """When --with-clusters is set, compiled-truth.md gets a Concept Clusters section.

    We monkey-patch the cluster source to a stable fixture so the test isn't
    coupled to live KB state.
    """
    from scripts import compile_truth

    monkeypatch.setattr(
        compile_truth,
        "_load_clusters_for_truth",
        lambda limit=5: [
            {"label": "Test Cluster", "size": 4, "hub_node": "article:concepts/x", "sample": ["X", "Y", "Z"]},
        ],
    )
    section = compile_truth._render_clusters_section(limit=5)
    assert section is not None
    assert "Concept Clusters" in section
    assert "Test Cluster" in section
    assert "article:concepts/x" in section


class TestExcerptTruth:
    """Entries are capped to an excerpt so the budget buys breadth, not depth."""

    def test_short_truth_is_returned_untouched(self):
        from scripts.compile_truth import excerpt_truth
        truth = "Short and sweet."
        assert excerpt_truth(truth, "concepts/foo", cap=1_200) == truth

    def test_cap_zero_disables_excerpting(self):
        from scripts.compile_truth import excerpt_truth
        truth = "x" * 5_000
        assert excerpt_truth(truth, "concepts/foo", cap=0) == truth

    def test_long_truth_is_capped_and_gets_a_full_text_pointer(self):
        from scripts.compile_truth import excerpt_truth
        truth = ("Paragraph one is here.\n\n" * 200)
        out = excerpt_truth(truth, "concepts/foo", cap=1_200)
        assert len(out) < len(truth)
        # The pointer is what makes the lossy cut safe — without it the model
        # cannot tell the entry is partial or how to recover the rest.
        assert 'get_article("concepts/foo")' in out
        assert "excerpt" in out

    def test_cut_lands_on_a_paragraph_boundary(self):
        from scripts.compile_truth import excerpt_truth
        truth = ("Sentence a. " * 50) + "\n\n" + ("Sentence b. " * 200)
        out = excerpt_truth(truth, "concepts/foo", cap=1_200)
        body = out.split("\n\n_…")[0]
        # Must not end mid-word
        assert not body.endswith("Sen")
        assert body.rstrip().endswith(".")

    def test_unbroken_paragraph_still_keeps_at_least_half_the_cap(self):
        from scripts.compile_truth import excerpt_truth
        truth = "x" * 5_000  # no separators at all
        out = excerpt_truth(truth, "concepts/foo", cap=1_200)
        body = out.split("\n\n_…")[0]
        assert len(body) >= 600, "a separator-free body must not collapse to nothing"
        assert len(body) <= 1_200


class TestCompiledTruthBreadth:
    """A budget of verbose articles must not crowd the KB down to a handful."""

    def test_excerpting_admits_far_more_articles_than_full_bodies(self, tmp_path, monkeypatch):
        import sys
        from scripts import compile_truth, config, utils

        knowledge = tmp_path / "knowledge"
        (knowledge / "concepts").mkdir(parents=True)
        (knowledge / "connections").mkdir()

        # 20 articles, each far larger than the per-entry cap.
        for i in range(20):
            (knowledge / "concepts" / f"a{i:02d}.md").write_text(
                f"---\ntitle: A{i}\nupdated: 2026-04-12\nconfidence: 0.9\n---\n\n"
                "## Truth\n\n" + (f"Body of article {i}. " * 300) + "\n",
                encoding="utf-8",
            )

        contradictions = knowledge / "contradictions.json"
        contradictions.write_text(json.dumps({"quarantined": [], "updated": "2026-04-12T00:00:00+00:00"}))

        for mod in (config, compile_truth):
            monkeypatch.setattr(mod, "KNOWLEDGE_DIR", knowledge, raising=False)
            monkeypatch.setattr(mod, "CONCEPTS_DIR", knowledge / "concepts", raising=False)
            monkeypatch.setattr(mod, "CONNECTIONS_DIR", knowledge / "connections", raising=False)
        monkeypatch.setattr(compile_truth, "COMPILED_TRUTH_FILE", knowledge / "compiled-truth.md")
        monkeypatch.setattr(utils, "CONTRADICTIONS_FILE", contradictions)
        # Bare-form module copies (see TestCompileTruthQuarantine for why).
        for name, patched in (("utils", utils), ("config", config), ("compile_truth", compile_truth)):
            bare = sys.modules.get(name)
            if bare is None or bare is patched:
                continue
            if name == "utils":
                monkeypatch.setattr(bare, "CONTRADICTIONS_FILE", contradictions)
                continue
            monkeypatch.setattr(bare, "KNOWLEDGE_DIR", knowledge, raising=False)
            monkeypatch.setattr(bare, "CONCEPTS_DIR", knowledge / "concepts", raising=False)
            monkeypatch.setattr(bare, "CONNECTIONS_DIR", knowledge / "connections", raising=False)
            if name == "compile_truth":
                monkeypatch.setattr(bare, "COMPILED_TRUTH_FILE", knowledge / "compiled-truth.md")

        budget = 20_000

        full, _, _ = compile_truth.compile_truth(budget=budget, max_article_chars=0)
        excerpted, _, _ = compile_truth.compile_truth(budget=budget, max_article_chars=1_200)

        # This is the regression: at ~6,000 chars/article a 20k budget held 3.
        assert full <= 4
        assert excerpted >= 15
        assert excerpted > full

        output = (knowledge / "compiled-truth.md").read_text(encoding="utf-8")
        assert len(output) <= budget + 2_000  # header + banner overhead
        assert "not** all of it" in output  # honest-scope banner


class TestExcerptMarkdownSafety:
    """compiled-truth.md concatenates entries, so a cut must not corrupt the doc."""

    def test_cut_inside_a_code_fence_closes_the_fence(self):
        from scripts.compile_truth import excerpt_truth
        truth = "Intro paragraph.\n\n```php\n" + ("$x = 1;\n" * 400) + "```\n"
        out = excerpt_truth(truth, "concepts/foo", cap=1_200)
        # An unclosed ``` would swallow every following article into a code block.
        assert out.count("```") % 2 == 0, "excerpt left an unbalanced code fence"

    def test_balanced_fence_is_left_alone(self):
        from scripts.compile_truth import excerpt_truth
        truth = "```php\n$x = 1;\n```\n\n" + ("Filler paragraph. " * 300)
        out = excerpt_truth(truth, "concepts/foo", cap=1_200)
        assert out.count("```") % 2 == 0

    def test_separator_below_half_cap_is_rejected_and_cut_is_hard(self):
        """The `idx >= cap // 2` rule: a separator too early must not be used.

        Otherwise one early newline in a long unbroken run would collapse the
        entry to a fragment.
        """
        from scripts.compile_truth import excerpt_truth
        # Only separator sits at char 100 — well below cap//2 (600).
        truth = ("a" * 100) + "\n\n" + ("b" * 5_000)
        out = excerpt_truth(truth, "concepts/foo", cap=1_200)
        body = out.split("\n\n_…")[0]
        assert len(body) > 600, "cut at an early separator collapsed the entry"
        assert len(body) <= 1_200

    def test_separator_past_half_cap_is_used(self):
        from scripts.compile_truth import excerpt_truth
        truth = ("a" * 800) + "\n\n" + ("b" * 5_000)
        out = excerpt_truth(truth, "concepts/foo", cap=1_200)
        body = out.split("\n\n_…")[0]
        assert body == "a" * 800, "should have cut at the paragraph break"

    def test_text_exactly_at_cap_is_untouched(self):
        from scripts.compile_truth import excerpt_truth
        truth = "a" * 1_200
        assert excerpt_truth(truth, "concepts/foo", cap=1_200) == truth
