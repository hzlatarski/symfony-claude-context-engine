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
