"""Pytest suite for source anchor extraction and verification."""
from __future__ import annotations

import pytest


class TestExtractSourceAnchors:
    def test_single_anchor_in_bullet(self):
        from scripts.utils import extract_source_anchors
        text = "- Fact statement about X [src:daily/2026-04-10.md]"
        anchors = extract_source_anchors(text)
        assert anchors == ["daily/2026-04-10.md"]

    def test_multiple_anchors_in_text(self):
        from scripts.utils import extract_source_anchors
        text = """
        - First fact [src:daily/2026-04-10.md]
        - Second fact [src:sources/design-specs/foo.md]
        - Third fact no anchor
        - Fourth [src:daily/2026-04-11.md]
        """
        anchors = extract_source_anchors(text)
        assert sorted(anchors) == sorted([
            "daily/2026-04-10.md",
            "sources/design-specs/foo.md",
            "daily/2026-04-11.md",
        ])

    def test_no_anchors_returns_empty(self):
        from scripts.utils import extract_source_anchors
        assert extract_source_anchors("- plain bullet") == []

    def test_anchor_with_path_separators(self):
        from scripts.utils import extract_source_anchors
        text = "- Fact [src:sources/implementation-plans/2026-04-10-foo.md]"
        assert extract_source_anchors(text) == ["sources/implementation-plans/2026-04-10-foo.md"]

    def test_wikilink_not_treated_as_anchor(self):
        from scripts.utils import extract_source_anchors
        # wikilinks use [[link]] not [src:...]; make sure extractor isn't too greedy
        text = "- Fact that links to [[concepts/foo]] with no anchor"
        assert extract_source_anchors(text) == []


class TestVerifySourceAnchor:
    def test_existing_daily_log_verifies(self, tmp_path, monkeypatch):
        from scripts import utils, config
        import sys
        monkeypatch.setattr(config, "KNOWLEDGE_DIR", tmp_path)
        monkeypatch.setattr(utils, "KNOWLEDGE_DIR", tmp_path)
        # Bare-module fallback (see TestCompileTruthQuarantine for rationale):
        # utils is importable both as `scripts.utils` AND as bare `utils` when
        # pyproject.toml has pythonpath = [".", "scripts"]. compile_truth etc
        # use bare imports, so the copy we care about may not be the one we
        # just patched.
        if "utils" in sys.modules and sys.modules["utils"] is not utils:
            monkeypatch.setattr(sys.modules["utils"], "KNOWLEDGE_DIR", tmp_path)
        if "config" in sys.modules and sys.modules["config"] is not config:
            monkeypatch.setattr(sys.modules["config"], "KNOWLEDGE_DIR", tmp_path)

        daily = tmp_path / "daily"
        daily.mkdir()
        (daily / "2026-04-10.md").write_text("content")

        assert utils.verify_source_anchor("daily/2026-04-10.md") is True

    def test_missing_file_fails(self, tmp_path, monkeypatch):
        from scripts import utils, config
        import sys
        monkeypatch.setattr(config, "KNOWLEDGE_DIR", tmp_path)
        monkeypatch.setattr(utils, "KNOWLEDGE_DIR", tmp_path)
        if "utils" in sys.modules and sys.modules["utils"] is not utils:
            monkeypatch.setattr(sys.modules["utils"], "KNOWLEDGE_DIR", tmp_path)
        if "config" in sys.modules and sys.modules["config"] is not config:
            monkeypatch.setattr(sys.modules["config"], "KNOWLEDGE_DIR", tmp_path)
        assert utils.verify_source_anchor("daily/nonexistent.md") is False
