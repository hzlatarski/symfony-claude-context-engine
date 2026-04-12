"""Pytest suite for canary.py question loader and substring matcher."""
from __future__ import annotations

import textwrap

import pytest


class TestCanaryParsing:
    def test_load_canaries_from_yaml(self, tmp_path):
        from scripts.canary import load_canaries
        yaml_path = tmp_path / "canaries.yaml"
        yaml_path.write_text(textwrap.dedent("""\
            version: 1
            canaries:
              - id: compile-model
                question: "What model does compile.py use by default?"
                must_contain:
                  - "sonnet"
                  - "compile"
              - id: flush-frequency
                question: "How often does flush run?"
                must_contain:
                  - "session end"
        """))
        canaries = load_canaries(yaml_path)
        assert len(canaries) == 2
        assert canaries[0].id == "compile-model"
        assert "sonnet" in canaries[0].must_contain

    def test_empty_file_returns_empty_list(self, tmp_path):
        from scripts.canary import load_canaries
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("version: 1\ncanaries: []\n")
        assert load_canaries(yaml_path) == []

    def test_missing_file_returns_empty_list(self, tmp_path):
        from scripts.canary import load_canaries
        assert load_canaries(tmp_path / "nope.yaml") == []


class TestSubstringCheck:
    def test_all_substrings_present_passes(self):
        from scripts.canary import check_answer
        answer = "compile.py uses claude-sonnet-4-6 by default for compilation."
        result = check_answer(answer, ["sonnet", "compile"])
        assert result.passed is True
        assert result.missing == []

    def test_missing_substring_fails(self):
        from scripts.canary import check_answer
        answer = "I'm not sure what model it uses."
        result = check_answer(answer, ["sonnet", "compile"])
        assert result.passed is False
        assert "sonnet" in result.missing

    def test_case_insensitive_match(self):
        from scripts.canary import check_answer
        answer = "Uses SONNET for compilation."
        result = check_answer(answer, ["sonnet"])
        assert result.passed is True

    def test_must_not_contain_blocks_negation(self):
        from scripts.canary import check_answer
        answer = "compile.py does NOT use sonnet; it uses haiku."
        result = check_answer(
            answer,
            must_contain=["sonnet"],
            must_not_contain=["not use sonnet", "does not use sonnet", "doesn't use sonnet"],
        )
        assert result.passed is False
        assert any("not use sonnet" in f for f in result.forbidden_found)

    def test_must_not_contain_empty_by_default(self):
        from scripts.canary import check_answer
        answer = "compile.py uses sonnet."
        result = check_answer(answer, must_contain=["sonnet"])
        assert result.passed is True
        assert result.forbidden_found == []

    def test_must_not_contain_is_case_insensitive(self):
        from scripts.canary import check_answer
        answer = "Compile.py does NOT USE sonnet."
        result = check_answer(
            answer,
            must_contain=["sonnet"],
            must_not_contain=["not use sonnet"],
        )
        assert result.passed is False


class TestMalformedCanaries:
    def test_empty_must_contain_raises(self, tmp_path):
        from scripts.canary import load_canaries
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text(
            "version: 1\n"
            "canaries:\n"
            "  - id: broken\n"
            "    question: \"What is X?\"\n"
            "    must_contain: []\n"
        )
        with pytest.raises(ValueError, match="broken"):
            load_canaries(yaml_path)

    def test_missing_id_raises(self, tmp_path):
        from scripts.canary import load_canaries
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text(
            "version: 1\n"
            "canaries:\n"
            "  - question: \"What is X?\"\n"
            "    must_contain: [\"y\"]\n"
        )
        with pytest.raises(ValueError, match="missing"):
            load_canaries(yaml_path)
