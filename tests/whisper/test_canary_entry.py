"""Verify the whisper canary function detects missing grounding."""
from __future__ import annotations

from unittest.mock import MagicMock


from whisper.types import EnhanceResult


def _result(prompt: str) -> EnhanceResult:
    return EnhanceResult(
        transcript="x", enhanced_prompt=prompt, mode="rewrite",
        citations=[], intent="audit", scope_used=["articles"],
        queries_used=["x"], warnings=[], timings_ms={},
    )


def test_canary_passes_when_all_substrings_present(monkeypatch):
    import canary
    from whisper import orchestrator

    monkeypatch.setattr(
        orchestrator,
        "enhance_from_transcript",
        MagicMock(return_value=_result(
            "Run npx @tailwindcss/cli -i assets/styles/app.css -o var/tailwind/app.built.css"
        )),
    )
    result = canary.run_whisper_canary()
    assert result["passed"] is True


def test_canary_fails_when_substring_missing(monkeypatch):
    import canary
    from whisper import orchestrator

    monkeypatch.setattr(
        orchestrator,
        "enhance_from_transcript",
        MagicMock(return_value=_result("Just run tailwind somehow")),
    )
    result = canary.run_whisper_canary()
    assert result["passed"] is False
    assert "missing" in result["detail"]


def test_canary_fails_cleanly_when_pipeline_raises(monkeypatch):
    import canary
    from whisper import orchestrator

    monkeypatch.setattr(
        orchestrator,
        "enhance_from_transcript",
        MagicMock(side_effect=RuntimeError("boom")),
    )
    result = canary.run_whisper_canary()
    assert result["passed"] is False
    assert "exception" in result["detail"]
