"""Unit tests for the whisper orchestrator.

We mock the four step functions (transcribe, expand, retrieve, enhance_*)
and verify the orchestrator wires them correctly, including the
degraded-mode fallbacks.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from whisper.types import Hit


def _hit(cid: str):
    return Hit(
        id=cid, source="article", category="governance",
        path="concepts/foo", title="Foo", snippet="snip",
        full_body="body", score=1.0, symbols=[], metadata={},
    )


@pytest.fixture
def mocked_steps(monkeypatch):
    import whisper.orchestrator as o

    m_transcribe = MagicMock(return_value="audit the S3 migration")
    m_expand = MagicMock()
    m_retrieve = MagicMock(return_value=[_hit("c1")])
    m_rewrite = MagicMock()
    m_verbatim = MagicMock(return_value="VERBATIM OUT")
    m_clean = MagicMock(return_value="CLEAN OUT")

    from whisper.expand_query import Expansion
    m_expand.return_value = Expansion(
        queries=["q1", "q2"], intent="audit", scope=["articles"],
    )

    from whisper.enhance import RewriteOutput
    m_rewrite.return_value = RewriteOutput(prompt="REWRITTEN OUT", warnings=[])

    monkeypatch.setattr(o, "transcribe", m_transcribe)
    monkeypatch.setattr(o, "expand", m_expand)
    monkeypatch.setattr(o, "retrieve", m_retrieve)
    monkeypatch.setattr(o, "enhance_rewrite", m_rewrite)
    monkeypatch.setattr(o, "enhance_verbatim", m_verbatim)
    monkeypatch.setattr(o, "enhance_clean", m_clean)

    return {
        "transcribe": m_transcribe, "expand": m_expand,
        "retrieve": m_retrieve, "rewrite": m_rewrite,
        "verbatim": m_verbatim, "clean": m_clean,
    }


def test_enhance_full_rewrite_flow_uses_rewrite_output(mocked_steps):
    from whisper.orchestrator import enhance_from_audio

    result = enhance_from_audio(audio=b"bytes", mode="rewrite", language="en")

    assert result.transcript == "audit the S3 migration"
    assert result.enhanced_prompt == "REWRITTEN OUT"
    assert result.mode == "rewrite"
    assert result.intent == "audit"
    assert result.scope_used == ["articles"]
    assert result.queries_used == ["q1", "q2"]
    assert len(result.citations) == 1
    assert "total" in result.timings_ms


def test_enhance_verbatim_mode_skips_rewrite(mocked_steps):
    from whisper.orchestrator import enhance_from_audio

    result = enhance_from_audio(audio=b"x", mode="verbatim", language="en")

    assert result.enhanced_prompt == "VERBATIM OUT"
    assert result.mode == "verbatim"
    assert mocked_steps["rewrite"].called is False
    assert mocked_steps["verbatim"].called is True


def test_enhance_clean_mode_skips_retrieval_and_expansion(mocked_steps):
    from whisper.orchestrator import enhance_from_audio

    result = enhance_from_audio(audio=b"x", mode="clean", language="en")

    assert result.enhanced_prompt == "CLEAN OUT"
    assert result.mode == "clean"
    assert mocked_steps["expand"].called is False
    assert mocked_steps["retrieve"].called is False


def test_enhance_rewrite_auto_downgrades_to_verbatim_when_no_hits(mocked_steps):
    mocked_steps["retrieve"].return_value = []
    from whisper.orchestrator import enhance_from_audio

    result = enhance_from_audio(audio=b"x", mode="rewrite", language="en")

    assert result.mode == "verbatim"
    assert any("No project context" in w for w in result.warnings)
    assert mocked_steps["rewrite"].called is False


def test_enhance_rewrite_downgrades_when_sonnet_fails(mocked_steps):
    from whisper.enhance import EnhanceError
    mocked_steps["rewrite"].side_effect = EnhanceError("boom")
    from whisper.orchestrator import enhance_from_audio

    result = enhance_from_audio(audio=b"x", mode="rewrite", language="en")

    assert result.mode == "verbatim"
    assert any("Rewrite failed" in w for w in result.warnings)


def test_enhance_empty_transcript_raises(mocked_steps):
    mocked_steps["transcribe"].return_value = ""
    from whisper.orchestrator import enhance_from_audio, NoSpeechError

    with pytest.raises(NoSpeechError):
        enhance_from_audio(audio=b"x", mode="rewrite", language="en")


def test_enhance_from_transcript_skips_transcription(mocked_steps):
    from whisper.orchestrator import enhance_from_transcript

    result = enhance_from_transcript(
        transcript="cached text",
        mode="rewrite",
        scope_override=["articles", "code"],
    )

    assert result.transcript == "cached text"
    assert mocked_steps["transcribe"].called is False
    # scope_override replaces Haiku's suggestion
    assert result.scope_used == ["articles", "code"]
    # retrieve still called with the override
    _args, kwargs = mocked_steps["retrieve"].call_args
    assert kwargs["scope"] == ["articles", "code"]
