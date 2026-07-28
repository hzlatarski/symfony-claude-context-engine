"""Tests for grounded rewriting and anchor verification."""

from __future__ import annotations

import subprocess

import pytest

from whisper.types import Hit


def _hit(cid: str, source: str, path: str, category=None) -> Hit:
    return Hit(
        id=cid,
        source=source,
        category=category,
        path=path,
        title=path,
        snippet="snippet",
        full_body=f"full body for {path}",
        score=1.0,
        symbols=[],
        metadata={},
    )


def _fake_cli(monkeypatch, *, returncode=0, stdout="REWRITTEN", stderr=""):
    import whisper.enhance as enhance

    calls: list[dict] = []

    def run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            command, returncode, stdout=stdout, stderr=stderr,
        )

    monkeypatch.setattr(enhance.subprocess, "run", run)
    return enhance, calls


def test_rewrite_passes_transcript_context_model_and_prompt(monkeypatch):
    enhance, calls = _fake_cli(monkeypatch)
    import config
    from whisper.prompts import REWRITE_SYSTEM_PROMPT

    hits = [
        _hit("c1", "article", "concepts/s3-migration", "captured-memory"),
        _hit("c2", "code", "src/Service/Foo.php:1-20"),
    ]
    enhance.enhance_rewrite("audit the S3 migration", hits)

    call = calls[0]
    assert "audit the S3 migration" in call["input"]
    assert "concepts/s3-migration" in call["input"]
    assert "category=captured-memory" in call["input"]
    assert "src/Service/Foo.php:1-20" in call["input"]
    assert call["command"][call["command"].index("--model") + 1] == config.MODEL_REWRITE
    assert (
        call["command"][call["command"].index("--system-prompt") + 1]
        == REWRITE_SYSTEM_PROMPT
    )


def test_rewrite_returns_complete_result(monkeypatch):
    enhance, _ = _fake_cli(
        monkeypatch, stdout="Use [src:concepts/real]",
    )
    hits = [_hit("c1", "article", "concepts/real")]

    result = enhance.enhance_rewrite(
        "audit",
        hits,
        intent="audit",
        scope_used=["articles"],
        queries_used=["q1"],
    )

    assert result.transcript == "audit"
    assert result.enhanced_prompt == "Use [src:concepts/real]"
    assert result.mode == "rewrite"
    assert result.citations == hits
    assert result.intent == "audit"
    assert result.scope_used == ["articles"]
    assert result.queries_used == ["q1"]
    assert set(result.timings_ms) == {"llm_ms", "enhance_ms"}


def test_rewrite_requires_at_least_one_hit(monkeypatch):
    enhance, calls = _fake_cli(monkeypatch)

    with pytest.raises(enhance.EnhanceError, match="at least one"):
        enhance.enhance_rewrite("transcript", [])
    assert calls == []


def test_rewrite_nonzero_exit_with_stdout_raises(monkeypatch):
    enhance, _ = _fake_cli(
        monkeypatch, returncode=2, stdout="authentication failed",
    )

    with pytest.raises(enhance.EnhanceError, match="exited 2"):
        enhance.enhance_rewrite(
            "transcript", [_hit("c1", "article", "concepts/real")],
        )


def test_verify_anchors_keeps_only_retrieved_paths():
    from whisper.enhance import verify_anchors

    hits = [_hit("c1", "article", "concepts/real")]
    cleaned, warnings = verify_anchors(
        "Keep [src:concepts/real], reject [src:src/Fake.php]", hits,
    )

    assert "[src:concepts/real]" in cleaned
    assert "[src:src/Fake.php]" not in cleaned
    assert "src/Fake.php" in cleaned
    assert warnings == ["Removed unverifiable anchor: src/Fake.php"]


def test_verify_anchors_accepts_code_path_with_or_without_line_range():
    from whisper.enhance import verify_anchors

    hits = [_hit("c1", "code", "src/Service/Foo.php:1-20")]
    text = "See [src:src/Service/Foo.php] and [src:src/Service/Foo.php:1-20]"

    assert verify_anchors(text, hits) == (text, [])


def test_verify_anchors_normalizes_whitespace():
    from whisper.enhance import verify_anchors

    cleaned, warnings = verify_anchors(
        "[src:  concepts/foo  ]",
        [_hit("c1", "article", "concepts/foo")],
    )

    assert cleaned == "[src:concepts/foo]"
    assert warnings == []
