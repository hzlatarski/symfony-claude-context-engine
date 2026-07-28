"""Tests for clean enhancement at the Claude CLI subprocess boundary."""

from __future__ import annotations

import subprocess

import pytest


def _fake_cli(monkeypatch, *, returncode=0, stdout="cleaned", stderr=""):
    import whisper.enhance as enhance

    calls: list[dict] = []

    def run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            command, returncode, stdout=stdout, stderr=stderr,
        )

    monkeypatch.setattr(enhance.subprocess, "run", run)
    return enhance, calls


def test_clean_returns_cli_output_stripped(monkeypatch):
    enhance, _ = _fake_cli(
        monkeypatch, stdout="  Hello world, how are you?\n",
    )

    assert enhance.enhance_clean("uh hello world um") == "Hello world, how are you?"


def test_clean_passes_model_prompt_and_transcript(monkeypatch):
    enhance, calls = _fake_cli(monkeypatch, stdout="hello")
    import config
    from whisper.prompts import CLEAN_SYSTEM_PROMPT

    enhance.enhance_clean("raw voice transcript")

    call = calls[0]
    assert call["input"] == "raw voice transcript"
    assert call["command"][call["command"].index("--model") + 1] == config.MODEL_CLEAN
    assert (
        call["command"][call["command"].index("--system-prompt") + 1]
        == CLEAN_SYSTEM_PROMPT
    )


def test_clean_nonzero_exit_with_stdout_raises(monkeypatch):
    enhance, _ = _fake_cli(
        monkeypatch, returncode=1, stdout="Prompt is too long",
    )

    with pytest.raises(enhance.EnhanceError, match="exited 1"):
        enhance.enhance_clean("hello")


def test_clean_empty_success_response_raises(monkeypatch):
    enhance, _ = _fake_cli(monkeypatch, returncode=0, stdout="  ")

    with pytest.raises(enhance.EnhanceError, match="empty response"):
        enhance.enhance_clean("hello")


def test_empty_transcript_skips_cli(monkeypatch):
    enhance, calls = _fake_cli(monkeypatch, stdout="unused")

    assert enhance.enhance_clean("  ") == "  "
    assert calls == []
