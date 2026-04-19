"""Unit tests for whisper.transcribe.

The real faster-whisper model is too heavy for a unit-test path (3-5s
cold start, 150MB model download). We mock it at import time and only
verify our wrapper's contract:

  - transcribe(bytes, lang) returns a stripped string
  - empty segments → empty string
  - language='auto' is passed as None to faster-whisper
  - the model is instantiated exactly once (singleton behavior)

A real-model contract test is added later and marked @pytest.mark.slow.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _fake_segment(text):
    return SimpleNamespace(text=text)


def _install_fake_model_cls(monkeypatch, transcribe_returns):
    """Patch faster_whisper.WhisperModel with a Mock that returns canned segments."""
    fake_instance = MagicMock()
    fake_instance.transcribe.return_value = (
        iter(transcribe_returns),
        SimpleNamespace(language="en", language_probability=0.99),
    )
    fake_cls = MagicMock(return_value=fake_instance)
    import faster_whisper
    monkeypatch.setattr(faster_whisper, "WhisperModel", fake_cls)
    # Reset the singleton cache between tests
    from whisper import transcribe as t
    t._MODEL = None
    return fake_cls, fake_instance


def test_transcribe_returns_joined_stripped_text(monkeypatch):
    _install_fake_model_cls(
        monkeypatch,
        [_fake_segment("  Hello world."), _fake_segment(" How are you?")],
    )
    from whisper.transcribe import transcribe

    result = transcribe(b"fake-audio-bytes", language="en")

    assert result == "Hello world. How are you?"


def test_transcribe_empty_audio_returns_empty_string(monkeypatch):
    _install_fake_model_cls(monkeypatch, [])
    from whisper.transcribe import transcribe

    assert transcribe(b"", language="en") == ""


def test_transcribe_auto_language_passes_none(monkeypatch):
    _, fake_instance = _install_fake_model_cls(
        monkeypatch, [_fake_segment("hi")]
    )
    from whisper.transcribe import transcribe

    transcribe(b"x", language="auto")

    # Check the language argument passed to faster-whisper
    _args, kwargs = fake_instance.transcribe.call_args
    assert kwargs.get("language") is None


def test_transcribe_singleton_model_instantiated_once(monkeypatch):
    fake_cls, _ = _install_fake_model_cls(
        monkeypatch, [_fake_segment("hi")]
    )
    from whisper.transcribe import transcribe

    transcribe(b"a", language="en")
    transcribe(b"b", language="en")
    transcribe(b"c", language="en")

    assert fake_cls.call_count == 1
