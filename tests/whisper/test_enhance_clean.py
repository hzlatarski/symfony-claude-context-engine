"""Unit tests for whisper.enhance.enhance_clean (Haiku cleanup mode)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _mock_response(text: str):
    return SimpleNamespace(content=[SimpleNamespace(text=text, type="text")])


@pytest.fixture
def mock_client(monkeypatch):
    client = MagicMock()
    import whisper.enhance as e
    # Reset cached client via lru_cache clear
    e._get_client.cache_clear()
    monkeypatch.setattr(e, "_get_client", lambda: client)
    return client


def test_clean_returns_haiku_output_stripped(mock_client):
    mock_client.messages.create.return_value = _mock_response(
        "  Hello world, how are you?\n"
    )
    from whisper.enhance import enhance_clean

    out = enhance_clean("uh hello world um how how are you")

    assert out == "Hello world, how are you?"


def test_clean_uses_haiku_model_from_config(mock_client):
    mock_client.messages.create.return_value = _mock_response("hello")
    import config
    from whisper.enhance import enhance_clean

    enhance_clean("hello")

    _args, kwargs = mock_client.messages.create.call_args
    assert kwargs["model"] == config.MODEL_CLEAN


def test_clean_passes_transcript_as_user_message(mock_client):
    mock_client.messages.create.return_value = _mock_response("x")
    from whisper.enhance import enhance_clean

    enhance_clean("raw voice transcript")

    _args, kwargs = mock_client.messages.create.call_args
    assert kwargs["messages"] == [{"role": "user", "content": "raw voice transcript"}]


def test_clean_raises_on_empty_response(mock_client):
    mock_client.messages.create.return_value = SimpleNamespace(content=[])
    from whisper.enhance import enhance_clean, EnhanceError

    with pytest.raises(EnhanceError):
        enhance_clean("hello")


def test_clean_uses_clean_system_prompt(mock_client):
    mock_client.messages.create.return_value = _mock_response("ok")
    from whisper.enhance import enhance_clean
    import whisper.prompts as p

    enhance_clean("hello")

    _args, kwargs = mock_client.messages.create.call_args
    assert kwargs["system"] == p.CLEAN_SYSTEM_PROMPT
