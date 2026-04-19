"""Unit tests for whisper.expand_query.

The Anthropic SDK call is mocked — we test our wrapper's parsing,
validation, and fallback behavior, not the LLM itself.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _mock_anthropic_response(text: str):
    """Shape a Messages API response to match what expand_query reads."""
    return SimpleNamespace(content=[SimpleNamespace(text=text, type="text")])


@pytest.fixture
def mock_client(monkeypatch):
    client = MagicMock()
    import whisper.expand_query as eq
    monkeypatch.setattr(eq, "_get_client", lambda: client)
    return client


def test_expand_parses_clean_json_response(mock_client):
    mock_client.messages.create.return_value = _mock_anthropic_response(
        json.dumps({
            "queries": ["S3 migration", "event_locations Twig function"],
            "intent": "audit",
            "scope": ["articles", "code", "daily"],
        })
    )
    from whisper.expand_query import expand

    result = expand("audit the S3 event location migration")

    assert result.queries == ["S3 migration", "event_locations Twig function"]
    assert result.intent == "audit"
    assert result.scope == ["articles", "code", "daily"]


def test_expand_strips_markdown_code_fences(mock_client):
    mock_client.messages.create.return_value = _mock_anthropic_response(
        "```json\n"
        + json.dumps({"queries": ["q1"], "intent": "explain", "scope": ["articles"]})
        + "\n```"
    )
    from whisper.expand_query import expand

    result = expand("explain the tailwind rebuild")

    assert result.queries == ["q1"]


def test_expand_invalid_intent_falls_back_to_generic(mock_client):
    mock_client.messages.create.return_value = _mock_anthropic_response(
        json.dumps({
            "queries": ["q1"],
            "intent": "invent-intent",
            "scope": ["articles"],
        })
    )
    from whisper.expand_query import expand

    result = expand("something")

    assert result.intent == "generic"


def test_expand_invalid_scope_item_is_dropped(mock_client):
    mock_client.messages.create.return_value = _mock_anthropic_response(
        json.dumps({
            "queries": ["q1"],
            "intent": "audit",
            "scope": ["articles", "bogus-channel", "code"],
        })
    )
    from whisper.expand_query import expand

    result = expand("audit X")

    assert result.scope == ["articles", "code"]


def test_expand_empty_queries_raises(mock_client):
    mock_client.messages.create.return_value = _mock_anthropic_response(
        json.dumps({"queries": [], "intent": "audit", "scope": ["articles"]})
    )
    from whisper.expand_query import expand, ExpansionError

    with pytest.raises(ExpansionError):
        expand("transcript")


def test_expand_malformed_json_raises(mock_client):
    mock_client.messages.create.return_value = _mock_anthropic_response(
        "not valid json at all { queries: hi"
    )
    from whisper.expand_query import expand, ExpansionError

    with pytest.raises(ExpansionError):
        expand("transcript")


def test_expand_scope_defaults_to_articles_if_all_items_invalid(mock_client):
    mock_client.messages.create.return_value = _mock_anthropic_response(
        json.dumps({"queries": ["q"], "intent": "audit", "scope": ["nonsense"]})
    )
    from whisper.expand_query import expand

    result = expand("transcript")

    assert result.scope == ["articles"]
