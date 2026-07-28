"""Tests for query expansion at the Claude CLI subprocess boundary."""

from __future__ import annotations

import json
import subprocess

import pytest


def _fake_cli(monkeypatch, payload, *, returncode=0, stderr=""):
    import whisper.expand_query as expand_query

    stdout = payload if isinstance(payload, str) else json.dumps(payload)
    calls: list[dict] = []

    def run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            command, returncode, stdout=stdout, stderr=stderr,
        )

    monkeypatch.setattr(expand_query.subprocess, "run", run)
    return expand_query, calls


def test_expand_parses_valid_json_and_cli_contract(monkeypatch):
    eq, calls = _fake_cli(monkeypatch, {
        "queries": ["S3 migration", "event_locations Twig function"],
        "intent": "audit",
        "scope": ["articles", "code", "daily"],
    })
    import config
    from whisper.prompts import QUERY_EXPANSION_SYSTEM_PROMPT

    result = eq.expand("audit the S3 migration")

    assert result.queries == ["S3 migration", "event_locations Twig function"]
    assert result.intent == "audit"
    assert result.scope == ["articles", "code", "daily"]
    call = calls[0]
    assert call["input"] == "audit the S3 migration"
    assert call["command"][call["command"].index("--model") + 1] == config.MODEL_EXPAND
    assert (
        call["command"][call["command"].index("--system-prompt") + 1]
        == QUERY_EXPANSION_SYSTEM_PROMPT
    )


def test_expand_strips_markdown_fences(monkeypatch):
    eq, _ = _fake_cli(
        monkeypatch,
        '```json\n{"queries":["q1"],"intent":"explain","scope":["articles"]}\n```',
    )

    assert eq.expand("explain").queries == ["q1"]


def test_expand_validates_intent_and_scope(monkeypatch):
    eq, _ = _fake_cli(monkeypatch, {
        "queries": ["q1"],
        "intent": "invented",
        "scope": ["articles", "bogus", "code"],
    })

    result = eq.expand("anything")

    assert result.intent == "generic"
    assert result.scope == ["articles", "code"]


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"queries": []}, "missing 'queries' list"),
        ({"queries": "q1"}, "missing 'queries' list"),
        ({"queries": [1, None, " "]}, "contained no valid strings"),
        ("not json", "non-JSON"),
    ],
)
def test_expand_rejects_unusable_output(monkeypatch, payload, message):
    eq, _ = _fake_cli(monkeypatch, payload)

    with pytest.raises(eq.ExpansionError, match=message):
        eq.expand("transcript")


def test_expand_defaults_invalid_scope_and_caps_queries(monkeypatch):
    long = "a" * 1000
    eq, _ = _fake_cli(monkeypatch, {
        "queries": [f"q{i}-{long}" for i in range(20)],
        "intent": "audit",
        "scope": ["invalid"],
    })

    result = eq.expand("transcript")

    assert result.scope == ["articles"]
    assert len(result.queries) == eq.MAX_QUERIES
    assert all(len(query) <= eq.MAX_QUERY_LENGTH for query in result.queries)


def test_expand_nonzero_exit_with_stdout_raises(monkeypatch):
    eq, _ = _fake_cli(
        monkeypatch,
        {"queries": ["looks-valid"]},
        returncode=1,
    )

    with pytest.raises(eq.ExpansionError, match="exited 1"):
        eq.expand("transcript")
