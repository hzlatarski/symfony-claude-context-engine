"""Tests for the multi-agent conversation-history importer.

Covers the pure parse functions of the codex and claude adapters plus the
shared rendering / filtering / discovery helpers in import_agent_history.
All inputs are synthetic jsonl strings or tmp_path stores — no reads of the
real ~/.codex or ~/.claude directories, no network, no Chroma.
"""
import json
from pathlib import Path

from scripts import import_agent_history as iah
from scripts.agent_adapters import codex, claude
from scripts import agent_adapters


# ── Codex adapter ──────────────────────────────────────────────────────

def _codex_lines():
    return [
        json.dumps({
            "timestamp": "2026-06-15T10:00:00Z",
            "type": "session_meta",
            "payload": {
                "id": "abc123def456",
                "timestamp": "2026-06-15T10:00:00Z",
                "cwd": "c:\\wamp64\\www\\eintollesfest",
            },
        }),
        json.dumps({
            "timestamp": "2026-06-15T10:00:05Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello there"}],
            },
        }),
        json.dumps({
            "timestamp": "2026-06-15T10:00:09Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi, "},
                            {"type": "output_text", "text": "how can I help?"}],
            },
        }),
        # developer message — must be skipped
        json.dumps({
            "timestamp": "2026-06-15T10:00:10Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "system instructions"}],
            },
        }),
        # tool/function call — non-message response_item, must be skipped
        json.dumps({
            "timestamp": "2026-06-15T10:00:11Z",
            "type": "response_item",
            "payload": {"type": "function_call", "name": "shell", "arguments": "{}"},
        }),
    ]


def test_codex_parse_extracts_meta():
    session = codex.parse(_codex_lines())
    assert session.agent == "codex"
    assert session.session_id == "abc123def456"
    assert session.cwd == "c:\\wamp64\\www\\eintollesfest"
    assert session.started_at == "2026-06-15T10:00:00Z"


def test_codex_parse_keeps_user_and_assistant_drops_developer_and_tools():
    session = codex.parse(_codex_lines())
    roles = [t["role"] for t in session.turns]
    assert roles == ["user", "assistant"]
    assert session.turns[0]["text"] == "Hello there"
    # concatenated content items
    assert session.turns[1]["text"] == "Hi, how can I help?"


# ── Claude adapter ─────────────────────────────────────────────────────

def _claude_lines():
    return [
        json.dumps({"type": "summary", "summary": "a summary line"}),
        json.dumps({"type": "queue-operation", "op": "noop"}),
        # string content
        json.dumps({
            "type": "user",
            "sessionId": "sess-789",
            "cwd": "c:/wamp64/www/AiTutor",
            "timestamp": "2026-06-16T08:00:00Z",
            "message": {"role": "user", "content": "Plain string question"},
        }),
        # list-of-blocks content (mix text + tool_use)
        json.dumps({
            "type": "assistant",
            "sessionId": "sess-789",
            "cwd": "c:/wamp64/www/AiTutor",
            "timestamp": "2026-06-16T08:00:03Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "First part. "},
                    {"type": "tool_use", "name": "Read", "input": {}},
                    {"type": "text", "text": "Second part."},
                ],
            },
        }),
        # tool-result only — must be skipped (no text blocks)
        json.dumps({
            "type": "user",
            "sessionId": "sess-789",
            "cwd": "c:/wamp64/www/AiTutor",
            "timestamp": "2026-06-16T08:00:04Z",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "content": "output"}],
            },
        }),
        # attachment / hook output — must be skipped
        json.dumps({
            "type": "user",
            "sessionId": "sess-789",
            "cwd": "c:/wamp64/www/AiTutor",
            "timestamp": "2026-06-16T08:00:05Z",
            "attachment": {"type": "hook"},
            "message": {"role": "user", "content": "hook noise"},
        }),
    ]


def test_claude_parse_meta_and_string_content():
    session = claude.parse(_claude_lines())
    assert session.agent == "claude"
    assert session.session_id == "sess-789"
    assert session.cwd == "c:/wamp64/www/AiTutor"
    assert session.turns[0]["role"] == "user"
    assert session.turns[0]["text"] == "Plain string question"


def test_claude_parse_list_blocks_and_skips_noise():
    session = claude.parse(_claude_lines())
    roles = [t["role"] for t in session.turns]
    # only the string-user + list-assistant survive
    assert roles == ["user", "assistant"]
    assert session.turns[1]["text"] == "First part. Second part."


# ── render_session_markdown ────────────────────────────────────────────

def test_render_session_markdown_frontmatter_and_labels():
    session = iah.AgentSession(
        agent="codex",
        session_id="abc123def456",
        cwd="c:/wamp64/www/eintollesfest",
        started_at="2026-06-15T10:00:00Z",
        turns=[
            {"ts": "2026-06-15T10:00:05Z", "role": "user", "text": "Hello there"},
            {"ts": "2026-06-15T10:00:09Z", "role": "assistant", "text": "Hi!"},
        ],
    )
    md = iah.render_session_markdown(session)
    assert md.startswith("---")
    assert "source_agent: codex" in md
    assert "session_id: abc123def456" in md
    assert "type: event" in md
    assert "date: 2026-06-15" in md
    assert "## Transcript" in md
    assert "**User:**" in md
    assert "**Assistant:**" in md
    assert "Hello there" in md
    assert "Hi!" in md


# ── session_matches ────────────────────────────────────────────────────

def _session(cwd, started_at="2026-06-15T10:00:00Z"):
    return iah.AgentSession(
        agent="codex", session_id="x", cwd=cwd, started_at=started_at, turns=[],
    )


def test_session_matches_project_normalizes_backslashes():
    s = _session("c:\\wamp64\\www\\AiTutor")
    assert iah.session_matches(s, project="c:/wamp64/www/AiTutor", since=None)


def test_session_matches_project_trailing_slash_and_case():
    s = _session("C:/WAMP64/www/AiTutor/")
    assert iah.session_matches(s, project="c:/wamp64/www/aitutor", since=None)


def test_session_matches_project_mismatch():
    s = _session("c:/wamp64/www/eintollesfest")
    assert not iah.session_matches(s, project="c:/wamp64/www/AiTutor", since=None)


def test_session_matches_project_none_matches_anything():
    s = _session("c:/anywhere")
    assert iah.session_matches(s, project=None, since=None)


def test_session_matches_since_cutoff():
    s = _session("c:/x", started_at="2026-06-10T00:00:00Z")
    assert iah.session_matches(s, project=None, since="2026-06-10")
    assert iah.session_matches(s, project=None, since="2026-06-09")
    assert not iah.session_matches(s, project=None, since="2026-06-11")


# ── discover_sessions ──────────────────────────────────────────────────

def test_discover_sessions_against_tmp_store(tmp_path):
    nested = tmp_path / "2026" / "06" / "15"
    nested.mkdir(parents=True)
    f1 = nested / "rollout-1.jsonl"
    f2 = nested / "rollout-2.jsonl"
    f1.write_text("{}", encoding="utf-8")
    f2.write_text("{}", encoding="utf-8")
    (nested / "ignore.txt").write_text("nope", encoding="utf-8")

    found = iah.discover_sessions(tmp_path, "**/rollout-*.jsonl")
    names = sorted(p.name for p in found)
    assert names == ["rollout-1.jsonl", "rollout-2.jsonl"]


# ── registry ───────────────────────────────────────────────────────────

def test_registry_has_builtin_adapters():
    assert "codex" in agent_adapters.available_agents()
    assert "claude" in agent_adapters.available_agents()
    adapter = agent_adapters.get_adapter("codex")
    assert callable(adapter.parse)
    assert callable(adapter.default_store)
