"""Claude Code conversation-history adapter.

On-disk format: ``~/.claude/projects/<encoded>/*.jsonl`` where ``<encoded>``
is the project path with separators replaced by ``-`` (e.g.
``c:/wamp64/www/AiTutor`` -> ``c--wamp64-www-AiTutor``).

Keep only lines whose top-level ``type in {user, assistant}`` that carry a
``message`` object ``{"role", "content"}``. ``content`` is either a string OR
a list of blocks; for the list, concatenate the ``text`` of blocks where
``block["type"] == "text"``. Skip lines of type ``queue-operation`` /
``summary``, lines carrying an ``attachment`` (hook output), and lines whose
content has no text (tool_use / tool_result only).

``sessionId`` is the session id; ``cwd`` is present on message lines.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from scripts.agent_adapters import AgentSession, register
except ImportError:  # pragma: no cover - import-path shim
    from agent_adapters import AgentSession, register  # type: ignore


_KEEP_TYPES = {"user", "assistant"}
# Transcripts live one level down: ~/.claude/projects/<encoded>/*.jsonl
_GLOB = "*/*.jsonl"


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def parse(lines: list[str]) -> AgentSession:
    session_id = ""
    cwd = ""
    started_at = ""
    turns: list[dict] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue

        if obj.get("type") not in _KEEP_TYPES:
            continue
        # Hook output / attachment noise — skip.
        if obj.get("attachment") is not None:
            continue

        message = obj.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in _KEEP_TYPES:
            continue

        text = _content_text(message.get("content"))
        if not text.strip():
            continue

        if not session_id:
            session_id = str(obj.get("sessionId", "") or "")
        if not cwd:
            cwd = str(obj.get("cwd", "") or "")
        ts = str(obj.get("timestamp", "") or "")
        if not started_at and ts:
            started_at = ts

        turns.append({"ts": ts, "role": role, "text": text})

    return AgentSession(
        agent="claude",
        session_id=session_id,
        cwd=cwd,
        started_at=started_at,
        turns=turns,
    )


def default_store() -> Path:
    return Path("~/.claude/projects").expanduser()


register("claude", parse, default_store, _GLOB)
