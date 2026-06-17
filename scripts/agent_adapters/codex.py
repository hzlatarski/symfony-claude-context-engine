"""Codex CLI conversation-history adapter.

On-disk format: ``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl``.

Line types:
* First line — ``{"type": "session_meta", "payload": {"id", "timestamp",
  "cwd", ...}}``.
* Turn lines — ``{"type": "response_item", "payload": {"type": "message",
  "role": "user|assistant|developer", "content": [{"type": "input_text"|
  "output_text"|"text", "text": ...}]}}``.

Parse rules: pull ``id``/``cwd``/``timestamp`` from ``session_meta.payload``.
For each ``response_item`` whose ``payload.type == "message"`` and
``role in {user, assistant}`` (skip ``developer``/``system``), concatenate
the ``.text`` of content items whose ``type`` is in
``{input_text, output_text, text}``. Non-message response_items (tool calls,
reasoning) are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from scripts.agent_adapters import AgentSession, register
except ImportError:  # pragma: no cover - import-path shim
    from agent_adapters import AgentSession, register  # type: ignore


_KEEP_ROLES = {"user", "assistant"}
_TEXT_TYPES = {"input_text", "output_text", "text"}
_GLOB = "**/rollout-*.jsonl"


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") in _TEXT_TYPES:
            text = item.get("text")
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

        ltype = obj.get("type")
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            continue

        if ltype == "session_meta":
            session_id = str(payload.get("id", "") or "")
            cwd = str(payload.get("cwd", "") or "")
            started_at = str(payload.get("timestamp", "") or obj.get("timestamp", "") or "")
            continue

        if ltype != "response_item":
            continue
        if payload.get("type") != "message":
            continue

        role = payload.get("role")
        if role not in _KEEP_ROLES:
            continue

        text = _content_text(payload.get("content"))
        if not text.strip():
            continue

        turns.append({
            "ts": str(obj.get("timestamp", "") or ""),
            "role": role,
            "text": text,
        })

    return AgentSession(
        agent="codex",
        session_id=session_id,
        cwd=cwd,
        started_at=started_at,
        turns=turns,
    )


def default_store() -> Path:
    return Path("~/.codex/sessions").expanduser()


register("codex", parse, default_store, _GLOB)
