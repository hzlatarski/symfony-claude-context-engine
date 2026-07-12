"""Shared transcript reader for the flush hooks.

``session-end.py`` and ``pre-compact.py`` both need to turn a Claude Code JSONL
transcript into a markdown context blob for flush.py. They used to carry
byte-identical copies of this function, which is how one of them could grow a
turn cursor while the other kept re-reading from turn zero. One copy, here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

DEFAULT_MAX_TURNS = 30
DEFAULT_MAX_CONTEXT_CHARS = 15_000


def extract_conversation_context(
    transcript_path: Path,
    start_turn: int = 0,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> tuple[str, int, int]:
    """Extract the un-flushed tail of a transcript as markdown.

    ``start_turn`` is the flush cursor (see ``flush_cursor.py``): turns before
    it were already summarized into the daily log by an earlier flush of this
    same session, so including them again just yields duplicate log entries.

    Returns ``(context, total_turns, window_turns)``:
      * ``context``      — the markdown blob to hand to flush.py
      * ``total_turns``  — the new cursor to record once the flush succeeds
      * ``window_turns`` — how many fresh turns ``context`` actually covers
    """
    turns: list[str] = []

    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = entry.get("message", {})
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
            else:
                role = entry.get("role", "")
                content = entry.get("content", "")

            if role not in ("user", "assistant"):
                continue

            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "\n".join(text_parts)

            if isinstance(content, str) and content.strip():
                label = "User" if role == "user" else "Assistant"
                turns.append(f"**{label}:** {content.strip()}\n")

    total_turns = len(turns)

    # Clamp to [0, total]: a cursor at the end yields an empty window ("nothing
    # new since the last flush"), and a stale/oversized cursor cannot produce a
    # negative slice that silently re-flushes the whole transcript.
    start = max(0, min(start_turn, total_turns))
    fresh = turns[start:]

    recent = fresh[-max_turns:]

    # The window keeps only the most recent `max_turns` — they are the turns
    # worth summarizing and the ones wip.md is built from. But the caller will
    # advance the cursor past ALL fresh turns, so anything dropped here is never
    # summarized by any later flush. That is a real (pre-existing) gap; make it
    # visible rather than silent, so a session that routinely overflows shows up
    # in flush.log instead of quietly losing its early turns.
    dropped = len(fresh) - len(recent)
    if dropped > 0:
        logging.warning(
            "Flush window overflow: %d turn(s) between cursor %d and %d exceed the "
            "%d-turn window and will not be summarized",
            dropped, start, total_turns - len(recent), max_turns,
        )

    context = "\n".join(recent)

    if len(context) > max_context_chars:
        context = context[-max_context_chars:]
        boundary = context.find("\n**")
        if boundary > 0:
            context = context[boundary + 1 :]

    return context, total_turns, len(recent)
