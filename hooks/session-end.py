"""
SessionEnd hook - captures conversation transcript for memory extraction.

When a Claude Code session ends, this hook reads the transcript path from
stdin, extracts conversation context, and spawns flush.py as a background
process to extract knowledge into the daily log.

The hook itself does NO API calls - only local file I/O for speed (<10s).
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

# Recursion guard: if we were spawned by flush.py (which calls Agent SDK,
# which runs Claude Code, which would fire this hook again), exit immediately.
if os.environ.get("CLAUDE_INVOKED_BY"):
    sys.exit(0)

# Hook disable mechanism: set MEMORY_COMPILER_DISABLED_HOOKS to skip hooks.
# Values: "all" (disable everything), or comma-separated names like "session-start,session-end"
_disabled = os.environ.get("MEMORY_COMPILER_DISABLED_HOOKS", "").lower().split(",")
if "all" in _disabled or "session-end" in _disabled:
    sys.exit(0)

ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / "daily"
SCRIPTS_DIR = ROOT / "scripts"
STATE_DIR = SCRIPTS_DIR

sys.path.insert(0, str(SCRIPTS_DIR))
from flush_cursor import load_cursor  # noqa: E402
from log_setup import configure as configure_logging  # noqa: E402
from pending_flush import create_pending_flush, load_pending_flushes  # noqa: E402
from transcript import archive_transcript, extract_conversation_context  # noqa: E402

configure_logging(
    SCRIPTS_DIR / "flush.log",
    "%(asctime)s %(levelname)s [hook] %(message)s",
)

MIN_TURNS_TO_FLUSH = 1


def _failure(message: str, *args) -> int:
    logging.error(message, *args)
    print(message % args if args else message, file=sys.stderr)
    return 1


def main() -> int:
    # Read hook input from stdin
    # Claude Code on Windows may pass paths with unescaped backslashes
    try:
        raw_input = sys.stdin.read()
        try:
            hook_input: dict = json.loads(raw_input)
        except json.JSONDecodeError:
            fixed_input = re.sub(r'(?<!\\)\\(?!["\\])', r'\\\\', raw_input)
            hook_input = json.loads(fixed_input)
    except (json.JSONDecodeError, ValueError, EOFError) as e:
        return _failure("Failed to parse stdin: %s", e)

    session_id = hook_input.get("session_id", "unknown")
    source = hook_input.get("source", "unknown")
    transcript_path_str = hook_input.get("transcript_path", "")

    logging.info("SessionEnd fired: session=%s source=%s", session_id, source)

    if not transcript_path_str or not isinstance(transcript_path_str, str):
        return _failure("No transcript path; session capture was not scheduled")

    transcript_path = Path(transcript_path_str)
    if not transcript_path.exists():
        # On Windows, Claude Code sometimes emits an uppercase drive letter (C:\...)
        # but the actual filesystem path uses lowercase (c:\...). Try both.
        if sys.platform == "win32" and len(transcript_path_str) >= 2 and transcript_path_str[1] == ":":
            alt = transcript_path_str[0].swapcase() + transcript_path_str[1:]
            transcript_path = Path(alt)
        if not transcript_path.exists():
            return _failure("transcript missing: %s", transcript_path_str)

    try:
        transcript_archive = archive_transcript(transcript_path, session_id)
        logging.info("Archived raw transcript: %s", transcript_archive)
    except (OSError, ValueError) as e:
        return _failure(
            "Raw transcript archive failed; refusing to advance cursor: %s", e
        )

    # Only summarize turns this session hasn't flushed yet. A PreCompact flush
    # earlier in the same session already covered everything up to the cursor.
    cursor = load_cursor(session_id)

    # Extract conversation context in the hook (fast, no API calls)
    try:
        context, total_turns, turn_count = extract_conversation_context(
            transcript_archive, start_turn=cursor
        )
    except Exception as e:
        return _failure("Context extraction failed: %s", e)

    if not context.strip():
        logging.info("No new text turns; scheduling raw transcript index only")
    elif turn_count < MIN_TURNS_TO_FLUSH:
        logging.info(
            "SKIP: only %d new turns since cursor %d (min %d)",
            turn_count, cursor, MIN_TURNS_TO_FLUSH,
        )
        return 0

    # Write context to a temp file for the background process
    try:
        context_file = create_pending_flush(
            STATE_DIR,
            "session-flush",
            context,
            session_id,
            total_turns,
            transcript_archive,
        )
    except OSError as e:
        return _failure("Failed to persist pending flush: %s", e)

    # Spawn flush.py as a background process
    flush_script = SCRIPTS_DIR / "flush.py"

    # On Windows, use CREATE_NO_WINDOW to avoid flash console window.
    # Do NOT use DETACHED_PROCESS — it breaks the Agent SDK's subprocess I/O.
    creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    try:
        jobs = load_pending_flushes(STATE_DIR)
        for _marker, job in jobs:
            cmd = [
                sys.executable,
                str(flush_script),
                job["context_file"],
                job["session_id"],
                str(job["new_cursor"]),
                job["transcript_archive"],
            ]
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
            )
        logging.info(
            "Spawned %d pending flush job(s); current session %s has %d new turns "
            "(cursor %d -> %d, %d chars)",
            len(jobs), session_id, turn_count, cursor, total_turns, len(context),
        )
    except Exception as e:
        return _failure("Failed to spawn pending flush job(s): %s", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
