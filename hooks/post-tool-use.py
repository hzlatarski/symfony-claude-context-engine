"""PostToolUse hook — structured tool-call drawer for lossless session capture.

Fires after every tool call Claude Code makes. Writes a single JSON line per
invocation to ``knowledge/daily/YYYY-MM-DD.tools.jsonl`` containing the tool
name, a compact input digest, the response size, and a success flag. The
resulting drawer is machine-readable, so ``flush.py`` no longer has to
reconstruct session events from the text transcript — a much lossier path.

Design constraints (in priority order):

1. **Never break the session.** A PostToolUse hook that errors or hangs
   degrades every tool call. All I/O is wrapped in broad try/except; any
   exception is swallowed silently (the whole point of this file is best-effort
   observability).
2. **Fast.** Pure stdlib imports only — no chromadb, no agent SDK, no LLM. A
   single fsync'd append per tool call is cheap enough at ~500 calls/session.
3. **Tool-aware digests.** Raw ``tool_input`` is often huge (a full Edit payload,
   a multi-kilobyte Bash stdout). We keep only the load-bearing fields per tool
   and cap everything else to ``_MAX_DIGEST_CHARS``.
4. **Respect the same disable + recursion guards as the other hooks.** Subagents
   spawned by flush.py must not recursively feed the drawer.

Output format (JSONL, one object per line)::

    {"ts": "2026-04-14T07:50:12+02:00", "session_id": "...", "tool": "Edit",
     "input": {"file_path": "..."}, "result_size": 1234, "ok": true}

The drawer is idempotent-safe: duplicated lines from a replayed session do no
harm because the compile pipeline already de-dupes on content hash.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Recursion guard: flush.py spawns Claude via the Agent SDK, which would fire
# PostToolUse hooks in the child session. We must NOT recursively append to
# the drawer from those sub-sessions — otherwise a single flush blows out
# the JSONL with duplicate sub-agent noise.
if os.environ.get("CLAUDE_INVOKED_BY"):
    sys.exit(0)

# Hook disable mechanism shared with the other memory-compiler hooks.
_disabled = os.environ.get("MEMORY_COMPILER_DISABLED_HOOKS", "").lower().split(",")
if "all" in _disabled or "post-tool-use" in _disabled:
    sys.exit(0)

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"

# Pull DAILY_DIR from config so the drawer lands next to the human-readable
# daily log (knowledge/daily/YYYY-MM-DD.md) — the outer project path, not
# .claude/memory-compiler/daily/ which is a stale earlier location.
sys.path.insert(0, str(SCRIPTS_DIR))
try:
    from config import DAILY_DIR  # noqa: E402
except Exception:  # pragma: no cover — absolute last-resort fallback
    DAILY_DIR = ROOT.parent.parent / "knowledge" / "daily"

logging.basicConfig(
    filename=str(SCRIPTS_DIR / "flush.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [post-tool-use] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Cap every digest value at this length. Bash commands, Grep patterns, and
# Task descriptions can run long; the drawer is for shape + intent, not for
# byte-perfect reproduction (the raw transcript still has that).
_MAX_DIGEST_CHARS = 240


def _truncate(value: object) -> object:
    """Coerce to str and hard-cap at _MAX_DIGEST_CHARS."""
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value)
    if len(text) <= _MAX_DIGEST_CHARS:
        return text
    return text[:_MAX_DIGEST_CHARS] + "…"


def _first_line(value: object) -> object:
    """For commands/queries: keep first line only, then truncate."""
    if not isinstance(value, str):
        return _truncate(value)
    head = value.splitlines()[0] if value else ""
    return _truncate(head)


def build_digest(tool_name: str, tool_input: dict) -> dict:
    """Extract the load-bearing fields for a given tool into a small dict.

    The goal is "enough to reconstruct what the session was DOING without
    storing the full payload." For well-known tools we pick hand-chosen
    fields; for unknown tools we keep top-level keys under the cap and drop
    anything large.
    """
    if not isinstance(tool_input, dict):
        return {"raw": _truncate(tool_input)}

    # Tool-specific extractors keep only the fields a reader cares about.
    # Order matches Claude Code's most common tools; fall-through below
    # handles anything else.
    if tool_name in ("Read", "Write", "Edit", "NotebookEdit"):
        return {
            "file_path": _truncate(tool_input.get("file_path")),
        }
    if tool_name == "Bash":
        return {
            "command": _first_line(tool_input.get("command")),
            "run_in_background": tool_input.get("run_in_background", False),
        }
    if tool_name == "Grep":
        return {
            "pattern": _truncate(tool_input.get("pattern")),
            "path": _truncate(tool_input.get("path")),
            "output_mode": tool_input.get("output_mode"),
        }
    if tool_name == "Glob":
        return {
            "pattern": _truncate(tool_input.get("pattern")),
            "path": _truncate(tool_input.get("path")),
        }
    if tool_name in ("Task", "Agent"):
        return {
            "description": _truncate(tool_input.get("description")),
            "subagent_type": tool_input.get("subagent_type"),
        }
    if tool_name == "WebFetch":
        return {
            "url": _truncate(tool_input.get("url")),
        }
    if tool_name == "WebSearch":
        return {
            "query": _truncate(tool_input.get("query")),
        }
    if tool_name == "TodoWrite":
        todos = tool_input.get("todos") or []
        return {"todo_count": len(todos) if isinstance(todos, list) else 0}
    if tool_name == "Skill":
        return {
            "skill": _truncate(tool_input.get("skill")),
            "args": _truncate(tool_input.get("args")),
        }

    # Unknown tool: best-effort summary. Skip keys whose values are obviously
    # large (lists/dicts/long strings) to avoid dumping full payloads.
    digest: dict = {}
    for key, value in tool_input.items():
        if isinstance(value, (dict, list)):
            continue
        if isinstance(value, str) and len(value) > _MAX_DIGEST_CHARS:
            digest[key] = _truncate(value)
        else:
            digest[key] = value
    return digest


def measure_response(tool_response: object) -> tuple[int, bool]:
    """Return (size_bytes_approx, ok_flag) for a tool response.

    Claude Code wraps tool results in varied shapes — sometimes a plain string,
    sometimes a dict with ``content``, sometimes a list of content blocks.
    We approximate size via ``len(str(...))`` and sniff for the ``is_error``
    flag or a top-level ``error`` key. Heuristic, but good enough for the
    drawer's "did it blow up?" signal.
    """
    ok = True
    if isinstance(tool_response, dict):
        if tool_response.get("is_error") or tool_response.get("error"):
            ok = False
    size = len(str(tool_response)) if tool_response is not None else 0
    return size, ok


def append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON line to ``path``, creating the parent dir if needed.

    Opens in append mode with UTF-8; the OS-level O_APPEND guarantees each
    ``write`` is atomic at the record boundary on POSIX, and a single-writer
    hook on Windows never competes for the file handle, so interleaved lines
    are not a concern.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _drawer_path_for_today() -> Path:
    today = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    return DAILY_DIR / f"{today}.tools.jsonl"


def main() -> None:
    try:
        raw_input_str = sys.stdin.read()
    except Exception as exc:
        logging.warning("stdin read failed: %s", exc)
        return

    if not raw_input_str:
        return

    try:
        hook_input = json.loads(raw_input_str)
    except json.JSONDecodeError:
        # Windows path escaping fallback — same pattern used by session-end.py.
        try:
            fixed = re.sub(r'(?<!\\)\\(?!["\\])', r'\\\\', raw_input_str)
            hook_input = json.loads(fixed)
        except Exception as exc:
            logging.warning("stdin JSON parse failed: %s", exc)
            return

    tool_name = hook_input.get("tool_name") or "unknown"
    tool_input = hook_input.get("tool_input") or {}
    tool_response = hook_input.get("tool_response")
    session_id = hook_input.get("session_id") or "unknown"

    try:
        digest = build_digest(tool_name, tool_input)
        size, ok = measure_response(tool_response)
    except Exception as exc:
        logging.warning("digest build failed for %s: %s", tool_name, exc)
        return

    record = {
        "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "session_id": session_id,
        "tool": tool_name,
        "input": digest,
        "result_size": size,
        "ok": ok,
    }

    try:
        append_jsonl(_drawer_path_for_today(), record)
    except Exception as exc:
        # Never let a drawer write fail the session — just log and move on.
        logging.warning("drawer append failed: %s", exc)


if __name__ == "__main__":
    main()
