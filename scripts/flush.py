"""
Memory flush agent - extracts important knowledge from conversation context.

Spawned by session-end.py or pre-compact.py as a background process. Reads
pre-extracted conversation context from a .md file, uses the Claude Agent SDK
to decide what's worth saving, and appends the result to today's daily log.

Usage:
    uv run python flush.py <context_file.md> <session_id>
"""

from __future__ import annotations

# Recursion prevention: set this BEFORE any imports that might trigger Claude
import os
os.environ["CLAUDE_INVOKED_BY"] = "memory_flush"

import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
STATE_FILE = SCRIPTS_DIR / "last-flush.json"
LOG_FILE = SCRIPTS_DIR / "flush.log"
WIP_FILE = ROOT / "wip.md"

# Daily logs live at ``knowledge/daily/`` in the outer project, NOT inside
# ``.claude/memory-compiler/daily/``. Pull DAILY_DIR from config.py so
# flush.py stays in sync with compile.py / reindex.py / embed_daily_file.
# Before 2026-04-12 flush.py computed DAILY_DIR locally as ROOT / "daily"
# which silently landed session flushes at a stale path the rest of the
# pipeline never saw.
sys.path.insert(0, str(SCRIPTS_DIR))
from config import DAILY_DIR  # noqa: E402

# Set up file-based logging so we can verify the background process ran.
# The parent process sends stdout/stderr to DEVNULL (to avoid the inherited
# file handle bug on Windows), so this is our only observability channel.
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_flush_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_flush_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state), encoding="utf-8")


# -----------------------------------------------------------------------------
# Tool drawer — structured PostToolUse events captured by hooks/post-tool-use.py
# -----------------------------------------------------------------------------
#
# The PostToolUse hook writes one JSONL line per tool call to
# ``knowledge/daily/YYYY-MM-DD.tools.jsonl``. Unlike the transcript (which
# Haiku has to re-parse from markdown), these records are already structured:
# tool name, input digest, result size, ok flag. Injecting a compact summary
# of them into the flush prompt gives the model ground-truth about what the
# session actually DID, so its summary of decisions and actions is less
# reconstruction and more reporting.
#
# Tiering: higher-priority tools are the ones that represent "real work"
# (file changes, shell commands, subagent dispatches, web fetches). Read /
# Grep / Glob are discovery noise — useful for counts but rarely for the
# "notable operations" list. TodoWrite is internal planning and never
# surfaces as a notable operation.

_TOOL_PRIORITY: dict[str, int] = {
    "Edit": 3, "Write": 3, "NotebookEdit": 3,
    "Bash": 3,
    "Task": 3, "Agent": 3,
    "WebFetch": 2, "WebSearch": 2,
    "Skill": 2,
    "Read": 1, "Grep": 1, "Glob": 1,
    "TodoWrite": 0,
}


def _describe_event(tool: str, inp: dict) -> str:
    """One-line human label for a tool event, tool-aware."""
    if not isinstance(inp, dict):
        return ""
    if tool in ("Edit", "Write", "Read", "NotebookEdit"):
        return inp.get("file_path", "") or ""
    if tool == "Bash":
        return inp.get("command", "") or ""
    if tool == "Grep":
        pattern = inp.get("pattern", "") or ""
        path = inp.get("path", "") or ""
        return f"{pattern}" + (f" in {path}" if path else "")
    if tool == "Glob":
        pattern = inp.get("pattern", "") or ""
        path = inp.get("path", "") or ""
        return f"{pattern}" + (f" in {path}" if path else "")
    if tool in ("Task", "Agent"):
        desc = inp.get("description", "") or ""
        sub = inp.get("subagent_type") or "?"
        return f"[{sub}] {desc}"
    if tool == "WebFetch":
        return inp.get("url", "") or ""
    if tool == "WebSearch":
        return inp.get("query", "") or ""
    if tool == "Skill":
        return inp.get("skill", "") or ""
    return ""


def load_tool_events(
    session_id: str,
    daily_dir: Path,
    today_iso: str | None = None,
) -> list[dict]:
    """Read today's tool drawer and return events for ``session_id``.

    Defensive against every failure mode the drawer could present: missing
    file, malformed JSONL, events from other sessions, corrupt metadata.
    A flush must NEVER crash because the drawer is weird — the transcript
    path alone still produces a usable summary.
    """
    if today_iso is None:
        today_iso = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    path = daily_dir / f"{today_iso}.tools.jsonl"
    if not path.exists():
        return []

    events: list[dict] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logging.warning("tool drawer read failed: %s", exc)
        return []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("session_id") == session_id:
            events.append(event)
    return events


def format_tool_events(events: list[dict], max_notable: int = 40) -> str:
    """Render tool events into a compact plain-text block for the flush prompt.

    Two sections: a one-line counts summary, and a ranked "notable operations"
    list capped at ``max_notable``. Ranking is tool-priority first (so file
    edits and shell commands surface ahead of reads), then original order to
    preserve causality within a tier.

    Returns an empty string if ``events`` is empty — callers should skip the
    whole prompt section in that case.
    """
    if not events:
        return ""

    counts: dict[str, int] = {}
    for event in events:
        tool = event.get("tool") or "unknown"
        counts[tool] = counts.get(tool, 0) + 1

    counts_str = ", ".join(
        f"{tool}: {count}"
        for tool, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )

    ranked = sorted(
        enumerate(events),
        key=lambda iv: (
            -_TOOL_PRIORITY.get(iv[1].get("tool") or "", 0),
            iv[0],
        ),
    )

    notable_lines: list[str] = []
    for _, event in ranked:
        if len(notable_lines) >= max_notable:
            break
        tool = event.get("tool") or "?"
        if _TOOL_PRIORITY.get(tool, 0) == 0:
            continue  # skip TodoWrite and other internal-planning noise
        label = _describe_event(tool, event.get("input") or {})
        if not label:
            continue
        err_suffix = "" if event.get("ok", True) else " [ERROR]"
        notable_lines.append(f"- [{tool}] {label}{err_suffix}")

    lines = [
        f"Tool calls this session: {len(events)} ({counts_str})",
    ]
    if notable_lines:
        lines.append("")
        lines.append("Notable operations (ranked by significance):")
        lines.extend(notable_lines)
    return "\n".join(lines)


def append_to_daily_log(content: str, section: str = "Session") -> None:
    """Append content to today's daily log.

    After the append, re-chunk and re-embed the whole daily file into
    ChromaDB's daily_chunks collection so the verbatim drawer layer
    stays in sync with the raw text. Embedding failures are swallowed
    with a warning — flush runs as a detached background process and
    must not crash on vector store hiccups.
    """
    today = datetime.now(timezone.utc).astimezone()
    log_path = DAILY_DIR / f"{today.strftime('%Y-%m-%d')}.md"

    if not log_path.exists():
        DAILY_DIR.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"# Daily Log: {today.strftime('%Y-%m-%d')}\n\n## Sessions\n\n## Memory Maintenance\n\n",
            encoding="utf-8",
        )

    time_str = today.strftime("%H:%M")
    entry = f"### {section} ({time_str})\n\n{content}\n\n"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)

    # Vector re-embed (verbatim drawer layer — MemPalace steal item).
    # Non-fatal: logs and continues so flush never aborts on a Chroma
    # or ONNX hiccup.
    try:
        from utils import embed_daily_file
        count = embed_daily_file(log_path)
        logging.info("Embedded %d daily chunks into vector store for %s", count, log_path.name)
    except Exception as exc:
        logging.warning("Vector embed skipped for %s: %s", log_path.name, exc)


# Matches a **Heading:** bold section at the start of a line. Used to scope
# the Work In Progress extraction so inline **bold** inside the body doesn't
# prematurely terminate the match.
_WIP_SECTION_RE = re.compile(
    r"\*\*Work In Progress:\*\*\s*(.*?)(?=\n\*\*[A-Z][\w /&-]*:\*\*|\Z)",
    re.DOTALL,
)

_WIP_EMPTY_MARKERS = {"", "(none)", "none", "n/a", "-", "—"}


def extract_wip_section(response: str) -> str | None:
    """Pull the **Work In Progress:** block out of the flush response.

    Returns None if the section is missing, empty, or explicitly marked as
    having nothing in flight — in which case the caller should leave the
    existing wip.md alone (last known state remains the source of truth).
    """
    match = _WIP_SECTION_RE.search(response)
    if not match:
        return None
    content = match.group(1).strip()
    if content.lower() in _WIP_EMPTY_MARKERS:
        return None
    return content


def update_wip_file(wip_content: str) -> None:
    """Rewrite wip.md (not append) with the latest resume-here snapshot."""
    today = datetime.now(timezone.utc).astimezone()
    header = (
        "# Work In Progress\n\n"
        f"_Last updated: {today.strftime('%Y-%m-%d %H:%M %Z').strip()}_\n\n"
    )
    WIP_FILE.write_text(header + wip_content + "\n", encoding="utf-8")


async def run_flush(context: str, tool_events_text: str = "") -> tuple[str, float]:
    """Use Claude Agent SDK to extract important knowledge from conversation context.

    ``tool_events_text`` is the rendered drawer summary from
    ``format_tool_events`` — empty string when no PostToolUse drawer was
    captured (early sessions, disabled hook, etc).

    Returns (response_text, cost_usd).
    """
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    tool_events_section = (
        f"\n## Tool Activity (ground truth)\n\n"
        f"The following structured log comes from the PostToolUse hook — "
        f"each line is a real tool call Claude Code made during the session. "
        f"Use it as **ground truth** for what was actually done. When the "
        f"conversation text and the tool log disagree, trust the tool log. "
        f"Cite file paths and commands from here when summarizing decisions "
        f"and actions.\n\n"
        f"```\n{tool_events_text}\n```\n"
        if tool_events_text
        else ""
    )

    prompt = f"""Review the conversation context below and respond with a concise summary
of important items that should be preserved in the daily log.
Do NOT use any tools — just return plain text.

Format your response as a structured daily log entry with these sections:

**Context:** [One line about what the user was working on]

**Key Exchanges:**
- [Important Q&A or discussions]

**Decisions Made:**
- [Any decisions with rationale]

**Lessons Learned:**
- [Gotchas, patterns, or insights discovered]

**Action Items:**
- [Follow-ups or TODOs mentioned]

**Work In Progress:**
- [If the session ended mid-task, describe the current resume-here state:
  which file/function was being edited, which branch, what's half-done,
  and the exact next concrete step to pick up. Keep it tight — 3-6 bullets max.]
- [OMIT this section entirely if the session wrapped as a complete unit of
  work with nothing in flight. Do not invent WIP that isn't there.]

Skip anything that is:
- Routine tool calls or file reads
- Content that's trivial or obvious
- Trivial back-and-forth or clarification exchanges

Only include sections that have actual content. If nothing is worth saving,
respond with exactly: FLUSH_OK
{tool_events_section}
## Conversation Context

{context}"""

    response = ""
    cost = 0.0

    # Use Haiku for flush — simple extraction, cheapest model (~60% savings)
    try:
        from config import MODEL_FLUSH
        model = MODEL_FLUSH
    except ImportError:
        model = "claude-haiku-4-5-20251001"

    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                cwd=str(ROOT),
                model=model,
                allowed_tools=[],
                max_turns=2,
            ),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response += block.text
            elif isinstance(message, ResultMessage):
                cost = message.total_cost_usd or 0.0
    except Exception as e:
        import traceback
        logging.error("Agent SDK error: %s\n%s", e, traceback.format_exc())
        response = f"FLUSH_ERROR: {type(e).__name__}: {e}"

    return response, cost


COMPILE_AFTER_HOUR = 18  # 6 PM local time


def maybe_trigger_compilation() -> None:
    """If it's past the compile hour and today's log hasn't been compiled, run compile.py."""
    import subprocess as _sp

    now = datetime.now(timezone.utc).astimezone()
    if now.hour < COMPILE_AFTER_HOUR:
        return

    # Check if today's log has already been compiled
    today_log = f"{now.strftime('%Y-%m-%d')}.md"
    compile_state_file = SCRIPTS_DIR / "state.json"
    if compile_state_file.exists():
        try:
            compile_state = json.loads(compile_state_file.read_text(encoding="utf-8"))
            ingested = compile_state.get("ingested", {})
            if today_log in ingested:
                # Already compiled today - check if the log has changed since
                from hashlib import sha256
                log_path = DAILY_DIR / today_log
                if log_path.exists():
                    current_hash = sha256(log_path.read_bytes()).hexdigest()[:16]
                    if ingested[today_log].get("hash") == current_hash:
                        return  # log unchanged since last compile
        except (json.JSONDecodeError, OSError):
            pass

    compile_script = SCRIPTS_DIR / "compile.py"
    if not compile_script.exists():
        return

    logging.info("End-of-day compilation triggered (after %d:00)", COMPILE_AFTER_HOUR)

    cmd = ["uv", "run", "--directory", str(ROOT), "python", str(compile_script)]

    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = _sp.CREATE_NEW_PROCESS_GROUP | _sp.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True

    try:
        log_handle = open(str(SCRIPTS_DIR / "compile.log"), "a")
        _sp.Popen(cmd, stdout=log_handle, stderr=_sp.STDOUT, cwd=str(ROOT), **kwargs)
    except Exception as e:
        logging.error("Failed to spawn compile.py: %s", e)


def _today_flush_total(state: dict) -> float:
    """Sum flush costs from today."""
    today_start = datetime.now(timezone.utc).astimezone().replace(
        hour=0, minute=0, second=0, microsecond=0
    ).timestamp()
    return sum(
        entry.get("cost_usd", 0.0)
        for entry in state.get("flush_costs", [])
        if entry.get("timestamp", 0) >= today_start
    )


def main():
    if len(sys.argv) < 3:
        logging.error("Usage: %s <context_file.md> <session_id>", sys.argv[0])
        sys.exit(1)

    context_file = Path(sys.argv[1])
    session_id = sys.argv[2]

    logging.info("flush.py started for session %s, context: %s", session_id, context_file)

    if not context_file.exists():
        logging.error("Context file not found: %s", context_file)
        return

    # Deduplication: skip if same session was flushed within 60 seconds
    state = load_flush_state()
    if (
        state.get("session_id") == session_id
        and time.time() - state.get("timestamp", 0) < 60
    ):
        logging.info("Skipping duplicate flush for session %s", session_id)
        context_file.unlink(missing_ok=True)
        return

    # Read pre-extracted context
    context = context_file.read_text(encoding="utf-8").strip()
    if not context:
        logging.info("Context file is empty, skipping")
        context_file.unlink(missing_ok=True)
        return

    logging.info("Flushing session %s: %d chars", session_id, len(context))

    # Load structured tool events from today's PostToolUse drawer (if any).
    # This supplements the transcript with ground-truth about what the
    # session actually did — file edits, shell commands, subagent dispatches —
    # so Haiku doesn't have to reconstruct them from the conversation text.
    try:
        tool_events = load_tool_events(session_id, DAILY_DIR)
        tool_events_text = format_tool_events(tool_events)
        if tool_events_text:
            logging.info(
                "Loaded %d tool events for session %s (%d chars summary)",
                len(tool_events), session_id, len(tool_events_text),
            )
    except Exception as exc:
        logging.warning("Tool drawer load failed: %s", exc)
        tool_events_text = ""

    # Run the LLM extraction
    response, flush_cost = asyncio.run(run_flush(context, tool_events_text))

    # Append to daily log
    if "FLUSH_OK" in response:
        logging.info("Result: FLUSH_OK")
        append_to_daily_log(
            "FLUSH_OK - Nothing worth saving from this session", "Memory Flush"
        )
    elif "FLUSH_ERROR" in response:
        logging.error("Result: %s", response)
        append_to_daily_log(response, "Memory Flush")
    else:
        logging.info("Result: saved to daily log (%d chars)", len(response))
        append_to_daily_log(response, "Session")

        # Update wip.md only when the flush surfaced real in-flight work.
        # Clean-wrapped sessions leave the previous wip.md intact so the
        # resume-here file always reflects the last known WIP, never gets
        # clobbered to empty just because the latest session had nothing pending.
        wip = extract_wip_section(response)
        if wip:
            try:
                update_wip_file(wip)
                logging.info("Updated wip.md (%d chars)", len(wip))
            except OSError as e:
                logging.error("Failed to write wip.md: %s", e)

    # Update dedup state + cost tracking
    state = load_flush_state()
    state["session_id"] = session_id
    state["timestamp"] = time.time()
    flush_costs = state.get("flush_costs", [])
    flush_costs.append({
        "session_id": session_id,
        "timestamp": time.time(),
        "cost_usd": flush_cost,
        "result": "FLUSH_OK" if "FLUSH_OK" in response else ("error" if "FLUSH_ERROR" in response else "saved"),
    })
    state["flush_costs"] = flush_costs
    save_flush_state(state)


    # Clean up context file
    context_file.unlink(missing_ok=True)

    # End-of-day auto-compilation: if it's past the compile hour and today's
    # log hasn't been compiled yet, trigger compile.py in the background.
    # Safe now that compile.py uses index-only context (O(1) per file).
    maybe_trigger_compilation()

    logging.info("Flush complete for session %s", session_id)


if __name__ == "__main__":
    main()
