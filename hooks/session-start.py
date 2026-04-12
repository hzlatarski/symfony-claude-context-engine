"""
SessionStart hook - injects knowledge base context into every conversation.

This is the "context injection" layer. When Claude Code starts a session,
this hook reads the knowledge base index and recent daily log, then injects
them as additional context so Claude always "remembers" what it has learned.

Configure in .claude/settings.json:
{
    "hooks": {
        "SessionStart": [{
            "matcher": "",
            "command": "uv run python hooks/session-start.py"
        }]
    }
}
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Hook disable mechanism: set MEMORY_COMPILER_DISABLED_HOOKS to skip hooks.
# Values: "all" (disable everything), or comma-separated names like "session-start,session-end"
_disabled = os.environ.get("MEMORY_COMPILER_DISABLED_HOOKS", "").lower().split(",")
if "all" in _disabled or "session-start" in _disabled:
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ""}}))
    sys.exit(0)

# Paths relative to project root
ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge"
DAILY_DIR = ROOT / "daily"
INDEX_FILE = KNOWLEDGE_DIR / "index.md"
WIP_FILE = ROOT / "wip.md"

MAX_CONTEXT_CHARS = 60_000
MAX_LOG_LINES = 30
MAX_WIP_CHARS = 2_000
# compiled-truth.md lives in the PROJECT root's knowledge/ dir (written by config.py's
# KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"), which is two levels up from the
# memory-compiler root. session-start.py's KNOWLEDGE_DIR points to the memory-compiler's
# own knowledge/ dir, which is a different path.
PROJECT_KNOWLEDGE_DIR = ROOT.parent.parent / "knowledge"
COMPILED_TRUTH_FILE = PROJECT_KNOWLEDGE_DIR / "compiled-truth.md"
MAX_COMPILED_TRUTH_CHARS = 10_000
STATE_FILE = ROOT / "scripts" / "state.json"
FLUSH_STATE_FILE = ROOT / "scripts" / "last-flush.json"


def get_recent_log() -> str:
    """Read the most recent daily log (today or yesterday)."""
    today = datetime.now(timezone.utc).astimezone()

    for offset in range(2):
        date = today - timedelta(days=offset)
        log_path = DAILY_DIR / f"{date.strftime('%Y-%m-%d')}.md"
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8").splitlines()
            # Return last N lines to keep context small
            recent = lines[-MAX_LOG_LINES:] if len(lines) > MAX_LOG_LINES else lines
            return "\n".join(recent)

    return "(no recent daily log)"


def get_wip() -> str | None:
    """Read wip.md if it exists and has content. Returns None if absent/empty."""
    if not WIP_FILE.exists():
        return None
    content = WIP_FILE.read_text(encoding="utf-8").strip()
    if not content:
        return None
    if len(content) > MAX_WIP_CHARS:
        content = content[:MAX_WIP_CHARS] + "\n\n...(truncated)"
    return content


def get_compiled_truth() -> str | None:
    """Read compiled-truth.md if it exists. Returns None if absent/empty."""
    if not COMPILED_TRUTH_FILE.exists():
        return None
    content = COMPILED_TRUTH_FILE.read_text(encoding="utf-8").strip()
    if not content:
        return None
    if len(content) > MAX_COMPILED_TRUTH_CHARS:
        # Truncate at the last complete article boundary
        truncated = content[:MAX_COMPILED_TRUTH_CHARS]
        last_sep = truncated.rfind("\n---\n")
        if last_sep > 0:
            truncated = truncated[:last_sep]
        content = truncated + "\n\n...(truncated)"
    return content


def get_codebase_summary() -> str | None:
    """Quick codebase shape summary for session-start injection.

    Uses parser `summary()` functions which are designed to be fast (no
    `parse()` calls — only glob counts + cached git-intel.json). Must stay
    under ~500ms total and return None on any failure so we never block
    session start.
    """
    try:
        # Ensure scripts/ is on sys.path for the parser package import
        import sys as _sys
        scripts_dir = (ROOT / "scripts").resolve()
        if str(scripts_dir.parent) not in _sys.path:
            _sys.path.insert(0, str(scripts_dir.parent))

        from scripts.parsers import php_graph, stimulus_map, git_intel
        php = php_graph.summary(ROOT.parent.parent)
        stim = stimulus_map.summary(ROOT.parent.parent)
        git = git_intel.summary(ROOT.parent.parent)
        twig_count = sum(1 for _ in (ROOT.parent.parent / "templates").rglob("*.twig"))

        return (
            f"- PHP: {php}\n"
            f"- Templates: {twig_count} Twig files\n"
            f"- Stimulus: {stim}\n"
            f"- {git}\n"
            f"- MCP tools: get_file_deps, get_route_map, get_template_graph, "
            f"get_stimulus_map, get_hotspots, get_codebase_overview"
        )
    except Exception:
        return None


def get_cost_summary() -> str | None:
    """Build a compact cost summary from today and this month."""
    now = datetime.now(timezone.utc).astimezone()
    day_start_ts = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    month_start_ts = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()

    # Read flush costs
    flush_state = {}
    if FLUSH_STATE_FILE.exists():
        try:
            flush_state = json.loads(FLUSH_STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    today_flushes = [
        e for e in flush_state.get("flush_costs", [])
        if e.get("timestamp", 0) >= day_start_ts
    ]
    month_flushes = [
        e for e in flush_state.get("flush_costs", [])
        if e.get("timestamp", 0) >= month_start_ts
    ]

    # Read compile/ingest costs
    state = {}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    def parse_iso(ts: str) -> float:
        try:
            return datetime.fromisoformat(ts).timestamp()
        except (ValueError, TypeError):
            return 0.0

    today_compile = sum(
        e.get("cost_usd", 0.0)
        for e in state.get("ingested_daily", {}).values()
        if parse_iso(e.get("compiled_at", "")) >= day_start_ts
    )
    today_ingest = sum(
        e.get("cost_usd", 0.0)
        for e in state.get("ingested_sources", {}).values()
        if parse_iso(e.get("ingested_at", "")) >= day_start_ts
    )
    today_flush_cost = sum(e.get("cost_usd", 0.0) for e in today_flushes)
    today_total = today_flush_cost + today_compile + today_ingest

    month_flush_cost = sum(e.get("cost_usd", 0.0) for e in month_flushes)
    month_compile = sum(
        e.get("cost_usd", 0.0)
        for e in state.get("ingested_daily", {}).values()
        if parse_iso(e.get("compiled_at", "")) >= month_start_ts
    )
    month_ingest = sum(
        e.get("cost_usd", 0.0)
        for e in state.get("ingested_sources", {}).values()
        if parse_iso(e.get("ingested_at", "")) >= month_start_ts
    )
    month_total = month_flush_cost + month_compile + month_ingest

    if today_total == 0 and month_total == 0:
        return None

    parts = []
    if today_total > 0:
        parts.append(
            f"Today: Flushes {len(today_flushes)}x ${today_flush_cost:.2f}"
            f" | Compile ${today_compile:.2f}"
            f" | Ingest ${today_ingest:.2f}"
            f" | **Total ${today_total:.2f}**"
        )
    if month_total > 0:
        parts.append(f"This month: ${month_total:.2f}")

    return "\n".join(parts)


def build_context() -> str:
    """Assemble the context to inject into the conversation."""
    parts = []

    # Today's date
    today = datetime.now(timezone.utc).astimezone()
    parts.append(f"## Today\n{today.strftime('%A, %B %d, %Y')}")

    # Autocompact threshold check
    autocompact = os.environ.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE")
    if not autocompact:
        parts.append(
            "## Warning\nCLAUDE_AUTOCOMPACT_PCT_OVERRIDE is not set. "
            "Set it to 50 for better long-session quality and more reliable pre-compact flushes."
        )

    # Cost summary — passive awareness of spending
    cost_summary = get_cost_summary()
    if cost_summary:
        parts.append(f"## Cost Summary\n{cost_summary}")

    # Codebase shape — lightweight (file counts + hotspots from cache)
    codebase = get_codebase_summary()
    if codebase:
        parts.append(f"## Codebase Shape\n{codebase}")

    # Work In Progress — "resume here" state from the last session that
    # ended mid-task. Placed second so Claude sees it immediately after
    # the date, before the larger knowledge base index.
    wip = get_wip()
    if wip:
        parts.append(f"## Work In Progress (resume here)\n\n{wip}")

    # Knowledge base index (the core retrieval mechanism)
    if INDEX_FILE.exists():
        index_content = INDEX_FILE.read_text(encoding="utf-8")
        parts.append(f"## Knowledge Base Index\n\n{index_content}")
    else:
        parts.append("## Knowledge Base Index\n\n(empty - no articles compiled yet)")

    # Compiled truth — dense summary of all current knowledge
    compiled_truth = get_compiled_truth()
    if compiled_truth:
        parts.append(f"## Compiled Truth (all current knowledge)\n\n{compiled_truth}")

    # Recent daily log
    recent_log = get_recent_log()
    parts.append(f"## Recent Daily Log\n\n{recent_log}")

    context = "\n\n---\n\n".join(parts)

    # Truncate if too long
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n...(truncated)"

    return context


def main():
    context = build_context()

    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
