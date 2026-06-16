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
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _project_slug(name: str) -> str:
    """Mirror install.py's slug logic: AiTutor → aitutor, My_Project → my-project."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "project"

# Hook disable mechanism: set MEMORY_COMPILER_DISABLED_HOOKS to skip hooks.
# Values: "all" (disable everything), or comma-separated names like "session-start,session-end"
_disabled = os.environ.get("MEMORY_COMPILER_DISABLED_HOOKS", "").lower().split(",")
if "all" in _disabled or "session-start" in _disabled:
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ""}}))
    sys.exit(0)

# Paths relative to project root
ROOT = Path(__file__).resolve().parent.parent
WIP_FILE = ROOT / "wip.md"

MAX_CONTEXT_CHARS = 60_000
MAX_LOG_LINES = 30
MAX_WIP_CHARS = 2_000
# Knowledge artifacts live in the PROJECT root's knowledge/ dir (written by
# config.py's KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"), which is two levels
# up from the memory-compiler root. The hook used to point INDEX_FILE/DAILY_DIR
# at memory-compiler's own knowledge/ and daily/ subdirs, which were stale —
# that is why every session said "empty - no articles compiled yet" even though
# index.md had 250+ articles.
PROJECT_KNOWLEDGE_DIR = ROOT.parent.parent / "knowledge"
INDEX_FILE = PROJECT_KNOWLEDGE_DIR / "index.md"
DAILY_DIR = PROJECT_KNOWLEDGE_DIR / "daily"
COMPILED_TRUTH_FILE = PROJECT_KNOWLEDGE_DIR / "compiled-truth.md"
MAX_COMPILED_TRUTH_CHARS = 10_000
MAX_INDEX_CHARS = 20_000
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


def get_update_notice() -> str | None:
    """Probe for a pending upgrade and return a markdown notice, or None.

    Runs ``scripts/check_update.py`` as a subprocess with a short timeout.
    The probe itself caches for 6h, so this runs at most a few times per
    day; the timeout guarantees we never stall the session if the network
    is slow.

    The notice tells the agent the user can run ``/memory-compiler-upgrade``
    to handle the prompt — that skill owns the AskUserQuestion flow,
    snooze logic, and the actual upgrade execution.
    """
    check_script = ROOT / "scripts" / "check_update.py"
    if not check_script.exists():
        return None
    try:
        # ``uv run python`` is the canonical invocation; the hook runs
        # from ``.claude/memory-compiler/`` so a relative path works.
        proc = subprocess.run(
            ["uv", "run", "python", str(check_script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    line = (proc.stdout or "").strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) != 3:
        return None
    tag, old, new = parts
    if tag == "UPGRADE_AVAILABLE":
        return (
            f"## Memory Compiler Update Available\n\n"
            f"A new memory-compiler release is available: **v{old} → v{new}**.\n\n"
            f"Tell the user that an update is available and run "
            f"`/memory-compiler-upgrade` to handle the upgrade prompt. "
            f"The skill walks the user through the choice "
            f"(upgrade now / always / not now / never ask again) and "
            f"performs the upgrade safely if accepted."
        )
    if tag == "JUST_UPGRADED":
        return (
            f"## Memory Compiler Just Upgraded\n\n"
            f"The memory compiler was just upgraded from **v{old}** to **v{new}**. "
            f"After replying, summarize what's new by reading "
            f"`.claude/memory-compiler/CHANGELOG.md` (entries between {old} and {new}) "
            f"in 5–7 bullets, then continue with the user's original request. "
            f"Once you've shown the summary, delete "
            f"`~/.memory-compiler/just-upgraded-from` so it doesn't repeat."
        )
    return None


def build_context() -> str:
    """Assemble the context to inject into the conversation."""
    parts = []

    # Today's date
    today = datetime.now(timezone.utc).astimezone()
    parts.append(f"## Today\n{today.strftime('%A, %B %d, %Y')}")

    # KB-first directive — the strongest, earliest nudge. Without this, Claude
    # routinely skips the knowledge base because the MCP tools are deferred
    # (their schemas aren't loaded by default — they require ToolSearch first).
    # Spelling the unlock command out removes the friction. The project slug
    # mirrors install.py's logic so the MCP tool names are correct for whichever
    # project this hook runs in.
    slug = _project_slug(ROOT.parent.parent.name)
    kb_prefix = f"mcp__{slug}-knowledge__"
    parts.append(
        "## Use the Knowledge Base FIRST\n\n"
        "Before answering any product, architecture, grading-logic, prompt, or "
        "**why-was-this-built-this-way** question, you MUST query the KB. The "
        "compiled-truth + index below are a summary, not the whole KB — when in "
        "doubt, search.\n\n"
        "**The KB tools are deferred.** To unlock them, call ToolSearch ONCE "
        "with this query (one call loads all four):\n\n"
        "```\n"
        f"ToolSearch(query=\"select:{kb_prefix}search_knowledge,"
        f"{kb_prefix}get_article,{kb_prefix}search_codebase,"
        f"{kb_prefix}search_raw_daily\", max_results=4)\n"
        "```\n\n"
        "Then call `search_knowledge(query=...)` (mode=\"hybrid\" by default). "
        "Fetch full articles with `get_article(slug)` only when a slim hit looks "
        "promising. Skip this whole flow only for trivial syntax questions, "
        "mechanical refactors, or when the codebase itself unambiguously answers."
    )

    # Code-intelligence directive — the structural twin of the KB block above.
    # Without an explicit "use first" nudge + unlock command, the code-intel
    # tools (also deferred) get skipped: the agent edits files blind to their
    # dependents, routes, and blast radius. A UserPromptSubmit hook auto-injects
    # context when the prompt names a concrete entity, but that only fires when
    # the user spells out a path/route/class — this block covers the rest.
    ci_prefix = f"mcp__{slug}-code-intel__"
    parts.append(
        "## Use Code Intelligence before touching code\n\n"
        "Before editing a file, tracing a request, or judging the blast radius "
        "of a change, query the `code-intel` graph instead of re-deriving "
        "structure by hand. These tools are also deferred — unlock them in ONE "
        "ToolSearch call:\n\n"
        "```\n"
        f"ToolSearch(query=\"select:{ci_prefix}get_file_deps,"
        f"{ci_prefix}trace_route,{ci_prefix}impact_of_change,"
        f"{ci_prefix}get_template_graph\", max_results=4)\n"
        "```\n\n"
        "Triggers:\n"
        "- **Before editing any PHP/Twig/JS file** → `get_file_deps(path)` "
        "(who depends on it, what it depends on, routes/templates it touches)\n"
        "- **Tracing a URL → handler → services** → `trace_route(method, path)`\n"
        "- **Before merging / after editing** → `impact_of_change(file=..., "
        "since_ref=\"main\")` (affected routes + Stimulus controllers, risk-scored)\n"
        "- **Before changing a Twig template** → `get_template_graph(template)` "
        "(inheritance chain, includes, Stimulus bindings)\n\n"
        "When the prompt names a concrete file/route/class, a hook may already "
        "have injected this context under \"Auto-fetched code intelligence\" — "
        "use it, and reach for the tools above only for what it didn't cover."
    )

    # Update notice — first thing after the date so the user (and the
    # agent) sees the upgrade prompt before diving into KB context.
    update_notice = get_update_notice()
    if update_notice:
        parts.append(update_notice)

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

    # Knowledge base index (the core retrieval mechanism). Cap separately so a
    # huge index never crowds out compiled-truth + daily log under MAX_CONTEXT_CHARS.
    if INDEX_FILE.exists():
        index_content = INDEX_FILE.read_text(encoding="utf-8")
        article_count = index_content.count("[[")
        if len(index_content) > MAX_INDEX_CHARS:
            truncated = index_content[:MAX_INDEX_CHARS]
            last_row = truncated.rfind("\n|")
            if last_row > 0:
                truncated = truncated[:last_row]
            index_content = (
                truncated
                + f"\n\n_…(index truncated — {article_count} articles total; use "
                "`search_knowledge` to query the full set)_"
            )
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
