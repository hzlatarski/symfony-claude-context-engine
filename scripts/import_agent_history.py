"""Multi-agent conversation-history importer (ZERO LLM cost).

Reads other agents' transcript stores (Codex, other Claude Code projects),
normalizes each session to a markdown file, and writes it to a staging dir
that the existing ``ingest.py`` pipeline can later compile. No LLM calls,
no network, no Chroma writes — pure on-disk transcript mining.

The normalized model is :class:`AgentSession`. Per-agent parsing lives in
``scripts/agent_adapters/`` (the registry is the extension point for new
agents such as Hermes once its format is confirmed).

CLI::

    uv run python scripts/import_agent_history.py \
        --agent codex|claude|all [--project PATH] [--since YYYY-MM-DD] \
        [--out DIR] [--limit N] [--dry-run]

Defaults: ``--project`` = ``config.PROJECT_ROOT`` (only this project's
sessions); ``--project all`` disables the cwd filter. ``--out`` =
``config.KNOWLEDGE_DIR / "imported" / <agent>``. Writes are idempotent
(skip when the target file already holds identical content).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ── Normalized model ───────────────────────────────────────────────────

@dataclass
class AgentSession:
    """A single normalized agent conversation, ready for markdown rendering."""

    agent: str
    session_id: str
    cwd: str
    started_at: str  # ISO timestamp
    turns: list[dict] = field(default_factory=list)  # {ts, role, text}


# ── Import-regime shims ────────────────────────────────────────────────
# Mirror unified_graph.build_for_project's try/except dual pattern so this
# module works both as ``scripts.import_agent_history`` (tests / package) and
# as a top-level ``import_agent_history`` (direct CLI invocation).
def _load_config():
    try:
        from scripts import config  # type: ignore
    except ImportError:
        import config  # type: ignore
    return config


def _load_registry():
    try:
        from scripts import agent_adapters  # type: ignore
    except ImportError:
        import agent_adapters  # type: ignore
    return agent_adapters


# ── Pure helpers ───────────────────────────────────────────────────────

def _normalize_path(p: str) -> str:
    """Lowercased, forward-slash, no trailing slash — for path comparison."""
    norm = p.replace("\\", "/").strip().lower()
    while norm.endswith("/") and len(norm) > 1:
        norm = norm[:-1]
    return norm


def session_matches(session: "AgentSession", project, since) -> bool:
    """True if the session passes the project-cwd and since-date filters.

    ``project`` ``None`` disables the cwd filter; otherwise the session's
    ``cwd`` is compared to ``project`` after normalizing case, ``\\``->``/``
    and trailing slashes. ``since`` ``None`` disables the date filter;
    otherwise ``started_at[:10] >= since``.
    """
    if project is not None:
        if _normalize_path(session.cwd) != _normalize_path(project):
            return False
    if since is not None:
        started_day = (session.started_at or "")[:10]
        if not started_day or started_day < since:
            return False
    return True


def _short_id(session_id: str) -> str:
    sid = (session_id or "unknown").strip()
    # Keep it filesystem-friendly and short; codex ids can be long, claude
    # uses uuids — first 12 chars of the leading hex/uuid segment is plenty.
    sid = sid.split("/")[-1]
    return sid[:12] if sid else "unknown"


def _yaml_scalar(value: str) -> str:
    """Render a YAML scalar, quoting only when the value needs it.

    Simple tokens (ids, plain dates) stay unquoted for readability; values
    with YAML-significant characters (``:``, ``\\``, quotes, leading/trailing
    space, or YAML markers) are double-quoted with backslashes escaped.
    """
    if value == "":
        return '""'
    needs_quote = (
        any(ch in value for ch in ':#\\"\'')
        or value != value.strip()
        or value[0] in "[]{}>|&*!%@`"
    )
    if not needs_quote:
        return value
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_session_markdown(session: "AgentSession") -> str:
    """Render an AgentSession to ingestible markdown (frontmatter + body)."""
    date = (session.started_at or "")[:10] or "unknown"
    short = _short_id(session.session_id)
    title = f"{session.agent} session {short} ({date})"

    lines: list[str] = ["---"]
    lines.append(f"title: {_yaml_scalar(title)}")
    lines.append(f"source_agent: {session.agent}")
    lines.append(f"session_id: {_yaml_scalar(session.session_id or 'unknown')}")
    lines.append(f"cwd: {_yaml_scalar(session.cwd or '')}")
    lines.append(f"date: {date}")
    lines.append("type: event")
    lines.append("---")
    lines.append("")
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Transcript")
    lines.append("")

    for turn in session.turns:
        role = turn.get("role", "")
        label = "User" if role == "user" else "Assistant" if role == "assistant" else role.capitalize()
        text = (turn.get("text", "") or "").strip()
        lines.append(f"**{label}:**")
        lines.append("")
        lines.append(text)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def discover_sessions(store: Path, glob: str) -> list[Path]:
    """Return all transcript files under ``store`` matching ``glob`` (sorted)."""
    store = Path(store)
    if not store.exists():
        return []
    return sorted(p for p in store.glob(glob) if p.is_file())


# ── CLI plumbing ───────────────────────────────────────────────────────

def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []


def _output_name(session: "AgentSession") -> str:
    date = (session.started_at or "")[:10] or "unknown"
    return f"{date}-{_short_id(session.session_id)}.md"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _import_for_agent(agent_name, registry, config, project, since, out_dir, limit, dry_run):
    adapter = registry.get_adapter(agent_name)
    store = adapter.default_store()
    files = discover_sessions(store, adapter.glob)

    created = 0
    skipped = 0
    processed = 0

    out_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        if limit is not None and processed >= limit:
            break
        lines = _read_lines(path)
        if not lines:
            continue
        session = adapter.parse(lines)
        if not session.turns:
            continue
        if not session_matches(session, project, since):
            continue

        processed += 1
        markdown = render_session_markdown(session)
        target = out_dir / _output_name(session)

        if target.exists():
            existing = target.read_text(encoding="utf-8")
            if _content_hash(existing) == _content_hash(markdown):
                skipped += 1
                continue

        if dry_run:
            print(f"[dry-run] would write {target}")
            created += 1
            continue

        target.write_text(markdown, encoding="utf-8")
        created += 1

    return created, skipped


def main(argv=None) -> int:
    config = _load_config()
    registry = _load_registry()

    parser = argparse.ArgumentParser(
        description="Import other agents' conversation history into the KB staging dir."
    )
    parser.add_argument(
        "--agent",
        required=True,
        choices=sorted(registry.available_agents()) + ["all"],
        help="Which agent's history to import, or 'all'.",
    )
    parser.add_argument(
        "--project",
        default=str(config.PROJECT_ROOT),
        help="Only import sessions whose cwd matches this path. "
             "Pass the literal 'all' to disable the cwd filter.",
    )
    parser.add_argument("--since", default=None, help="Only sessions on/after YYYY-MM-DD.")
    parser.add_argument("--out", default=None, help="Output staging dir (default: knowledge/imported/<agent>).")
    parser.add_argument("--limit", type=int, default=None, help="Max sessions per agent.")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing.")
    args = parser.parse_args(argv)

    project = None if args.project == "all" else args.project

    if args.agent == "all":
        agents = registry.available_agents()
    else:
        agents = [args.agent]

    total_created = 0
    total_skipped = 0
    for agent_name in agents:
        if args.out is not None:
            out_dir = Path(args.out)
        else:
            out_dir = config.KNOWLEDGE_DIR / "imported" / agent_name
        created, skipped = _import_for_agent(
            agent_name, registry, config, project, args.since, out_dir,
            args.limit, args.dry_run,
        )
        total_created += created
        total_skipped += skipped
        print(f"{agent_name}: created={created} skipped={skipped} (out={out_dir})")

    print(f"TOTAL: created={total_created} skipped={total_skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
