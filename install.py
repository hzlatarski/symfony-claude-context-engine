#!/usr/bin/env python3
"""
One-command installer for Claude Context Engine — Symfony Edition.

Usage (from inside the cloned repo):
    uv run python install.py

Full one-liner clone + install:
    git clone https://github.com/hzlatarski/symfony-claude-context-engine.git .claude/memory-compiler
    uv run --directory .claude/memory-compiler python install.py

What it does:
    1. Merges Claude Code hooks  →  .claude/settings.json
    2. Registers MCP servers     →  ~/.claude.json  (User scope, per-project slug)
    3. Copies sources.yaml.example → sources.yaml  (if not already present)
    4. Asks for ANTHROPIC_API_KEY  →  writes to .env.local  (if missing)
    5. Asks about memory symlink   →  links .claude/memory/ to your Claude memory dir
    6. Runs initial ingest + ChromaDB vector reindex
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ── Path anchors ──────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent      # .claude/memory-compiler/
CLAUDE_DIR = HERE.parent                    # .claude/
PROJECT_ROOT = HERE.parent.parent           # your Symfony project root

# ── Hook config ───────────────────────────────────────────────────────────────
HOOKS: dict[str, list] = {
    "SessionStart": [{"matcher": "", "hooks": [{"type": "command", "command": "cd .claude/memory-compiler && uv run python hooks/session-start.py", "timeout": 15}]}],
    "PreCompact":   [{"matcher": "", "hooks": [{"type": "command", "command": "cd .claude/memory-compiler && uv run python hooks/pre-compact.py",    "timeout": 10}]}],
    "SessionEnd":   [{"matcher": "", "hooks": [{"type": "command", "command": "cd .claude/memory-compiler && uv run python hooks/session-end.py",    "timeout": 10}]}],
    "PostToolUse":  [{"matcher": "", "hooks": [{"type": "command", "command": "cd .claude/memory-compiler && uv run python hooks/post-tool-use.py",  "timeout": 5 }]}],
}

# ── MCP server config ─────────────────────────────────────────────────────────
# The VS Code Claude Code extension reads user-scope MCP servers from
# ~/.claude.json (top-level "mcpServers" object) — NOT project-root .mcp.json,
# which is a CLI-only convention. So the installer registers per-project
# entries there using absolute paths + a project slug derived from the folder
# name. Example keys: aitutor-code-intel, aitutor-knowledge.
_MCP_SUFFIXES: dict[str, str] = {
    "code-intel": "scripts/mcp_server.py",            # Symfony parser surface
    "knowledge":  "scripts/knowledge_mcp_server.py",  # Knowledge retrieval surface
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _h1(msg: str) -> None:
    print(f"\n\033[1;34m▶ {msg}\033[0m")

def _ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m {msg}")

def _skip(msg: str) -> None:
    print(f"  \033[33m–\033[0m {msg}  (already present — skipped)")

def _warn(msg: str) -> None:
    print(f"  \033[33m!\033[0m {msg}")

def _fail(msg: str) -> None:
    print(f"  \033[31m✗\033[0m {msg}")

def _ask(prompt: str, default: str = "") -> str:
    """Prompt user for input; return default on empty enter."""
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"  \033[36m?\033[0m {prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return answer if answer else default

def _confirm(prompt: str, default: bool = True) -> bool:
    """Ask a yes/no question."""
    suffix = " [Y/n]" if default else " [y/N]"
    try:
        answer = input(f"  \033[36m?\033[0m {prompt}{suffix}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if answer in ("y", "yes"):
        return True
    if answer in ("n", "no"):
        return False
    return default


# ── Step 1: merge hooks ───────────────────────────────────────────────────────
def merge_settings_json() -> None:
    _h1("Merging hooks → .claude/settings.json")
    path = CLAUDE_DIR / "settings.json"

    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            _warn("settings.json exists but is not valid JSON — starting fresh")

    hooks = data.setdefault("hooks", {})
    added = 0
    for event, entries in HOOKS.items():
        bucket = hooks.setdefault(event, [])
        our_cmd: str = entries[0]["hooks"][0]["command"]
        already_there = any(
            h.get("hooks", [{}])[0].get("command", "") == our_cmd
            for h in bucket
        )
        if not already_there:
            bucket.extend(entries)
            added += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    if added:
        _ok(f"Added {added} hook(s) to {path.relative_to(PROJECT_ROOT)}")
    else:
        _skip(f"Hooks already present in {path.relative_to(PROJECT_ROOT)}")


# ── Step 2: register MCP servers at User scope ───────────────────────────────
def merge_mcp_json() -> None:
    _h1("Registering MCP servers → ~/.claude.json (User scope)")

    claude_json = Path.home() / ".claude.json"
    if not claude_json.exists():
        _fail(f"{claude_json} not found — open Claude Code once to initialize it, then re-run.")
        return

    slug = _project_slug(PROJECT_ROOT.name)
    uv_exe = _find_uv()
    if uv_exe is None:
        _fail("Could not find `uv` on PATH — install uv (https://docs.astral.sh/uv/) and re-run.")
        return

    mc_abs = str(HERE).replace("\\", "/")   # absolute path to .claude/memory-compiler
    entries: dict[str, dict] = {
        f"{slug}-{suffix}": {
            "command": uv_exe,
            "args": ["run", "--directory", mc_abs, "python", script],
        }
        for suffix, script in _MCP_SUFFIXES.items()
    }

    # Parse + mutate + atomic write: protects a Claude Code session that may
    # be reading this file while we write.
    try:
        data = json.loads(claude_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _fail(f"{claude_json} is not valid JSON — aborting to avoid corrupting user config.")
        return

    # Timestamped backup so a bad edit can always be rolled back.
    backup = claude_json.with_suffix(
        f".json.bak-install-{_timestamp()}"
    )
    backup.write_text(claude_json.read_text(encoding="utf-8"), encoding="utf-8")

    servers = data.setdefault("mcpServers", {})
    added, updated = 0, 0
    for key, config in entries.items():
        if key in servers:
            if servers[key] != config:
                servers[key] = config
                updated += 1
        else:
            servers[key] = config
            added += 1

    tmp = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=claude_json.parent, suffix=".tmp",
    )
    json.dump(data, tmp, indent=2)
    tmp.flush(); os.fsync(tmp.fileno()); tmp.close()
    os.replace(tmp.name, claude_json)

    noun = ", ".join(entries.keys())
    if added and updated:
        _ok(f"Added {added} + updated {updated} server(s) in ~/.claude.json ({noun})")
    elif added:
        _ok(f"Added {added} server(s) to ~/.claude.json ({noun})")
    elif updated:
        _ok(f"Updated {updated} server(s) in ~/.claude.json ({noun})")
    else:
        _skip(f"Servers already registered in ~/.claude.json ({noun})")

    _warn(f"Backup saved: {backup}")

    # Warn about legacy configs that earlier installer versions wrote to.
    for legacy in (PROJECT_ROOT / ".mcp.json", CLAUDE_DIR / ".mcp.json"):
        if not legacy.exists():
            continue
        try:
            legacy_data = json.loads(legacy.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        legacy_servers = legacy_data.get("mcpServers") or {}
        ours_in_legacy = [k for k in legacy_servers if k.endswith("-code-intel") or k.endswith("-knowledge")
                          or k in ("symfony-code-intel", "knowledge-compiler",
                                   "memory-compiler-intel", "memory-compiler-knowledge")]
        if ours_in_legacy:
            _warn(
                f"Legacy MCP entries found in {legacy} ({', '.join(ours_in_legacy)}) — "
                "the VS Code extension ignores this file. Safe to remove these entries."
            )


def _project_slug(name: str) -> str:
    """AiTutor → aitutor, My_Project → my-project, mixed123 → mixed123."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "project"


def _find_uv() -> str | None:
    """Resolve absolute path to uv/uv.exe; fall back to bare 'uv' if on PATH."""
    for candidate in ("uv", "uv.exe"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _timestamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# ── Step 2b: patch CLAUDE.md ──────────────────────────────────────────────────
# Sentinel tags used to bracket the managed block so re-running install is safe.
_CLAUDE_MD_SENTINEL_START = "<!-- memory-compiler-mcp-start -->"
_CLAUDE_MD_SENTINEL_END   = "<!-- memory-compiler-mcp-end -->"

_CLAUDE_MD_MCP_BLOCK = """\
<!-- memory-compiler-mcp-start -->
### Code Intelligence MCP (`symfony-code-intel`)

Exposes 6 tools for on-demand code structure queries — mtime-cached, never needs a rebuild. Tools appear in the session as `mcp__symfony-code-intel__<tool>`:

| Tool | When to call it |
|---|---|
| `get_codebase_overview()` | Start of any architecture discussion or when you need to orient in an unfamiliar area |
| `get_file_deps(path)` | Before editing a PHP/Twig/JS file — reveals what depends on it and what it depends on |
| `get_route_map(prefix)` | When tracing how a URL maps to a controller, or finding which routes touch a service |
| `get_template_graph(template)` | Before changing a Twig file — shows inheritance chain, all includes, and Stimulus bindings |
| `get_stimulus_map(controller)` | When adding or removing a Stimulus controller — finds every template that references it |
| `get_hotspots(top_n)` | Before a refactor — surfaces high-churn files so you avoid merging into a hot area |

**Skip calling these** when you are making a trivial one-file change with no dependencies, or when the user has already confirmed the scope explicitly.

### Knowledge Base MCP (`knowledge-compiler`)

Two-tier retrieval — **search first, fetch only what you need**. Tools appear in the session as `mcp__knowledge-compiler__<tool>`:

| Tool | When to call it |
|---|---|
| `search_knowledge(query, ...)` | **Always call this first** when you need context about a product decision, architecture choice, past finding, or behavioural scenario. Use `mode="hybrid"` (default). Use `mode="bm25"` for exact identifiers. |
| `get_article(slug)` / `get_articles([slugs])` | After `search_knowledge` returns promising slim hits — fetch only the slugs worth reading in full |
| `search_raw_daily(query, ...)` | When `search_knowledge` gives weak results and you need verbatim session material to verify a claim |
| `search_codebase(query, file_type)` | When you need to find *where* a concept is implemented in code — returns chunked file excerpts with line ranges. Complements `get_file_deps`. |
| `list_contradictions()` | When a knowledge article seems inconsistent — check if it is already quarantined before acting on it |

**`search_knowledge` filter tips:**
- `type_filter`: `fact` | `event` | `discovery` | `preference` | `advice` | `decision`
- `zone_filter`: `observed` (low hallucination risk) | `synthesized` (compiler inferences — verify before trusting)
- `min_confidence`: use `0.7` when you need firm answers; omit when exploring

**Call `search_knowledge` before:**
- Implementing a feature that touches product behaviour, grading logic, or AI prompts
- Answering a question about why something was built a certain way
- Writing copy or setting strategy — check `preference` and `decision` type articles first

**Do NOT call it** for pure syntax questions, refactoring mechanical code, or anything the codebase itself answers unambiguously.

The session-start hook injects a compact summary (file counts + top hotspots + available MCP tools) so you always know the codebase shape without burning tokens on full dumps.
<!-- memory-compiler-mcp-end -->"""


def patch_claude_md() -> None:
    """Idempotently insert the MCP usage section into CLAUDE.md.

    Strategy:
    - If the sentinel block already exists, skip (idempotent).
    - If a legacy "### Code Intelligence (MCP)" section heading exists
      (older installs), replace that paragraph with the new sentinel block.
    - Otherwise append the block at the end of the file.
    """
    _h1("Patching CLAUDE.md with MCP usage instructions")

    # Try .claude/CLAUDE.md first, then project root CLAUDE.md
    candidates = [CLAUDE_DIR / "CLAUDE.md", PROJECT_ROOT / "CLAUDE.md"]
    path = next((p for p in candidates if p.exists()), None)

    if path is None:
        _warn("No CLAUDE.md found — creating .claude/CLAUDE.md with MCP block")
        path = CLAUDE_DIR / "CLAUDE.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# CLAUDE.md\n\nThis file provides guidance to Claude Code when working with code in this repository.\n\n"
            + _CLAUDE_MD_MCP_BLOCK + "\n",
            encoding="utf-8",
        )
        _ok(f"Created {path.relative_to(PROJECT_ROOT)}")
        return

    content = path.read_text(encoding="utf-8")

    # Already patched — sentinel present
    if _CLAUDE_MD_SENTINEL_START in content:
        _skip(f"MCP block already present in {path.relative_to(PROJECT_ROOT)}")
        return

    # Replace legacy heading block if it exists
    import re
    legacy = re.compile(
        r"### Code Intelligence \(MCP\)\n.*?(?=\n## |\n### |\Z)",
        re.DOTALL,
    )
    if legacy.search(content):
        new_content = legacy.sub(_CLAUDE_MD_MCP_BLOCK, content)
        path.write_text(new_content, encoding="utf-8")
        _ok(f"Replaced legacy MCP section in {path.relative_to(PROJECT_ROOT)}")
        return

    # Append before the first top-level `## Code Style` section if present,
    # otherwise just append at end of file.
    insert_marker = "\n## Code Style"
    if insert_marker in content:
        new_content = content.replace(
            insert_marker,
            "\n" + _CLAUDE_MD_MCP_BLOCK + "\n" + insert_marker,
            1,
        )
    else:
        separator = "\n" if content.endswith("\n") else "\n\n"
        new_content = content + separator + _CLAUDE_MD_MCP_BLOCK + "\n"

    path.write_text(new_content, encoding="utf-8")
    _ok(f"Inserted MCP block into {path.relative_to(PROJECT_ROOT)}")


# ── Step 3: sources.yaml ──────────────────────────────────────────────────────
def copy_sources_yaml() -> None:
    _h1("Setting up sources.yaml")
    dst = HERE / "sources.yaml"
    src = HERE / "sources.yaml.example"

    if dst.exists():
        _skip("sources.yaml")
        return

    if not src.exists():
        _warn("sources.yaml.example not found — skipping")
        return

    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    _ok("Copied sources.yaml.example → sources.yaml")
    _warn("Edit sources.yaml to point at your project's docs and specs")


# ── Step 4: Anthropic API key ─────────────────────────────────────────────────
def setup_api_key() -> None:
    _h1("Anthropic API key")

    # Already in the environment (e.g. set system-wide)
    if os.environ.get("ANTHROPIC_API_KEY"):
        _skip("ANTHROPIC_API_KEY already set in environment")
        return

    env_local = PROJECT_ROOT / ".env.local"

    # Already in .env.local
    if env_local.exists():
        content = env_local.read_text(encoding="utf-8")
        if "ANTHROPIC_API_KEY" in content:
            _skip("ANTHROPIC_API_KEY already present in .env.local")
            return

    print("  The compile and flush pipelines require an Anthropic API key.")
    print("  Get one at https://console.anthropic.com/settings/keys")
    key = _ask("Paste your ANTHROPIC_API_KEY (leave blank to skip)")

    if not key:
        _warn("Skipped — you will need to set ANTHROPIC_API_KEY manually before running compile.py")
        return

    # Append to .env.local (or create it)
    existing = env_local.read_text(encoding="utf-8") if env_local.exists() else ""
    separator = "\n" if existing and not existing.endswith("\n") else ""
    env_local.write_text(existing + separator + f"ANTHROPIC_API_KEY={key}\n", encoding="utf-8")
    _ok(f"Written to {env_local.relative_to(PROJECT_ROOT)}")


# ── Step 5: memory symlink ────────────────────────────────────────────────────
def setup_memory_symlink() -> None:
    _h1("Claude persistent memory symlink")

    memory_link = CLAUDE_DIR / "memory"

    if memory_link.exists() or memory_link.is_symlink():
        _skip(".claude/memory already exists")
        return

    print("  The 'captured-memory' source in sources.yaml reads from .claude/memory/")
    print("  This folder holds auto-memory Claude Code writes between sessions.")
    print()

    # Compute the default path
    home = Path.home()
    # Claude Code stores project memory at ~/.claude/projects/<slug>/memory/
    # Try to guess the slug from the project root path
    slug = PROJECT_ROOT.as_posix().replace("/", "-").lstrip("-")
    default_memory = home / ".claude" / "projects" / slug / "memory"

    print(f"  Default location: {default_memory}")
    if not default_memory.exists():
        # Try just the directory name as slug
        name_slug = PROJECT_ROOT.name
        alt = home / ".claude" / "projects" / f"c--wamp64-www-{name_slug}" / "memory"
        if alt.exists():
            default_memory = alt
        else:
            print("  (Directory not found yet — Claude Code creates it on first session)")

    if not _confirm("Create .claude/memory → your Claude memory folder?"):
        _warn("Skipped — captured-memory source will not load until the symlink is created")
        return

    target_str = _ask("Path to your Claude memory folder", str(default_memory))
    target = Path(target_str).expanduser().resolve()

    if not target.exists():
        create = _confirm(f"  {target} does not exist. Create it?", default=True)
        if create:
            target.mkdir(parents=True, exist_ok=True)
            _ok(f"Created {target}")
        else:
            _warn("Skipped — symlink not created")
            return

    try:
        if platform.system() == "Windows":
            # Use mklink /J (directory junction — works without admin on Windows 10+)
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(memory_link), str(target)],
                check=True, capture_output=True,
            )
        else:
            memory_link.symlink_to(target)
        _ok(f"Linked .claude/memory → {target}")
    except Exception as exc:
        _fail(f"Could not create symlink: {exc}")
        _warn(f"Create it manually:  mklink /J .claude\\memory \"{target}\"  (Windows)")
        _warn(f"or:  ln -s \"{target}\" .claude/memory  (Mac/Linux)")


# ── Step 6: ingest + reindex ──────────────────────────────────────────────────
def run_ingest_and_reindex() -> None:
    _h1("Running initial ingest + ChromaDB vector reindex")
    uv_prefix = ["uv", "run", "--directory", str(HERE)]

    print("  ingest.py …")
    r = subprocess.run([*uv_prefix, "python", "scripts/ingest.py"], cwd=HERE)
    if r.returncode != 0:
        _fail("ingest.py failed — check output above")
        sys.exit(1)
    _ok("ingest.py done")

    print("  reindex.py --all …")
    r = subprocess.run([*uv_prefix, "python", "scripts/reindex.py", "--all"], cwd=HERE)
    if r.returncode != 0:
        _fail("reindex.py failed — check output above")
        sys.exit(1)
    _ok("reindex.py done (ChromaDB vector index built)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("\033[1m")
    print("╔════════════════════════════════════════════════╗")
    print("║   Claude Context Engine — Symfony Edition      ║")
    print("║   Setup                                        ║")
    print("╚════════════════════════════════════════════════╝")
    print("\033[0m", end="")
    print(f"  Project root  :  {PROJECT_ROOT}")
    print(f"  Engine path   :  {HERE}")

    merge_settings_json()
    merge_mcp_json()
    patch_claude_md()
    copy_sources_yaml()
    setup_api_key()
    setup_memory_symlink()
    run_ingest_and_reindex()

    print("\n\033[1;32m✅  Setup complete!\033[0m\n")
    print("Next steps:")
    print("  1. Edit  .claude/memory-compiler/sources.yaml  to point at your docs")
    print("  2. Re-run ingest after editing:")
    print("       uv run --directory .claude/memory-compiler python scripts/ingest.py")
    print("  3. Start the knowledge dashboard:")
    print("       uv run --directory .claude/memory-compiler python scripts/viewer.py")
    print("       → http://127.0.0.1:37778")
    print("  4. Open Claude Code — hooks fire automatically on next session\n")


if __name__ == "__main__":
    main()
