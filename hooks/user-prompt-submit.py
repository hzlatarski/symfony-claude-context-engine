"""UserPromptSubmit hook — auto-inject code-intel context for code tasks.

This is the "per-task" context layer (the SessionStart hook is "per-session").
When the user's prompt names a concrete code entity — a file path, a Symfony
route, a PascalCase class, or a Stimulus controller — this hook resolves it and
runs the matching code-intel builder (file deps / route trace / template graph),
then injects the result as additional context. Claude gets "what depends on this
file before I edit it" *without having to decide to fetch it*.

Design constraints (in priority order):
  1. NEVER block or break a turn. Any failure → empty context, exit 0.
  2. Zero cost on conversational prompts. The expensive import (mcp_server, which
     parses the PHP/Twig/call graphs) only happens if regex matched an entity.
  3. Bounded latency. Cap resolved entities; the in-process cache is warmed once
     and reused across the few builder calls in this single hook process.

Wired in .claude/settings.json:
    "UserPromptSubmit": [{
        "matcher": "",
        "hooks": [{"type": "command",
                   "command": "cd .claude/memory-compiler && unset VIRTUAL_ENV && PATH=... uv run python hooks/user-prompt-submit.py",
                   "timeout": 12}]
    }]
"""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # .../memory-compiler
PROJECT_ROOT = ROOT.parent.parent                      # repo root (AiTutor)

# Caps — keep total work bounded so we never add more than ~1-2s on a code task.
MAX_FILES = 3
MAX_ROUTES = 2
MAX_SECTION_CHARS = 1_600
MAX_TOTAL_CHARS = 7_000

# Hook disable mechanism, mirroring the other memory-compiler hooks.
_disabled = os.environ.get("MEMORY_COMPILER_DISABLED_HOOKS", "").lower().split(",")
if "all" in _disabled or "user-prompt-submit" in _disabled:
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit", "additionalContext": ""}}))
    sys.exit(0)


def _emit(context: str) -> None:
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": context,
    }}))
    sys.exit(0)


# -----------------------------------------------------------------------------
# Entity extraction (cheap — pure regex, runs on every prompt)
# -----------------------------------------------------------------------------

# Symfony class suffixes worth resolving to a source file.
_CLASS_SUFFIXES = (
    "Controller", "Service", "Repository", "Subscriber", "Manager",
    "Builder", "Resolver", "Voter", "Handler", "Composer", "Provider",
    "Processor", "Factory", "Listener", "Command",
)

_PATH_RE = re.compile(r"[A-Za-z0-9_./\\-]+\.(?:php|twig|js)\b")
_ROUTE_RE = re.compile(r"\b(GET|POST|PUT|DELETE|PATCH)\s+(/[A-Za-z0-9/_{}.-]*)")
_CLASS_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]+(?:" + "|".join(_CLASS_SUFFIXES) + r"))\b"
)


def _glob_first(pattern: str) -> Path | None:
    """Return the first existing match for a glob under PROJECT_ROOT, or None."""
    try:
        for match in PROJECT_ROOT.glob(pattern):
            if match.is_file():
                return match
    except OSError:
        return None
    return None


def _to_rel(path: Path) -> str | None:
    """Repo-relative, forward-slashed path — the key the builders expect."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return None


def _resolve_files(prompt: str) -> list[str]:
    """Resolve file paths + class/controller names mentioned in the prompt to
    existing repo-relative paths. Order-preserving, deduplicated."""
    rels: list[str] = []
    seen: set[str] = set()

    def add(rel: str | None) -> None:
        if rel and rel not in seen:
            seen.add(rel)
            rels.append(rel)

    # 1. Explicit paths with a known extension.
    for raw in _PATH_RE.findall(prompt):
        token = raw.replace("\\", "/").lstrip("./")
        if token.endswith(".js") and not token.endswith("_controller.js"):
            continue  # only Stimulus controllers are in the JS graph
        candidate = PROJECT_ROOT / token
        if candidate.is_file():
            add(_to_rel(candidate))
            continue
        # Bare filename → glob it into place.
        name = Path(token).name
        if token.endswith(".php"):
            add(_to_rel(_glob_first(f"src/**/{name}") or Path("/nonexistent")))
        elif token.endswith(".twig"):
            add(_to_rel(_glob_first(f"templates/**/{name}") or Path("/nonexistent")))
        elif token.endswith("_controller.js"):
            add(_to_rel(_glob_first(f"assets/controllers/{name}") or Path("/nonexistent")))

    # 2. PascalCase Symfony classes → src/**/<Class>.php
    for cls in _CLASS_RE.findall(prompt):
        if len(rels) >= MAX_FILES:
            break
        hit = _glob_first(f"src/**/{cls}.php")
        if hit:
            add(_to_rel(hit))

    return rels[:MAX_FILES]


def _resolve_routes(prompt: str) -> list[tuple[str, str]]:
    routes: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for method, path in _ROUTE_RE.findall(prompt):
        key = (method.upper(), path)
        if key not in seen:
            seen.add(key)
            routes.append(key)
    return routes[:MAX_ROUTES]


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "\n…(truncated)"


def main() -> None:
    # 1. Read the prompt (stdin JSON). Failure → no-op.
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        prompt = payload.get("prompt", "") or ""
    except (json.JSONDecodeError, ValueError):
        _emit("")

    if not prompt.strip():
        _emit("")

    # 2. Cheap entity extraction. If nothing matched, exit before the costly
    #    import — conversational prompts pay only the regex cost.
    files = _resolve_files(prompt)
    routes = _resolve_routes(prompt)
    if not files and not routes:
        _emit("")

    # 3. Now (and only now) import the code-intel builders. Wrapped so an import
    #    or parse failure degrades to no injected context, never a broken turn.
    sections: list[str] = []
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from scripts.mcp_server import _build_file_deps, _build_trace_route

        _MISS_MARKERS = ("not found", "Unknown file type", "No route found")

        for rel in files:
            out = _build_file_deps(rel)
            if out and not any(m in out for m in _MISS_MARKERS):
                sections.append(f"### `{rel}`\n{_clip(out, MAX_SECTION_CHARS)}")

        for method, path in routes:
            out = _build_trace_route(method, path)
            if out and not any(m in out for m in _MISS_MARKERS):
                sections.append(
                    f"### Route trace: {method} {path}\n{_clip(out, MAX_SECTION_CHARS)}"
                )
    except Exception:
        _emit("")

    if not sections:
        _emit("")

    body = "\n\n".join(sections)
    header = (
        "## Auto-fetched code intelligence\n\n"
        "Your prompt named code below. This structure was pulled automatically "
        "from the `aitutor-code-intel` graph (dependencies, routes, call chains) "
        "so you don't have to re-derive it. Treat it as ground truth for *what "
        "connects to what*. For anything not shown — deeper call chains, "
        "blast-radius of a change, template inheritance — unlock and call the "
        "`aitutor-code-intel` MCP tools (`get_file_deps`, `trace_route`, "
        "`impact_of_change`, `get_template_graph`).\n\n"
    )
    _emit(_clip(header + body, MAX_TOTAL_CHARS))


if __name__ == "__main__":
    main()
