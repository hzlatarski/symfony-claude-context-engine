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
MAX_TOTAL_CHARS = 9_000
MAX_CODEBASE_HITS = 3
MAX_KB_HITS = 4

# Conceptual-question triggers. When the prompt has NO concrete code entity but
# DOES read like a why/decision/how question, we run a KB search so the curated
# knowledge base is surfaced automatically — the gap where the code-entity
# regexes match nothing and the agent would otherwise never see the KB.
#
# Stems use ``\w*`` so derivatives fire too (reason→reasons/reasoning,
# strateg→strategy/strategic, architect→architecture) — a plain trailing ``\b``
# silently killed every plural/derivative. Bare domain nouns (grading, scenario,
# persona, pricing) are deliberately NOT triggers: they're everyday nouns in
# this codebase and would fire the ~1.5s KB load on ordinary mechanical prompts
# ("improve the grading UI", "bump the persona image size"). We trigger on
# genuine question/decision language instead.
_CONCEPTUAL_RE = re.compile(
    r"\b(?:"
    r"why|reason\w*|rational\w*|purpose|"
    r"decision\w*|decid\w*|"
    r"architect\w*|approach\w*|trade-?off\w*|convention\w*|prefer\w*|strateg\w*|"
    r"how (?:does|do we|is|are|should|did)|"
    r"we (?:discussed|decided|agreed)|the decision|"
    r"what.?s the (?:point|purpose|reason)"
    r")\b",
    re.IGNORECASE,
)


def _looks_conceptual(prompt: str) -> bool:
    return bool(_CONCEPTUAL_RE.search(prompt))

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


# -----------------------------------------------------------------------------
# Section builders (each self-contained + failure-isolated — one broken section
# must never suppress the others or break the turn).
# -----------------------------------------------------------------------------

def _kb_section(prompt: str) -> str:
    """Top curated-KB matches for a conceptual prompt (hybrid BM25+vector).

    Cold cost measured at ~1.3s (import + query) — well inside the hook
    budget. Uses the same `_search_knowledge_impl` the MCP tool exposes.
    """
    try:
        from scripts.knowledge_mcp_server import _search_knowledge_impl
        hits = _search_knowledge_impl(query=prompt, limit=MAX_KB_HITS, mode="hybrid")
    except Exception:
        return ""
    if not hits:
        return ""
    lines = [
        "## Auto-fetched knowledge base",
        "",
        "Your prompt reads like a why/decision/how question. Top matches from the "
        "curated KB — call `get_article(slug)` for the full body, or "
        "`search_knowledge` for more/filtered results:",
        "",
    ]
    for h in hits:
        slug = h.get("slug")
        snippet = (h.get("snippet") or "").replace("\n", " ")
        md = h.get("metadata") or {}
        mtype = md.get("type")
        conf = md.get("confidence")
        meta = f" _(type={mtype}, conf={conf})_" if mtype else ""
        lines.append(f"- **{slug}**{meta} — {snippet}")
    return "\n".join(lines)


def _codebase_section(prompt: str, exclude_rels: set[str]) -> str:
    """Semantically-related code chunks not already named in the prompt.

    Exclusion and dedup are by **basename**: the codebase index can key the
    same file under both a repo-relative path and a bare filename (a Windows
    drive-letter-case fallback in index_codebase), so `src/Foo.php` and
    `Foo.php` are the same hit — comparing full paths would leak the named
    file back in and let duplicates crowd out the slots. Over-fetch generously
    since the prompt usually names the most-similar file itself.
    """
    exclude_names = {r.rsplit("/", 1)[-1] for r in exclude_rels}
    try:
        from scripts.knowledge_mcp_server import _search_codebase_impl
        hits = _search_codebase_impl(query=prompt, limit=MAX_CODEBASE_HITS * 2 + len(exclude_rels) + 2)
    except Exception:
        return ""
    rows = []
    seen_base: set[str] = set()
    for h in hits:
        rel = (h.get("path") or "").split(":")[0]
        # Skip indexed copies living in sibling worktrees / vendor — they're
        # duplicates of real src files and pure noise in a "related code" list.
        if ".worktrees/" in rel or rel.startswith("vendor/"):
            continue
        base = rel.rsplit("/", 1)[-1]
        if not base or base in exclude_names or base in seen_base:
            continue
        seen_base.add(base)
        rows.append(h)
        if len(rows) >= MAX_CODEBASE_HITS:
            break
    if not rows:
        return ""
    lines = [
        "## Auto-fetched related code",
        "",
        "Semantically-related code the dependency graph does not directly link:",
        "",
    ]
    for h in rows:
        sym = f" — {h['symbols']}" if h.get("symbols") else ""
        lines.append(f"- `{h.get('path')}`{sym}")
    return "\n".join(lines)


def main() -> None:
    # 1. Read the prompt (stdin JSON). Failure → no-op.
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        if not isinstance(payload, dict):
            _emit("")
        prompt = payload.get("prompt", "") or ""
    except (json.JSONDecodeError, ValueError):
        _emit("")

    if not prompt.strip():
        _emit("")

    # 2. Cheap entity extraction + conceptual sniff. If nothing at all matched,
    #    exit before any costly import — conversational prompts pay only regex.
    files = _resolve_files(prompt)
    routes = _resolve_routes(prompt)
    conceptual = _looks_conceptual(prompt)
    if not files and not routes and not conceptual:
        _emit("")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # 3. Code-intel sections (structure from the graph). Failure-isolated so a
    #    parse error here still lets the KB section below run.
    code_sections: list[str] = []
    if files or routes:
        try:
            from scripts.mcp_server import _build_file_deps, _build_trace_route

            _MISS_MARKERS = ("not found", "Unknown file type", "No route found")

            for rel in files:
                out = _build_file_deps(rel)
                if out and not any(m in out for m in _MISS_MARKERS):
                    code_sections.append(f"### `{rel}`\n{_clip(out, MAX_SECTION_CHARS)}")

            for method, path in routes:
                out = _build_trace_route(method, path)
                if out and not any(m in out for m in _MISS_MARKERS):
                    code_sections.append(
                        f"### Route trace: {method} {path}\n{_clip(out, MAX_SECTION_CHARS)}"
                    )
        except Exception:
            code_sections = code_sections  # keep whatever succeeded

    # 4. Retrieval sections (curated KB + semantically-related code). These load
    #    Chroma once; both share the import cost.
    kb_sections: list[str] = []
    if conceptual:
        s = _kb_section(prompt)
        if s:
            kb_sections.append(_clip(s, MAX_SECTION_CHARS))
    if files:
        s = _codebase_section(prompt, set(files))
        if s:
            kb_sections.append(_clip(s, MAX_SECTION_CHARS))

    all_sections = code_sections + kb_sections
    if not all_sections:
        _emit("")

    header = (
        "## Auto-fetched context\n\n"
        "Pulled automatically from the memory-compiler MCP servers so you don't "
        "re-derive it: `aitutor-code-intel` for *what connects to what* "
        "(dependencies, routes, call chains) and `aitutor-knowledge` for *why it "
        "was built this way* (curated articles). Treat the code structure as "
        "ground truth. For anything not shown — deeper chains, template "
        "inheritance, full articles — unlock and call the MCP tools "
        "(`get_file_deps`, `trace_route`, `impact_of_change`, `get_article`, "
        "`search_knowledge`).\n\n"
    )
    _emit(_clip(header + "\n\n".join(all_sections), MAX_TOTAL_CHARS))


if __name__ == "__main__":
    main()
