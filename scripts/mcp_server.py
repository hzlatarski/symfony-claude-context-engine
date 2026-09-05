"""Symfony Code Intelligence MCP Server.

Exposes 6 tools via FastMCP (stdio transport) for on-demand code queries:

    get_codebase_overview() -> file counts, hotspots, module map
    get_file_deps(path)     -> imports, reverse deps, routes, templates
    get_route_map(prefix)   -> route -> controller -> service table
    get_template_graph(t)   -> inheritance, includes, Stimulus bindings
    get_stimulus_map(name)  -> controller <-> template links
    get_hotspots(top_n)     -> churn-ranked files with ownership

Parsers are cached in-memory with mtime-based invalidation. Git intel
caches to knowledge/git-intel.json (HEAD-based invalidation).
"""
from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path

# Bootstrap: when Python runs this file directly (as Claude Code does via
# `python scripts/mcp_server.py`), only scripts/ is on sys.path, so
# `from scripts.parsers import ...` fails. Pytest gets the right path via
# pyproject.toml's `pythonpath = ["."]`, but direct execution doesn't.
# Manually add the memory-compiler root to sys.path so imports resolve
# regardless of how this script was invoked. The regression test for this
# lives at tests/test_mcp_server.py::test_mcp_server_launches_without_import_error
_HERE = Path(__file__).resolve().parent  # .../memory-compiler/scripts
_MEMORY_COMPILER_ROOT = _HERE.parent     # .../memory-compiler
if str(_MEMORY_COMPILER_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEMORY_COMPILER_ROOT))

from scripts.parsers import PROJECT_ROOT, php_graph, route_map, twig_graph, stimulus_map, git_intel, call_graph, messenger_map
from scripts import unified_graph
from scripts import entities
from scripts import parent_watchdog, mermaid_render

log = logging.getLogger("mcp_server")


# -----------------------------------------------------------------------------
# Cache layer
# -----------------------------------------------------------------------------


class ParseCache:
    """In-memory parser cache invalidated by file mtime changes.

    Each parser has a scan-root + glob pattern. We take max(mtime) over all
    matched files and compare to the stored mtime. Cheap (~100ms for
    hundreds of files) since os.stat is fast.
    """

    def __init__(self) -> None:
        self._php_cache: dict | None = None
        self._php_mtime: float = 0.0
        self._route_cache: dict | None = None
        self._route_mtime: float = 0.0
        self._twig_cache: dict | None = None
        self._twig_mtime: float = 0.0
        self._stim_cache: dict | None = None
        self._stim_mtime: float = 0.0
        self._call_graph_cache: dict | None = None
        self._call_graph_mtime: float = 0.0
        self._messenger_cache: dict | None = None
        self._messenger_mtime: float = 0.0
        self._unified_graph_cache: dict | None = None
        self._unified_graph_signature: tuple = ()

    @staticmethod
    def _max_mtime(paths) -> float:
        try:
            return max((p.stat().st_mtime for p in paths), default=0.0)
        except OSError:
            return 0.0

    def get_php_graph(self) -> dict:
        current = self._max_mtime((PROJECT_ROOT / "src").rglob("*.php"))
        if self._php_cache is None or current != self._php_mtime:
            log.info("Rebuilding PHP graph cache (mtime %s -> %s)", self._php_mtime, current)
            self._php_cache = php_graph.parse(PROJECT_ROOT)
            self._php_mtime = current
        return self._php_cache

    def get_route_map(self) -> dict:
        current = self._max_mtime((PROJECT_ROOT / "src" / "Controller").rglob("*.php"))
        if self._route_cache is None or current != self._route_mtime:
            log.info("Rebuilding route map cache")
            self._route_cache = route_map.parse(PROJECT_ROOT)
            self._route_mtime = current
        return self._route_cache

    def get_twig_graph(self) -> dict:
        current = self._max_mtime((PROJECT_ROOT / "templates").rglob("*.twig"))
        if self._twig_cache is None or current != self._twig_mtime:
            log.info("Rebuilding Twig graph cache")
            self._twig_cache = twig_graph.parse(PROJECT_ROOT)
            self._twig_mtime = current
        return self._twig_cache

    def get_stimulus_map(self) -> dict:
        stim_mtime = self._max_mtime((PROJECT_ROOT / "assets" / "controllers").glob("*_controller.js"))
        twig_mtime = self._max_mtime((PROJECT_ROOT / "templates").rglob("*.twig"))
        current = max(stim_mtime, twig_mtime)
        if self._stim_cache is None or current != self._stim_mtime:
            log.info("Rebuilding Stimulus map cache")
            self._stim_cache = stimulus_map.parse(PROJECT_ROOT)
            self._stim_mtime = current
        return self._stim_cache

    def get_git_intel(self) -> dict:
        return git_intel.load_or_parse(PROJECT_ROOT)

    def get_messenger_map(self) -> dict:
        current = self._max_mtime((PROJECT_ROOT / "src").rglob("*.php"))
        if self._messenger_cache is None or current != self._messenger_mtime:
            log.info("Rebuilding messenger map cache")
            self._messenger_cache = messenger_map.parse(PROJECT_ROOT)
            self._messenger_mtime = current
        return self._messenger_cache

    def get_call_graph(self) -> dict:
        # Both PHP and Stimulus JS files affect the graph — invalidate on either.
        php_mtime = self._max_mtime((PROJECT_ROOT / "src").rglob("*.php"))
        js_mtime = self._max_mtime((PROJECT_ROOT / "assets" / "controllers").rglob("*_controller.js"))
        current = max(php_mtime, js_mtime)
        if self._call_graph_cache is None or current != self._call_graph_mtime:
            log.info("Rebuilding call graph cache")
            graph = call_graph.parse(PROJECT_ROOT)
            # Resolve JS fetch placeholders to PHP controller symbols using the
            # current route map. Must run inside the cache so trace_route /
            # impact_of_change consumers see crossed-boundary edges.
            call_graph.resolve_fetch_edges(graph, self.get_route_map())
            self._call_graph_cache = graph
            self._call_graph_mtime = current
        return self._call_graph_cache

    def get_unified_graph(self) -> dict:
        """Cache the unified knowledge graph.

        Invalidates when either the call graph or any article markdown file
        in ``knowledge/{concepts,connections,qa}/`` changes. Signature is
        ``(call_graph_mtime, articles_mtime)`` — recomputing the unified
        graph is cheap (single linear pass over articles + dict copy of the
        call graph) so we don't need a finer-grained dirty-set protocol.
        """
        from scripts.config import KNOWLEDGE_DIR

        call_graph = self.get_call_graph()  # ensures cache + freshness
        article_mtime = 0.0
        for sub in ("concepts", "connections", "qa"):
            root = KNOWLEDGE_DIR / sub
            if root.exists():
                article_mtime = max(article_mtime, self._max_mtime(root.glob("*.md")))

        signature = (self._call_graph_mtime, article_mtime)
        if self._unified_graph_cache is None or signature != self._unified_graph_signature:
            log.info("Rebuilding unified knowledge graph cache")
            self._unified_graph_cache = unified_graph.build(
                call_graph=call_graph,
                knowledge_root=KNOWLEDGE_DIR,
            )
            self._unified_graph_signature = signature
        return self._unified_graph_cache


_cache = ParseCache()


# -----------------------------------------------------------------------------
# Tool implementations (pure Python, testable without MCP stack)
# -----------------------------------------------------------------------------


def _build_codebase_overview() -> str:
    php = _cache.get_php_graph()
    routes = _cache.get_route_map()
    twig = _cache.get_twig_graph()
    stim = _cache.get_stimulus_map()
    git = _cache.get_git_intel()

    lines = [
        "# Codebase Overview",
        "",
        "## PHP",
        f"- Total files: {php['stats']['total_files']}",
        f"- By type: {php['stats']['by_type']}",
        "",
        "## Routes",
        f"- Total: {routes['stats']['total_routes']}",
        f"- By prefix: {routes['stats']['by_prefix']}",
        "",
        "## Templates",
        f"- Total Twig files: {twig['stats']['total_templates']}",
        f"- Inheritance chains: {len(twig['inheritance_chains'])}",
        "",
        "## Stimulus",
        f"- Total controllers: {stim['stats']['total_controllers']}",
        f"- Total template usages: {stim['stats']['total_usages']}",
        f"- Orphan controllers: {stim['stats']['orphan_count']}",
        f"- Missing (referenced but no JS file): {stim['stats']['missing_count']}",
        "",
        "## Git Hotspots (top 10)",
    ]
    for h in git.get("hotspots", [])[:10]:
        lines.append(
            f"- {h['file']}: score={h['score']} "
            f"commits={h['commits_total']} owner={h['primary_owner']}"
        )

    unified = _cache.get_unified_graph()
    article_nodes = [n for n in unified["nodes"] if n.startswith("article:")]
    file_nodes = [n for n in unified["nodes"] if n.startswith("file:")]
    wikilink_edges = [e for e in unified["edges"] if e["kind"] == "wikilink"]
    cites_edges = [e for e in unified["edges"] if e["kind"] == "cites"]
    referenced = {e["from"] for e in unified["edges"]} | {e["to"] for e in unified["edges"]}
    orphan_articles = [n for n in article_nodes if n not in referenced]

    lines.extend([
        "",
        "## Unified Graph",
        f"- Articles: {len(article_nodes)}",
        f"- Files: {len(file_nodes)}",
        f"- Wikilink edges: {len(wikilink_edges)}",
        f"- Cites edges: {len(cites_edges)}",
        f"- Orphan articles (no in/out edges): {len(orphan_articles)}",
    ])
    return "\n".join(lines)


def _build_file_deps(file_path: str) -> str:
    """Return markdown describing dependencies for a given file.

    Handles PHP files, Twig templates, and Stimulus controller JS files.
    """
    # Normalise
    file_path = file_path.replace("\\", "/").lstrip("./")

    # Dispatch by extension
    if file_path.endswith(".php"):
        return _file_deps_php(file_path)
    if file_path.endswith(".twig"):
        return _file_deps_twig(file_path)
    if file_path.endswith("_controller.js"):
        return _file_deps_stimulus(file_path)

    return f"Unknown file type: {file_path}"


def _file_deps_php(file_path: str) -> str:
    php = _cache.get_php_graph()
    node = php["nodes"].get(file_path)
    if not node:
        return f"File not found in PHP graph: {file_path}"

    routes = _cache.get_route_map()
    git = _cache.get_git_intel()

    lines = [
        f"# {file_path}",
        f"- Type: **{node['type']}**",
        f"- Class: `{node['class']}`",
        f"- Namespace: `{node['namespace']}`",
        f"- In-degree (depended on by): {node['in_degree']}",
        f"- Out-degree (depends on): {node['out_degree']}",
        "",
        "## Imports (App\\... only)",
    ]
    for imp in node["imports"]:
        lines.append(f"- `{imp}`")

    # Reverse deps
    reverse = [e["from"] for e in php["edges"] if e["to"] == file_path]
    if reverse:
        lines.append("")
        lines.append("## Imported By")
        for r in reverse[:30]:
            lines.append(f"- {r}")

    # If it's a controller, show routes it handles
    if node["type"] == "controller":
        controller_routes = [
            (p, r) for p, r in routes["routes"].items() if r["file"] == file_path
        ]
        if controller_routes:
            lines.append("")
            lines.append("## Routes Handled")
            for p, r in controller_routes:
                lines.append(f"- `{p}` ({','.join(r['methods'])}) -> `{r['action']}()`"
                             + (f" -> `{r['template']}`" if r['template'] else ""))

    # Git intel: hotspot info + co-change partners
    hotspot = next((h for h in git.get("hotspots", []) if h["file"] == file_path), None)
    if hotspot:
        lines.append("")
        lines.append("## Git Intelligence")
        lines.append(f"- Commits: {hotspot['commits_total']} (30d: {hotspot['commits_30d']})")
        lines.append(f"- Hotspot score: {hotspot['score']}")
        lines.append(f"- Primary owner: {hotspot['primary_owner']}")
        if hotspot["co_change_partners"]:
            lines.append("- Co-change partners:")
            for p in hotspot["co_change_partners"][:5]:
                lines.append(f"  - {p['file']} (score={p['score']})")

    # Inline rationale (WHY / HACK / TODO / @deprecated). Parse ONLY this one
    # file — a single tree-sitter pass, ~milliseconds. Do NOT touch the whole
    # call graph here: the UserPromptSubmit hook imports _build_file_deps into a
    # fresh cold process per prompt, so anything that parses (or disk-caches)
    # all of src/**/*.php would make every PHP-naming prompt pay for the whole
    # tree. find_rationale (the cross-codebase view) uses the warm ParseCache;
    # this per-file view stays cheap.
    rationale_rows: list[tuple[int, str, str]] = []
    try:
        abs_path = PROJECT_ROOT / file_path
        syms, _edges, classes = call_graph._parse_file(abs_path, file_path)
        for info in list(syms.values()) + list(classes.values()):
            for n in info.get("rationale", []):
                rationale_rows.append((n.get("line", 0), n.get("tag", ""), n.get("text", "")))
    except Exception:  # noqa: BLE001 — rationale is a best-effort add-on
        rationale_rows = []
    if rationale_rows:
        rationale_rows.sort()
        lines.append("")
        lines.append("## Inline Rationale")
        for line_no, tag_name, text in rationale_rows:
            lines.append(f"- L{line_no} **{tag_name}**: {text}")

    return "\n".join(lines)


def _file_deps_twig(file_path: str) -> str:
    twig = _cache.get_twig_graph()
    routes = _cache.get_route_map()
    info = twig["templates"].get(file_path)
    if not info:
        return f"Template not found: {file_path}"

    # Find controllers that render this template
    rendered_by = [
        (p, r) for p, r in routes["routes"].items() if r["template"] == file_path.removeprefix("templates/")
    ]

    lines = [f"# {file_path}"]
    if info["extends"]:
        lines.append(f"- Extends: `{info['extends']}`")
    if info["includes"]:
        lines.append("- Includes:")
        for inc in info["includes"]:
            lines.append(f"  - {inc}")
    if info["included_by"]:
        lines.append("- Included by:")
        for inc in info["included_by"]:
            lines.append(f"  - {inc}")
    if info["stimulus_controllers"]:
        lines.append(f"- Stimulus controllers: {', '.join(info['stimulus_controllers'])}")
    if rendered_by:
        lines.append("")
        lines.append("## Rendered By")
        for p, r in rendered_by:
            lines.append(f"- {r['file']}::{r['action']} (route `{p}`)")
    return "\n".join(lines)


def _file_deps_stimulus(file_path: str) -> str:
    stim = _cache.get_stimulus_map()
    # Map back from file path to controller name
    name = None
    info = None
    for n, i in stim["controllers"].items():
        if i["file"] == file_path:
            name = n
            info = i
            break
    if not info:
        return f"Stimulus controller not found: {file_path}"

    lines = [
        f"# {file_path}",
        f"- Stimulus name: `{name}`",
        f"- Values: {info['values']}",
        f"- Targets: {info['targets']}",
        f"- Outlets: {info['outlets']}",
        "",
        f"## Used In ({len(info['used_in'])} templates)",
    ]
    for t in info["used_in"][:30]:
        lines.append(f"- {t}")
    return "\n".join(lines)


def _slice_signature(text: str) -> str:
    """Return a method declaration with its body stripped.

    Scans ``text`` (the declaration region, attributes included) char by char
    and stops at the first ``{`` or ``;`` seen while *outside* any ``()``/``[]``
    and outside a string literal. That guard is what keeps route-attribute
    placeholders like ``#[Route('/user/{id}')]`` from truncating the signature
    at their ``{`` — the placeholder sits inside ``[...]`` and ``(...)``, so its
    brace is skipped and only the real body brace ends the slice. Internal
    whitespace is collapsed so multi-line / promoted-constructor signatures
    render on one line.

    String literals (``'`` ``"`` and JS backtick) and comments (``//``, ``#``
    line comments, ``/* */`` blocks) inside the param list are skipped so a
    stray ``)`` / ``{`` in their text cannot end the slice early. Heredoc/nowdoc
    param defaults are not handled — they do not occur in this codebase.
    """
    out: list[str] = []
    depth_paren = 0
    depth_brack = 0
    in_str: str | None = None
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_str is not None:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == in_str:
                in_str = None
            i += 1
            continue
        nxt = text[i + 1] if i + 1 < n else ""
        # Comments — skip their text entirely (``#[`` is an attribute, not one).
        if (ch == "/" and nxt == "/") or (ch == "#" and nxt != "["):
            j = text.find("\n", i)
            if j == -1:
                break
            i = j
            continue
        if ch == "/" and nxt == "*":
            j = text.find("*/", i + 2)
            if j == -1:
                break
            i = j + 2
            continue
        if ch in ("'", '"', "`"):
            in_str = ch
            out.append(ch)
            i += 1
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]":
            depth_brack -= 1
        elif ch in ("{", ";") and depth_paren <= 0 and depth_brack <= 0:
            break
        out.append(ch)
        i += 1
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _build_file_api(file_path: str) -> str:
    """Return the API surface of a file — class + method signatures, no bodies.

    A token-cheap way to learn a file's shape before reading it in full
    (mirrors Graft's ``file_api``). Reuses the mtime-cached call graph — each
    symbol already carries ``file``/``line``/``end_line``/``visibility`` — and
    slices the declaration from source up to the body-opening ``{`` (or the
    ``;`` of an abstract/interface method). Attributes such as ``#[Route(...)]``
    sit inside the sliced region, so controller actions show their URL binding.

    Covers PHP classes and Stimulus controller JS. Interfaces, traits, and
    free functions are not in the call graph, so they are not listed here;
    Twig templates have no method surface — use ``get_template_graph`` for those.
    """
    file_path = file_path.replace("\\", "/").lstrip("./")

    graph = _cache.get_call_graph()
    symbols = graph["symbols"]
    classes = graph["classes"]

    # Symbols defined in this file, in source order (call graph is sorted by line).
    file_symbols = sorted(
        ((sid, info) for sid, info in symbols.items() if info.get("file") == file_path),
        key=lambda pair: pair[1].get("line", 0),
    )
    if not file_symbols:
        return (
            f"No parsed API surface for: {file_path}\n"
            "This tool covers PHP classes and Stimulus controller JS. "
            "Interfaces/traits/free functions and Twig templates are not indexed "
            "here — use get_file_deps or get_template_graph instead."
        )

    # Read the source once for signature slicing.
    try:
        src_lines = (PROJECT_ROOT / file_path).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError:
        src_lines = []

    def _signature(info: dict) -> str:
        start_line = info.get("line", 0)
        end_line = info.get("end_line", start_line)
        if not src_lines or start_line < 1:
            return ""
        lo = start_line - 1
        hi = min(end_line, start_line + 60)  # cap the scan region
        region = "\n".join(src_lines[lo:hi])
        return _slice_signature(region)

    # Group methods by owner (PHP FQCN or ``js:<name>``), preserving first-seen order.
    methods_by_owner: dict[str, list] = {}
    owner_first_line: dict[str, int] = {}
    for sid, info in file_symbols:
        if sid.startswith("js:"):
            owner = sid.split("::", 1)[0]
        elif "::" in sid:
            owner = sid.rsplit("::", 1)[0]
        else:
            owner = "(file)"
        methods_by_owner.setdefault(owner, []).append((sid, info))
        owner_first_line.setdefault(owner, info.get("line", 0))

    total = sum(len(v) for v in methods_by_owner.values())
    lines = [
        f"# API surface: {file_path}",
        f"_{total} method(s) across {len(methods_by_owner)} type(s). "
        "Signatures only — no bodies._",
        "",
    ]

    for owner in sorted(methods_by_owner, key=lambda o: owner_first_line[o]):
        if owner.startswith("js:"):
            lines.append(f"## Stimulus controller `{owner[3:]}`")
        else:
            cinfo = classes.get(owner, {})
            header = f"## `{owner}`"
            if cinfo.get("extends"):
                header += f" extends `{cinfo['extends']}`"
            lines.append(header)
            for note in cinfo.get("rationale", []):
                lines.append(f"> {note['tag']}: {note['text']}")
        lines.append("")
        for sid, info in sorted(methods_by_owner[owner], key=lambda t: t[1].get("line", 0)):
            sig = _signature(info)
            if not sig:  # fallback when source is unreadable
                name = sid.rsplit("::", 1)[-1]
                vis = info.get("visibility", "")
                sig = f"{vis} {name}(…)".strip()
            lines.append(f"- L{info.get('line', 0)} `{sig}`")
            for note in info.get("rationale", []):
                lines.append(f"  > {note['tag']}: {note['text']}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _build_route_map(prefix: str = "") -> str:
    routes = _cache.get_route_map()
    filtered = {
        p: r for p, r in routes["routes"].items()
        if not prefix or p.startswith(prefix)
    }
    if not filtered:
        return f"No routes match prefix: {prefix}"

    lines = [
        f"# Routes ({len(filtered)} matching `{prefix or 'ALL'}`)",
        "",
        "| Method | Path | Controller::action | Template | Services |",
        "|---|---|---|---|---|",
    ]
    for p in sorted(filtered):
        r = filtered[p]
        methods = ",".join(r["methods"])
        ctrl_short = r["controller"].split("\\")[-1]
        template = r["template"] or "-"
        services = ", ".join(r["services"][:4]) or "-"
        lines.append(f"| {methods} | `{p}` | {ctrl_short}::{r['action']} | {template} | {services} |")
    return "\n".join(lines)


def _build_template_graph(template: str = "") -> str:
    twig = _cache.get_twig_graph()
    if template:
        # Allow both "arena/index.html.twig" and "templates/arena/index.html.twig"
        key = template if template.startswith("templates/") else f"templates/{template}"
        info = twig["templates"].get(key)
        if not info:
            return f"Template not found: {template}"
        # Delegate to file_deps which already has the right formatting
        return _file_deps_twig(key)

    # Full tree — list inheritance chains
    lines = ["# Template Inheritance Tree", ""]
    for parent, children in sorted(twig["inheritance_chains"].items()):
        lines.append(f"## {parent} ({len(children)} children)")
        for c in sorted(children)[:20]:
            lines.append(f"- {c}")
        if len(children) > 20:
            lines.append(f"- ... and {len(children) - 20} more")
        lines.append("")
    return "\n".join(lines)


def _build_stimulus_map(controller: str = "") -> str:
    stim = _cache.get_stimulus_map()
    if controller:
        info = stim["controllers"].get(controller)
        if not info:
            return f"Stimulus controller not found: `{controller}`"
        lines = [
            f"# Stimulus controller: `{controller}`",
            f"- File: `{info['file']}`",
            f"- Values: {info['values']}",
            f"- Targets: {info['targets']}",
            f"- Outlets: {info['outlets']}",
            "",
            f"## Used In ({len(info['used_in'])} templates)",
        ]
        for t in info["used_in"]:
            lines.append(f"- {t}")
        return "\n".join(lines)

    lines = [
        f"# Stimulus Map ({stim['stats']['total_controllers']} controllers, {stim['stats']['total_usages']} usages)",
        "",
    ]
    for name in sorted(stim["controllers"].keys()):
        info = stim["controllers"][name]
        lines.append(f"- `{name}` ({len(info['used_in'])} usages)")
    if stim["orphan_controllers"]:
        lines.append("")
        lines.append(f"## Orphan controllers (no template usage, {len(stim['orphan_controllers'])})")
        for o in stim["orphan_controllers"]:
            lines.append(f"- `{o}`")
    if stim["missing_controllers"]:
        lines.append("")
        lines.append(f"## Missing controllers (referenced but no JS file, {len(stim['missing_controllers'])})")
        for m in stim["missing_controllers"]:
            lines.append(f"- `{m}`")
    return "\n".join(lines)


# Entity getter/setter calls (User::setEmail, User::getId, …) dominate a trace
# tree without adding architectural signal — a real usefulness complaint. We
# collapse leaf calls to `\Entity\` accessors into a single summary line per
# parent so the meaningful service/repository hops stand out.
_ACCESSOR_METHOD_RE = re.compile(r"^(get|set|is|has|add|remove)[A-Z0-9_]")


def _is_entity_accessor(node: dict) -> bool:
    """True if ``node`` is a childless call to a getter/setter on an Entity class."""
    if node.get("children") or node.get("truncated"):
        # A setter that does real work isn't noise; a cycle-/depth-truncated
        # node must keep its own line so the ``_(cycle)_`` marker isn't lost.
        return False
    symbol = node.get("symbol", "")
    if "::" not in symbol:
        return False
    class_part, _, method = symbol.rpartition("::")
    if "\\Entity\\" not in class_part:
        return False
    return bool(_ACCESSOR_METHOD_RE.match(method))


def _build_trace_route(
    method: str,
    path: str,
    max_depth: int = 6,
    output_format: str = "text",
    collapse_accessors: bool = True,
) -> str:
    """Trace the call graph from a route's controller action down through services + repos.

    Returns markdown with the route header and an indented tree of resolved
    callees. ``method`` is matched case-insensitively against the route's
    declared methods (or any method when the route declares none).

    ``output_format='mermaid'`` returns the call tree as a ``flowchart TD``
    fenced block instead of an indented bullet list — easier to read once
    branching exceeds 2-3 levels.
    """
    routes = _cache.get_route_map()
    route_entry = routes["routes"].get(path)
    if route_entry is None:
        return f"No route found at path: `{path}`"

    method = method.upper()
    if route_entry["methods"] and method not in route_entry["methods"]:
        return (
            f"Route `{path}` does not handle `{method}` "
            f"(handles: {', '.join(route_entry['methods'])})"
        )

    from_id = f"{route_entry['controller']}::{route_entry['action']}"
    graph = _cache.get_call_graph()
    tree = call_graph.trace(graph, from_id, max_depth=max_depth)

    if output_format == "mermaid":
        if tree.get("missing"):
            return (
                f"# Trace: `{method} {path}`\n\n"
                f"Controller action `{from_id}` not in call graph — "
                f"nothing to render."
            )
        body = mermaid_render.render_trace_tree(tree, root_label=f"**{from_id}**")
        return (
            f"# Trace: `{method} {path}`\n\n"
            f"```mermaid\n{body}\n```"
        )

    lines = [
        f"# Trace: `{method} {path}`",
        f"- Controller: `{route_entry['controller']}`",
        f"- Action: `{route_entry['action']}`",
        f"- File: `{route_entry['file']}`",
        f"- Max depth: {max_depth}",
        "",
        "## Call tree",
    ]
    if tree.get("missing"):
        lines.append(
            f"- `{from_id}` (no symbol — controller action not picked up by call graph)"
        )
    else:
        _render_trace_node(tree, lines, indent=0, is_root=True, collapse_accessors=collapse_accessors)
    return "\n".join(lines)


def _render_trace_node(
    node: dict,
    lines: list[str],
    indent: int,
    is_root: bool,
    collapse_accessors: bool = True,
) -> None:
    """Append one indented line per node in the trace tree, depth-first.

    A ``missing=True`` flag (target not in local symbol table — vendor or
    inherited from a vendor base class) is conveyed by the FQCN itself, so
    we don't add a noisy marker. ``truncated="cycle"`` IS marked because
    the reader cannot otherwise tell why the subtree ends.
    """
    prefix = "  " * indent + "- "
    if is_root:
        lines.append(f"{prefix}**{node['symbol']}**")
    else:
        kind = node.get("kind", "call")
        confidence = node.get("confidence")
        evidence = node.get("evidence", "")
        marker_text = " _(cycle)_" if node.get("truncated") == "cycle" else ""
        conf_text = f" c={confidence}" if confidence is not None else ""
        ev_text = f" :: `{evidence}`" if evidence else ""
        kind_tag = f"[{kind}]"
        lines.append(f"{prefix}{kind_tag} `{node['symbol']}`{conf_text}{ev_text}{marker_text}")

    children = node["children"]
    if collapse_accessors:
        accessors = [c for c in children if _is_entity_accessor(c)]
        rest = [c for c in children if not _is_entity_accessor(c)]
    else:
        accessors, rest = [], children

    for child in rest:
        _render_trace_node(child, lines, indent + 1, is_root=False, collapse_accessors=collapse_accessors)

    if accessors:
        child_prefix = "  " * (indent + 1) + "- "
        names = [a["symbol"].rpartition("::")[0].rsplit("\\", 1)[-1] + "::" + a["symbol"].rpartition("::")[2]
                 for a in accessors]
        # Dedup while preserving order (the same setter can appear twice).
        seen: set[str] = set()
        uniq = [n for n in names if not (n in seen or seen.add(n))]
        shown = ", ".join(uniq[:6])
        more = f" +{len(uniq) - 6} more" if len(uniq) > 6 else ""
        lines.append(
            f"{child_prefix}_[{len(accessors)} entity accessor call(s) collapsed]_ {shown}{more}"
        )


def _run_git_diff(since_ref: str, file: str | None = None) -> str:
    """Run ``git diff -U0 <since_ref> [-- <file>]`` against PROJECT_ROOT.

    Returns raw stdout. Empty string if diff produces no output or git
    fails — caller handles 'no changes' as an empty-string case.
    """
    cmd = ["git", "diff", "-U0", since_ref]
    if file:
        cmd += ["--", file]
    try:
        return subprocess.check_output(
            cmd,
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        ).decode("utf-8", errors="replace")
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""


def _git_lines(args: list[str]) -> list[str]:
    """Run ``git <args>`` in PROJECT_ROOT, return non-empty stripped stdout lines.

    Returns ``[]`` on any git failure (missing ref, not a repo, git absent) —
    callers treat empty as "nothing / unavailable".
    """
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        ).decode("utf-8", errors="replace")
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


# Branch-name prefixes auto-detection skips by default. This repo's worktree
# workflow mints a `backup/main-pre-merge-<date>` safety branch on every merge;
# each is "ahead of main" by a whole feature and would flood the report. Pass
# explicit `branches` to override.
_MERGE_RISK_SKIP_PREFIXES = ("backup/", "archive/", "wip/", "tmp/")


def _branches_ahead_of(base: str, max_branches: int) -> list[str]:
    """Local branches (excluding ``base`` + noise prefixes) ahead of ``base``."""
    all_branches = _git_lines(["for-each-ref", "--format=%(refname:short)", "refs/heads/"])
    ahead: list[str] = []
    for b in all_branches:
        if b == base or b.startswith(_MERGE_RISK_SKIP_PREFIXES):
            continue
        cnt = _git_lines(["rev-list", "--count", f"{base}..{b}"])
        if cnt and cnt[0].isdigit() and int(cnt[0]) > 0:
            ahead.append(b)
        if len(ahead) >= max_branches:
            break
    return ahead


def _branch_ancestry(branches: list[str]) -> set[tuple[str, str]]:
    """Return ``{(ancestor, descendant)}`` for every ordered pair that is one.

    Uses ``git merge-base --is-ancestor a b`` (exit 0 ⇒ a is an ancestor of
    b). O(n²) subprocesses, but ``branches`` is capped small (≤ max_branches).
    """
    pairs: set[tuple[str, str]] = set()
    for a in branches:
        for b in branches:
            if a == b:
                continue
            try:
                rc = subprocess.call(
                    ["git", "merge-base", "--is-ancestor", a, b],
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                )
            except (FileNotFoundError, OSError):
                return set()
            if rc == 0:
                pairs.add((a, b))
    return pairs


def _build_find_message_handlers(message: str | None = None) -> str:
    """Render the Symfony Messenger contract map (message ↔ handler ↔ producer).

    Reads ``messenger_map.parse`` output. With ``message`` set, shows the full
    detail (every handler + every dispatch site) for matching messages; without
    it, a compact overview plus the orphan lists.
    """
    mm = _cache.get_messenger_map()
    messages: dict = mm["messages"]
    stats: dict = mm["stats"]

    if not messages:
        return "No Symfony Messenger messages, handlers, or dispatch sites found in `src/`."

    def _short(fqcn: str) -> str:
        return messages.get(fqcn, {}).get("short") or fqcn.split("\\")[-1]

    # --- Filtered detail view ----------------------------------------------
    if message:
        needle = message.lower().lstrip("\\")
        hits = sorted(
            fqcn for fqcn in messages
            if needle in fqcn.lower() or needle in _short(fqcn).lower()
        )
        if not hits:
            return f"No message matching `{message}` (searched {len(messages)} known messages)."
        lines: list[str] = []
        for fqcn in hits:
            e = messages[fqcn]
            lines.append(f"### `{fqcn}`")
            if e["handlers"]:
                for h in e["handlers"]:
                    lines.append(
                        f"- **handled by** `{h['handler_class']}::{h['handler_method']}` "
                        f"— {h['file']}:{h['line']}"
                    )
            else:
                lines.append("- ⚠️ **no handler** — dispatched but nothing consumes it")
            if e["producers"]:
                for p in sorted(e["producers"], key=lambda x: (x["file"], x["line"])):
                    lines.append(f"- dispatched at {p['file']}:{p['line']} (`{p['via']}`)")
            else:
                lines.append("- no literal `dispatch(new …)` site found (may be dispatched via a variable)")
            lines.append("")
        return "\n".join(lines).rstrip()

    # --- Overview view ------------------------------------------------------
    lines = [
        f"**Messenger contract** — {stats['total_messages']} handled messages, "
        f"{stats['total_handlers']} handlers, {stats['total_producer_sites']} dispatch sites.",
        "",
    ]
    handled = sorted(fqcn for fqcn, e in messages.items() if e["handlers"])
    for fqcn in handled:
        e = messages[fqcn]
        handler_str = ", ".join(f"`{h['handler_class'].split(chr(92))[-1]}::{h['handler_method']}`" for h in e["handlers"])
        nprod = len(e["producers"])
        prod_str = f"{nprod} dispatch site{'s' if nprod != 1 else ''}" if nprod else "⚠️ never dispatched (no literal site)"
        lines.append(f"- `{e['short']}` → {handler_str} — {prod_str}")

    unhandled = mm["orphans"]["unhandled"]
    if unhandled:
        lines.append("")
        lines.append("**⚠️ Unhandled (dispatched, no handler):**")
        for fqcn in unhandled:
            lines.append(f"- `{fqcn}`")

    undispatched = mm["orphans"]["undispatched"]
    if undispatched:
        lines.append("")
        lines.append("**Handled but no literal dispatch site (check for variable dispatch):**")
        for fqcn in undispatched:
            lines.append(f"- `{_short(fqcn)}`")

    if stats.get("unresolved_dispatch"):
        lines.append("")
        lines.append(
            f"_{stats['unresolved_dispatch']} `dispatch(...)` call(s) pass a variable "
            f"rather than `new X()` — not resolved (may include EventDispatcher calls)._"
        )
    return "\n".join(lines)


def _build_merge_order_risk(
    base: str = "main",
    branches: list[str] | None = None,
    max_branches: int = 12,
) -> str:
    """Assess merge-order conflict risk across in-flight branches.

    Auto-detects local branches ahead of ``base`` when ``branches`` is not
    given, diffs each against ``base`` (three-dot / merge-base), maps the
    changed files into unified-graph communities, and reports direct file
    conflicts + community-overlap coupling risk.
    """
    from scripts import communities as _comm, merge_risk
    from scripts.config import KNOWLEDGE_DIR

    graph = _cache.get_unified_graph()
    # min_size=2 so even small clusters participate (same partition as
    # find_community), maximizing overlap detection.
    cache_path = KNOWLEDGE_DIR / "communities.min2.json"
    clusters = _comm.load_or_compute(graph, cache_path=cache_path, min_size=2)
    node_community = merge_risk.build_node_community_map(clusters)

    if branches is None:
        branches = _branches_ahead_of(base, max_branches)
    else:
        branches = [b for b in branches if b != base][:max_branches]

    if not branches:
        return (
            f"No branches found ahead of `{base}`. Nothing in flight to compare. "
            f"(Pass explicit branch names, or check that `{base}` exists.)"
        )

    branch_files: dict[str, list[str]] = {}
    for b in branches:
        files = _git_lines(["diff", "--name-only", f"{base}...{b}"])
        branch_files[b] = [f.replace("\\", "/") for f in files]

    # Detect stacked branches so we don't report a parent's files restated in
    # a child's diff as a conflict.
    ancestry = _branch_ancestry(branches)

    result = merge_risk.compute(branch_files, node_community, ancestry=ancestry)
    return merge_risk.render(result, base)


def _build_impact_of_change(
    file: str | None = None,
    since_ref: str = "HEAD",
    max_depth: int = 6,
    output_format: str = "text",
) -> str:
    """Show what's downstream-affected by code changes since ``since_ref``.

    Pipeline:
        1. ``git diff -U0`` to get changed line ranges per file.
        2. Map ranges to changed symbols via ``call_graph.find_changed_symbols``.
        3. For each changed symbol, walk reverse callers up to ``max_depth``.
        4. Match upstream callers against the route map → affected routes.
        5. Score each route by (changed-symbols-reached × hotspot-multiplier).
    """
    diff_text = _run_git_diff(since_ref, file)
    if not diff_text.strip():
        return f"# Impact of change\n\nNo changes between working tree and `{since_ref}`."

    file_ranges = call_graph.parse_diff_hunks(diff_text)
    if not file_ranges:
        return f"# Impact of change\n\nNo PHP files changed in diff vs `{since_ref}`."

    graph = _cache.get_call_graph()
    changed = call_graph.find_changed_symbols(graph, file_ranges)
    if not changed:
        return (
            f"# Impact of change\n\n"
            f"PHP files changed but no method bodies overlapped: {sorted(file_ranges)}"
        )

    # Build hotspot lookup once for risk scoring
    git = _cache.get_git_intel()
    hotspot_score: dict[str, float] = {
        h["file"]: h["score"] for h in git.get("hotspots", [])
    }

    # Index controller actions by symbol_id so reverse-walked callers can
    # be matched back to their public route.
    routes = _cache.get_route_map()
    route_by_action_symbol: dict[str, list[tuple[str, dict]]] = {}
    for path, route in routes["routes"].items():
        action_symbol = f"{route['controller']}::{route['action']}"
        route_by_action_symbol.setdefault(action_symbol, []).append((path, route))

    # For each changed symbol, walk upstream. Two views of the result:
    #   - HTTP route matches (controller action symbols matched against route map)
    #   - Stimulus frontend matches (any caller with `js:` prefix)
    affected: dict[str, dict] = {}  # path -> {route, reaches: [{symbol, depth}]}
    js_reaches: dict[str, list[dict]] = {}  # js_symbol -> [{symbol, depth}]
    for changed_symbol in changed:
        callers = call_graph.reverse_callers(graph, changed_symbol, max_depth=max_depth)
        for caller in callers:
            for path, route in route_by_action_symbol.get(caller["symbol"], []):
                entry = affected.setdefault(path, {"route": route, "reaches": []})
                entry["reaches"].append({
                    "symbol": changed_symbol,
                    "depth": caller["depth"],
                })
            if caller["symbol"].startswith("js:"):
                js_reaches.setdefault(caller["symbol"], []).append({
                    "symbol": changed_symbol,
                    "depth": caller["depth"],
                })

    # Risk score: sum over reaches of (1 / depth) × file's hotspot score (default 1).
    for entry in affected.values():
        risk = 0.0
        for reach in entry["reaches"]:
            sym = graph["symbols"].get(reach["symbol"], {})
            multiplier = hotspot_score.get(sym.get("file", ""), 1.0)
            risk += (1.0 / max(reach["depth"], 1)) * multiplier
        entry["risk"] = round(risk, 2)

    sorted_routes = sorted(affected.items(), key=lambda kv: kv[1]["risk"], reverse=True)

    if output_format == "mermaid":
        body = mermaid_render.render_impact_graph(
            changed_symbols=sorted(changed),
            affected_routes=sorted_routes,
            js_reaches=js_reaches,
        )
        return (
            f"# Impact of change vs `{since_ref}`\n\n"
            f"- Files touched: {len(file_ranges)}\n"
            f"- Changed methods: {len(changed)}\n"
            f"- Affected routes: {len(affected)}\n\n"
            f"```mermaid\n{body}\n```"
        )

    lines = [
        f"# Impact of change vs `{since_ref}`",
        f"- Files touched: {len(file_ranges)}",
        f"- Changed methods: {len(changed)}",
        f"- Affected routes: {len(affected)}",
        "",
        "## Changed methods",
    ]
    for sid in sorted(changed):
        sym = graph["symbols"].get(sid, {})
        lines.append(f"- `{sid}` :: `{sym.get('file', '?')}:{sym.get('line', '?')}`")

    if sorted_routes:
        lines.append("")
        lines.append("## Affected routes (sorted by risk)")
        for path, info in sorted_routes:
            r = info["route"]
            methods = ",".join(r["methods"])
            ctrl_short = r["controller"].rsplit("\\", 1)[-1]
            lines.append(
                f"- **{methods} `{path}`** -> `{ctrl_short}::{r['action']}` "
                f"(risk {info['risk']}, {len(info['reaches'])} reaches)"
            )
            for reach in sorted(info["reaches"], key=lambda x: x["depth"]):
                lines.append(
                    f"    - reaches `{reach['symbol']}` (depth {reach['depth']})"
                )

    if js_reaches:
        lines.append("")
        lines.append("## Affected Stimulus controllers (JS frontend)")
        for js_symbol in sorted(js_reaches):
            reaches = js_reaches[js_symbol]
            min_depth = min(r["depth"] for r in reaches)
            lines.append(
                f"- `{js_symbol}` ({len(reaches)} reach"
                f"{'es' if len(reaches) != 1 else ''}, min depth {min_depth})"
            )
            for reach in sorted(reaches, key=lambda x: x["depth"]):
                lines.append(
                    f"    - reaches `{reach['symbol']}` (depth {reach['depth']})"
                )

    return "\n".join(lines)


def _build_circular_dependencies(scope: str = "all", output_format: str = "text") -> str:
    """Detect strongly-connected components in the call graph (Tarjan's SCC).

    A cycle is any SCC of size > 1. Singleton SCCs that contain a self-edge
    are also reported (rare but real — a method that calls itself directly).

    ``scope`` filters the symbol set:
        ``all``    every symbol (PHP + JS)
        ``php``    only ``App\\...`` PHP symbols
        ``js``     only ``js:...`` Stimulus symbols
        ``vendor-excluded``  PHP minus vendor namespaces (Symfony, Doctrine, etc.)
    """
    graph = _cache.get_call_graph()
    edges = graph.get("edges", []) or []
    symbols = graph.get("symbols", {}) or {}

    def _accept(sym_id: str) -> bool:
        if scope == "all":
            return True
        if scope == "php":
            return not sym_id.startswith("js:")
        if scope == "js":
            return sym_id.startswith("js:")
        if scope == "vendor-excluded":
            if sym_id.startswith("js:"):
                return False
            # Heuristic: a symbol is "vendor" if it's referenced in edges but
            # has no entry in the symbol table (we only parse src/), OR if
            # its file path lives outside src/.
            sym = symbols.get(sym_id)
            if sym is None:
                return False
            f = sym.get("file") or ""
            return f.startswith("src/")
        return True

    # Build adjacency only for accepted nodes — Tarjan on the full graph
    # would surface dozens of vendor-internal cycles that are noise.
    adj: dict[str, list[str]] = {}
    for edge in edges:
        a, b = edge.get("from"), edge.get("to")
        if not a or not b:
            continue
        if not _accept(a) or not _accept(b):
            continue
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, [])  # ensure target is a node even if it has no out-edges

    # Iterative Tarjan to avoid recursion limit on large graphs.
    index_counter = [0]
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    sccs: list[list[str]] = []

    def _strongconnect(start: str) -> None:
        # Iterative DFS using an explicit work stack of (node, child_iterator).
        work: list[tuple[str, "iter"]] = []
        indices[start] = index_counter[0]
        lowlinks[start] = index_counter[0]
        index_counter[0] += 1
        stack.append(start)
        on_stack.add(start)
        work.append((start, iter(adj.get(start, []))))

        while work:
            node, it = work[-1]
            advanced = False
            for w in it:
                if w not in indices:
                    indices[w] = index_counter[0]
                    lowlinks[w] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(w)
                    on_stack.add(w)
                    work.append((w, iter(adj.get(w, []))))
                    advanced = True
                    break
                elif w in on_stack:
                    lowlinks[node] = min(lowlinks[node], indices[w])
            if not advanced:
                if lowlinks[node] == indices[node]:
                    component: list[str] = []
                    while True:
                        v = stack.pop()
                        on_stack.discard(v)
                        component.append(v)
                        if v == node:
                            break
                    sccs.append(component)
                work.pop()
                if work:
                    parent_node = work[-1][0]
                    lowlinks[parent_node] = min(lowlinks[parent_node], lowlinks[node])

    for node in list(adj.keys()):
        if node not in indices:
            _strongconnect(node)

    self_loops = {edge["from"] for edge in edges if edge.get("from") == edge.get("to")}
    cycles: list[list[str]] = []
    for comp in sccs:
        if len(comp) > 1:
            cycles.append(sorted(comp))
        elif len(comp) == 1 and comp[0] in self_loops and _accept(comp[0]):
            cycles.append(comp)

    cycles.sort(key=lambda c: (-len(c), c[0]))

    if output_format == "mermaid":
        return (
            f"# Circular dependencies (scope={scope})\n\n"
            f"Found {len(cycles)} cycle(s).\n\n"
            f"```mermaid\n{mermaid_render.render_cycles(cycles)}\n```"
        )

    lines = [f"# Circular dependencies (scope={scope})", ""]
    if not cycles:
        lines.append("No cycles detected.")
        return "\n".join(lines)
    lines.append(f"Found **{len(cycles)}** cycle(s).")
    lines.append("")
    for idx, cycle in enumerate(cycles, 1):
        lines.append(f"## Cycle {idx} ({len(cycle)} symbols)")
        for sym in cycle:
            sym_meta = symbols.get(sym, {})
            location = ""
            if sym_meta:
                location = f" :: `{sym_meta.get('file', '?')}:{sym_meta.get('line', '?')}`"
            lines.append(f"- `{sym}`{location}")
        lines.append("")
    return "\n".join(lines)


def _build_hotspots(top_n: int = 10) -> str:
    git = _cache.get_git_intel()
    hotspots = git.get("hotspots", [])[:top_n]
    if not hotspots:
        return "No hotspots available (git intel empty)"
    lines = [f"# Top {len(hotspots)} Hotspots", ""]
    for h in hotspots:
        lines.append(f"## {h['file']}")
        lines.append(f"- Score: **{h['score']}**")
        lines.append(f"- Commits: {h['commits_total']} total / {h['commits_30d']} in 30d / {h['commits_90d']} in 90d")
        lines.append(f"- Lines (90d): +{h['lines_added_90d']} -{h['lines_deleted_90d']}")
        lines.append(f"- Primary owner: {h['primary_owner']}")
        lines.append(f"- Bus factor: {h['bus_factor']}")
        if h["co_change_partners"]:
            lines.append("- Co-change partners:")
            for p in h["co_change_partners"]:
                lines.append(f"  - {p['file']} (score={p['score']})")
        lines.append("")
    return "\n".join(lines)


def _build_unified_neighbors(node_id: str, depth: int = 1) -> str:
    """Render a markdown summary of nodes within ``depth`` hops of ``node_id``.

    Depth is clamped to ``[1, 3]``. Edges are grouped by direction (outgoing
    first, then incoming) and by ``kind`` within each direction. Each
    neighboring node shows kind + label; the edge between root and neighbor
    shows the edge kind and ``relation``/``confidence`` if present.

    Returns a friendly ``"Node not found"`` string when ``node_id`` is not in
    the graph rather than raising — MCP tool callers prefer a string they
    can hand to the model over an exception trace.
    """
    depth = max(1, min(3, int(depth)))
    graph = _cache.get_unified_graph()
    nodes = graph["nodes"]
    edges = graph["edges"]
    if node_id not in nodes:
        return f"Node not found: `{node_id}`"

    out_by_kind: dict[str, list[dict]] = {}
    in_by_kind: dict[str, list[dict]] = {}
    for e in edges:
        if e["from"] == node_id:
            out_by_kind.setdefault(e["kind"], []).append(e)
        if e["to"] == node_id:
            in_by_kind.setdefault(e["kind"], []).append(e)

    root = nodes[node_id]
    lines: list[str] = [
        f"# `{node_id}`",
        f"- Kind: **{root['kind']}**",
        f"- Label: {root.get('label', '')}",
        "",
    ]
    if out_by_kind:
        lines.append("## Outgoing")
        for kind, group in sorted(out_by_kind.items()):
            lines.append(f"### {kind} ({len(group)})")
            for e in group[:50]:
                target = nodes.get(e["to"], {})
                label = target.get("label", "")
                suffix = ""
                if "relation" in e:
                    suffix = f" *(relation: {e['relation']})*"
                elif "confidence" in e:
                    suffix = f" *(conf={e['confidence']})*"
                lines.append(f"- `{e['to']}` — {label}{suffix}")
            lines.append("")
    if in_by_kind:
        lines.append("## Incoming")
        for kind, group in sorted(in_by_kind.items()):
            lines.append(f"### {kind} ({len(group)})")
            for e in group[:50]:
                source = nodes.get(e["from"], {})
                label = source.get("label", "")
                lines.append(f"- `{e['from']}` — {label}")
            lines.append("")

    if depth >= 2:
        adj: dict[str, set[str]] = {}
        for e in edges:
            adj.setdefault(e["from"], set()).add(e["to"])
            adj.setdefault(e["to"], set()).add(e["from"])
        visited = {node_id}
        frontier = {node_id}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for n in frontier:
                next_frontier |= adj.get(n, set())
            next_frontier -= visited
            visited |= next_frontier
            frontier = next_frontier
        far = sorted(visited - {node_id} - set(e["to"] for e in edges if e["from"] == node_id) - set(e["from"] for e in edges if e["to"] == node_id))
        if far:
            lines.append(f"## {depth}-hop reachable (not direct)")
            for nid in far[:80]:
                lines.append(f"- `{nid}` — {nodes.get(nid, {}).get('label', '')}")

    return "\n".join(lines)


def _build_trace_path(from_node: str, to_node: str, max_depth: int = 8) -> str:
    """Shortest connection between two nodes in the unified graph.

    BFS over the undirected projection (edge direction is informational —
    "how are these related?" doesn't care which way an edge points, matching
    the community-detection projection). Returns the hop-by-hop path with the
    edge kind / relation / confidence annotating each step, so the agent sees
    *why* two concepts are connected — e.g. an article that cites a file whose
    class defines a symbol that another article also cites.

    ``max_depth`` (clamped ``[1, 12]``) bounds the search: if the shortest
    path is longer, the tool reports the nodes as not connected within that
    many hops rather than walking the whole graph.
    """
    max_depth = max(1, min(12, int(max_depth)))
    graph = _cache.get_unified_graph()
    nodes, edges = graph["nodes"], graph["edges"]

    missing = [n for n in (from_node, to_node) if n not in nodes]
    if missing:
        return "Node(s) not found in unified graph: " + ", ".join(f"`{m}`" for m in missing)
    if from_node == to_node:
        return f"`{from_node}` is the same node — path length 0."

    # Adjacency with the connecting edge + direction relative to traversal.
    adj: dict[str, list[tuple[str, dict, str]]] = {}
    for e in edges:
        adj.setdefault(e["from"], []).append((e["to"], e, "→"))
        adj.setdefault(e["to"], []).append((e["from"], e, "←"))

    # BFS carrying (prev_node, edge, arrow) so we can reconstruct + annotate.
    parent: dict[str, tuple[str, dict, str]] = {from_node: ("", {}, "")}
    frontier = [from_node]
    depth = 0
    found = False
    while frontier and depth < max_depth and not found:
        nxt: list[str] = []
        for n in frontier:
            for neighbor, edge, arrow in adj.get(n, ()):
                if neighbor in parent:
                    continue
                parent[neighbor] = (n, edge, arrow)
                if neighbor == to_node:
                    found = True
                    break
                nxt.append(neighbor)
            if found:
                break
        frontier = nxt
        depth += 1

    if to_node not in parent:
        return (
            f"No path from `{from_node}` to `{to_node}` within {max_depth} hops. "
            f"They may be in disconnected components — try `find_community` on each."
        )

    # Reconstruct the path from to_node back to from_node.
    chain: list[tuple[str, dict, str]] = []
    cur = to_node
    while cur != from_node:
        prev, edge, arrow = parent[cur]
        chain.append((cur, edge, arrow))
        cur = prev
    chain.reverse()

    def _label(nid: str) -> str:
        n = nodes.get(nid, {})
        return f"{n.get('kind', '?')}:{n.get('label', nid)}"

    lines = [
        f"# Path: `{from_node}` → `{to_node}`",
        f"- Hops: **{len(chain)}**",
        "",
        f"1. `{from_node}` — {_label(from_node)}",
    ]
    for i, (nid, edge, arrow) in enumerate(chain, start=2):
        kind = edge.get("kind", "?")
        extra = ""
        if "relation" in edge:
            extra = f", relation={edge['relation']}"
        elif "confidence" in edge:
            extra = f", conf={edge['confidence']}"
        lines.append(f"{i}. {arrow} *[{kind}{extra}]* `{nid}` — {_label(nid)}")
    return "\n".join(lines)


_FENCE_LANGS = {
    ".php": "php",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".twig": "twig",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".html": "html",
    ".css": "css",
    ".md": "markdown",
}


def _fence_lang(rel: str) -> str:
    """Return the markdown fence language tag for a relative file path."""
    from os.path import splitext

    return _FENCE_LANGS.get(splitext(rel)[1].lower(), "")


def _build_neighborhood(
    node_id: str,
    depth: int = 1,
    include_source: bool = True,
    max_source_lines: int = 200,
) -> str:
    """Neighbors + source for ``symbol:`` nodes grouped by owning file.

    Codegraph_explore-inspired bundler. Replaces ``get_unified_neighbors`` →
    N × ``Read`` with a single call that returns:

    1. The relationship map (delegated to ``_build_unified_neighbors``).
    2. A source bundle: every reachable ``symbol:`` node's source slice,
       grouped under its owning file's heading so the agent sees coherent
       units instead of scattered snippets.
    3. Header excerpts for any reachable ``file:`` node that has no
       ``contains`` edge — covers Twig templates, YAML configs, and plain
       function-only PHP files that aren't represented by class symbols.
    4. A list of reachable articles (call ``get_articles`` for bodies).

    Budgets: ``max_source_lines`` per symbol; total source output capped
    at ``5 * max_source_lines`` to keep one call from blowing the agent's
    context. Truncation is announced inline so the agent knows when to
    drill in further.
    """
    depth = max(1, min(3, int(depth)))
    graph = _cache.get_unified_graph()
    nodes, edges = graph["nodes"], graph["edges"]
    if node_id not in nodes:
        return f"Node not found: `{node_id}`"

    # BFS over both directions for `depth` hops.
    adj: dict[str, set[str]] = {}
    for e in edges:
        adj.setdefault(e["from"], set()).add(e["to"])
        adj.setdefault(e["to"], set()).add(e["from"])
    visited = {node_id}
    frontier = {node_id}
    for _ in range(depth):
        nxt = {t for n in frontier for t in adj.get(n, set())} - visited
        visited |= nxt
        frontier = nxt

    # Bucket reachable nodes. Symbols group by their owning file — this is
    # the codegraph_explore differentiator: agent sees `User` as a coherent
    # unit, not 5 unrelated snippets. The root is included when it's a
    # symbol; excluding it would be a UX trap — the agent asked about a
    # specific symbol and would get source for everything except that one.
    syms = _cache.get_call_graph().get("symbols", {})
    by_file: dict[str, list[tuple[str, dict]]] = {}
    article_ids: list[str] = []
    template_ids: list[str] = []
    for nid in visited:
        kind = nodes[nid]["kind"]
        if kind == "symbol":
            info = syms.get(nid.removeprefix("symbol:"))
            if info and info.get("file"):
                by_file.setdefault(info["file"], []).append((nid, info))
        elif kind == "article" and nid != node_id:
            article_ids.append(nid)
        elif kind == "template" and nid != node_id:
            template_ids.append(nid)

    root = nodes[node_id]
    out: list[str] = [
        f"# Neighborhood of `{node_id}` (depth={depth})",
        f"- Kind: **{root['kind']}** · Label: {root.get('label', '')}",
        f"- Reached {len(visited) - 1} nodes · "
        f"{len(by_file)} source files · "
        f"{len(article_ids)} articles · "
        f"{len(template_ids)} templates",
        "",
        _build_unified_neighbors(node_id, depth=1),
        "",
    ]

    if include_source and by_file:
        out += ["---", "## Source bundle", ""]
        budget = max_source_lines * 5
        spent = 0
        exhausted = False
        for rel in sorted(by_file):
            try:
                file_lines = (PROJECT_ROOT / rel).read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            lang = _fence_lang(rel)
            out.append(f"### `{rel}`")
            out.append("")
            for sym_id, info in sorted(by_file[rel], key=lambda x: x[1]["line"]):
                start = int(info["line"])
                end = int(info.get("end_line", start))
                # Trust metadata for the truncation decision: a symbol's
                # claimed length is what the index says it spans, even if the
                # file got truncated post-indexing. Use the actual sliced
                # content though — a shrunk file just gives us less to show.
                claimed_len = end - start + 1
                slice_ = file_lines[start - 1:end]
                if claimed_len > max_source_lines:
                    trimmed = claimed_len - max_source_lines
                    slice_ = slice_[:max_source_lines] + [
                        f"// ... {trimmed} more lines truncated"
                    ]
                # Charge the larger of (claimed, actual) so huge symbols
                # exhaust the budget even when their truncated emission is
                # small — the agent gets one slice + one signal to narrow.
                spent += max(claimed_len, len(slice_))
                out.append(f"**`{sym_id}`** — lines {start}-{end}")
                out.append(f"```{lang}")
                out.extend(slice_)
                out.append("```")
                out.append("")
                if spent >= budget:
                    out.append(
                        f"_Source budget exhausted ({budget} lines). "
                        f"Use `Read` or call this tool with a narrower node "
                        f"for remaining symbols._"
                    )
                    exhausted = True
                    break
            if exhausted:
                break

    # Bare files (no `contains` edge → not represented by class symbols)
    # get a header excerpt — typically Twig templates, YAML configs, or
    # plain function-only PHP. Skip files already in the source bundle.
    if include_source:
        file_ids = {nid for nid in visited - {node_id} if nodes[nid]["kind"] == "file"}
        files_with_classes = {
            e["from"] for e in edges if e["kind"] == "contains" and e["from"] in file_ids
        }
        bundled = {f"file:{r}" for r in by_file}
        bare_files = sorted(file_ids - files_with_classes - bundled)
        if bare_files:
            out += ["---", "## File headers (no classes extracted)", ""]
            for fid in bare_files:
                rel = fid.removeprefix("file:")
                try:
                    head = (PROJECT_ROOT / rel).read_text(encoding="utf-8").splitlines()[:40]
                except OSError:
                    continue
                lang = _fence_lang(rel)
                out += [f"### `{rel}`", f"```{lang}", *head, "```", ""]

    if article_ids:
        out += [
            "---",
            "## Related articles (call `get_articles` for full bodies)",
            "",
        ]
        for aid in sorted(article_ids):
            n = nodes[aid]
            out.append(
                f"- `{aid}` — **{n.get('label', '')}** "
                f"(type={n.get('type', '?')}, conf={n.get('confidence', '?')})"
            )

    return "\n".join(out)


def _build_communities(min_size: int = 3, top_n: int = 10) -> str:
    """Render the top-N largest semantic communities as markdown."""
    from scripts import communities as _comm
    from scripts.config import KNOWLEDGE_DIR

    graph = _cache.get_unified_graph()
    cache_path = KNOWLEDGE_DIR / f"communities.min{min_size}.json"
    clusters = _comm.load_or_compute(graph, cache_path=cache_path, min_size=min_size)
    if not clusters:
        return "No communities found (graph too small or fully disconnected)."

    lines = [f"# Semantic Communities (top {min(top_n, len(clusters))})", ""]
    for c in clusters[:top_n]:
        lines.append(f"## Community {c['community_id']}: {c['label']}")
        lines.append(f"- Size: **{c['size']}**")
        lines.append(f"- Hub node: `{c['hub_node']}`")
        sample = c["members"][:8]
        lines.append("- Sample members:")
        for nid in sample:
            label = graph["nodes"].get(nid, {}).get("label", "")
            lines.append(f"  - `{nid}` — {label}")
        if len(c["members"]) > 8:
            lines.append(f"  - ... and {len(c['members']) - 8} more")
        lines.append("")
    return "\n".join(lines)


def _build_salience(kind: str = "all", top_n: int = 10, scope: str = "all") -> str:
    """Render the graph salience report, or one section of it."""
    from scripts import salience as _sal
    from scripts.config import KNOWLEDGE_DIR

    graph = _cache.get_unified_graph()
    report = _sal.load_or_compute(graph, cache_path=KNOWLEDGE_DIR / "salience.json")

    if kind == "all":
        return _sal.render(report, top_n=top_n, scope=scope)

    valid = ("hubs", "brokers", "bridges", "orphans")
    if kind not in valid:
        return f"Unknown kind '{kind}'. Expected one of: all, {', '.join(valid)}."

    # Render the full report, then keep only the requested section. Cheaper
    # than a second renderer and guarantees the two can never drift.
    full = _sal.render(report, top_n=top_n, scope=scope)
    heading_starts = {
        "hubs": "## Hubs", "brokers": "## Brokers",
        "bridges": "## Bridges", "orphans": "## Orphans",
    }
    lines = full.split("\n")
    start = next((i for i, ln in enumerate(lines) if ln.startswith(heading_starts[kind])), None)
    if start is None:
        return full
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start:end]).rstrip()


def _build_find_rationale(tag: str | None = None, query: str | None = None, limit: int = 100) -> str:
    """List inline rationale comments (WHY / HACK / TODO / @deprecated / …).

    Scans the call graph's per-symbol and per-class ``rationale`` buckets —
    design-intent comments the parser lifted out of ``src/**/*.php``. Filter
    by ``tag`` (case-insensitive exact match) and/or ``query`` (case-
    insensitive substring over the comment text). Results are grouped by
    file so related notes read together.
    """
    cg = _cache.get_call_graph()
    tag_up = tag.upper() if tag else None
    q_low = query.lower() if query else None

    # Collect (file, line, tag, text, owner) rows from symbols + classes.
    rows: list[tuple[str, int, str, str, str]] = []
    for sid, info in cg.get("symbols", {}).items():
        for n in info.get("rationale", []):
            rows.append((info.get("file", ""), n.get("line", 0), n.get("tag", ""), n.get("text", ""), sid))
    for fqcn, info in cg.get("classes", {}).items():
        for n in info.get("rationale", []):
            rows.append((info.get("file", ""), n.get("line", 0), n.get("tag", ""), n.get("text", ""), fqcn))

    filtered = [
        r for r in rows
        if (tag_up is None or r[2] == tag_up)
        and (q_low is None or q_low in (r[3] or "").lower())
    ]
    if not filtered:
        crit = []
        if tag:
            crit.append(f"tag={tag}")
        if query:
            crit.append(f"query={query!r}")
        suffix = f" matching {', '.join(crit)}" if crit else ""
        return f"No inline rationale comments found{suffix}."

    filtered.sort(key=lambda r: (r[0], r[1]))
    total = len(filtered)
    by_tag: dict[str, int] = {}
    for r in filtered:
        by_tag[r[2]] = by_tag.get(r[2], 0) + 1

    lines = [
        f"# Inline rationale ({total} comment{'s' if total != 1 else ''})",
        "- By tag: " + ", ".join(f"{k}={v}" for k, v in sorted(by_tag.items())),
        "",
    ]
    current_file = None
    for file_path, line, tag_name, text, owner in filtered[:limit]:
        if file_path != current_file:
            current_file = file_path
            lines.append(f"## {file_path}")
        owner_short = owner.rsplit("\\", 1)[-1]
        lines.append(f"- L{line} **{tag_name}** ({owner_short}): {text}")
    if total > limit:
        lines.append("")
        lines.append(f"_… {total - limit} more truncated. Narrow with tag= or query=._")
    return "\n".join(lines)


def _build_find_entity(query: str, entity_type: str | None = None) -> str:
    """Render matching mention-targets + the notes that reference each.

    Thin renderer over ``entities.find_entity_matches`` — all matching logic
    lives there so it stays unit-testable without booting the MCP server.
    """
    graph = _cache.get_unified_graph()
    matches = entities.find_entity_matches(graph, query, entity_type)
    if not (query or "").strip():
        return 'Provide a search string, e.g. `find_entity("kokoro")`.'
    if not matches:
        scope = f" of type `{entity_type}`" if entity_type else ""
        return f"No mentioned entities{scope} match `{query}`."

    header = f"# find_entity: `{query}`"
    if entity_type:
        header += f"  (type=`{entity_type}`)"
    lines = [header, f"{len(matches)} match(es)", ""]
    for m in matches[:40]:
        tag = m["entity_type"] or m["kind"]
        lines.append(f"## `{m['target']}` — {tag} · {len(m['mentioners'])} note(s)")
        for src in m["mentioners"][:30]:
            label = graph["nodes"].get(src, {}).get("label", "")
            lines.append(f"- `{src}` — {label}")
        lines.append("")
    lines.append(
        "Tip: pass any node id above to `get_unified_neighbors` for its full radius."
    )
    return "\n".join(lines)


def _build_find_community(node_id: str) -> str:
    """Report which community a given node belongs to + sibling members."""
    from scripts import communities as _comm
    from scripts.config import KNOWLEDGE_DIR

    graph = _cache.get_unified_graph()
    if node_id not in graph["nodes"]:
        return f"Node not found in unified graph: `{node_id}`"

    # min_size=2 (detect()'s default) so a lookup sees every community,
    # including small ones get_communities() filters out. Distinct cache
    # file per min_size — no thrash against get_communities(min_size=3).
    min_size = 2
    cache_path = KNOWLEDGE_DIR / f"communities.min{min_size}.json"
    clusters = _comm.load_or_compute(graph, cache_path=cache_path, min_size=min_size)
    for c in clusters:
        if node_id in c["members"]:
            lines = [
                f"# `{node_id}`",
                f"- Community: **{c['community_id']} — {c['label']}**",
                f"- Hub of this community: `{c['hub_node']}`",
                f"- Community size: {c['size']}",
                "",
                "## Sibling members",
            ]
            for nid in c["members"]:
                if nid == node_id:
                    continue
                label = graph["nodes"].get(nid, {}).get("label", "")
                lines.append(f"- `{nid}` — {label}")
            return "\n".join(lines)
    return f"`{node_id}` is not in any community (likely singleton — below min_size)."


# -----------------------------------------------------------------------------
# explore() — one call, one natural target, everything about it
# -----------------------------------------------------------------------------
#
# CodeGraph's measured advantage (88% fewer tool calls on its own benchmark) is
# that *one* call answers the question. Our granular tools each return one slice,
# so orienting on a file today costs get_file_api + get_file_deps + find_rationale,
# and orienting on a URL costs trace_route + get_file_deps. get_neighborhood is
# the graph-node analogue but needs a `symbol:`/`file:` node id — not the file
# path, URL, or class name a developer actually starts from. explore() closes
# that gap: it takes the natural target, detects its kind, and composes the
# existing _build_* helpers into a single budgeted payload. It reuses the warm
# ParseCache like every other tool, so it must NOT be wired into the cold
# per-prompt UserPromptSubmit hook.

_ROUTE_METHOD_RE = re.compile(
    r"^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(\S+)$", re.IGNORECASE
)


def _classify_target(target: str) -> tuple:
    """Detect what kind of thing ``target`` names.

    Returns one of:
        ("route", method_or_None, path)   — "POST /x", or a bare "/x"
        ("file", normalised_path)         — a .php / .twig / _controller.js path
        ("class", fqcn_or_symbol)         — contains a namespace sep or "::"
        ("name", short_name)              — a bare identifier to resolve
        ("unknown", raw)                  — nothing usable

    Order matters: route verbs and leading "/" win first; a namespace
    separator or "::" marks a class/symbol before extension checks, so an
    FQCN is never mistaken for a path.
    """
    t = target.strip()
    if not t:
        return ("unknown", target)

    m = _ROUTE_METHOD_RE.match(t)
    if m:
        return ("route", m.group(1).upper(), m.group(2))
    if t.startswith("/"):
        return ("route", None, t)

    # Normalise separators FIRST so a Windows path (``src\Entity\User.php``) is
    # recognised as a file before its backslashes are read as a namespace sep.
    # Extension match is case-insensitive (``User.PHP`` is still a PHP file).
    norm = t.replace("\\", "/").lstrip("./")
    low = norm.lower()
    if low.endswith(".php") or low.endswith(".twig") or low.endswith("_controller.js"):
        return ("file", norm)

    # FQCN / method symbol — a backslash or "::" now unambiguously means a
    # namespace, since real file-suffixed paths were already handled above.
    if "::" in t or "\\" in t:
        return ("class", t.lstrip("\\"))

    if "/" in norm and "." in norm.rsplit("/", 1)[-1]:
        # A path with an extension we don't special-case (yaml, md, …).
        return ("file", norm)

    return ("name", t)


def _resolve_class_to_file(target: str) -> tuple:
    """Resolve an FQCN, ``FQCN::method``, or bare short-name to a source file.

    Returns ``(rel_path, matched_fqcn)`` on a unique hit, ``(None, [candidates])``
    when a short-name is ambiguous, or ``(None, [])`` when nothing matches.
    Tries the call graph (method-aware) first, then the PHP graph (covers
    interfaces/traits/enums the call graph doesn't index).
    """
    t = target.lstrip("\\")
    cls = t.rsplit("::", 1)[0] if "::" in t else t
    short = cls.rsplit("\\", 1)[-1]

    classes = _cache.get_call_graph().get("classes", {})
    if cls in classes and classes[cls].get("file"):
        return classes[cls]["file"], cls

    php_nodes = _cache.get_php_graph().get("nodes", {})

    def _fqcn(node: dict) -> str:
        ns = (node.get("namespace") or "").strip("\\")
        name = node.get("class") or ""
        return f"{ns}\\{name}".strip("\\") if name else ""

    # Exact FQCN in the PHP graph.
    for rel, node in php_nodes.items():
        if _fqcn(node) == cls:
            return rel, cls

    # Short-name match across both graphs (dedup by file).
    matches: dict[str, str] = {}  # fqcn -> file
    for fqcn, info in classes.items():
        if fqcn.rsplit("\\", 1)[-1] == short and info.get("file"):
            matches[fqcn] = info["file"]
    for rel, node in php_nodes.items():
        if node.get("class") == short:
            matches.setdefault(_fqcn(node) or short, rel)

    if len(matches) == 1:
        fqcn, rel = next(iter(matches.items()))
        return rel, fqcn
    if len(matches) > 1:
        return None, sorted(matches)

    # Case-insensitive fallback — PHP class names are case-insensitive, so
    # ``accesscontrolservice`` should still resolve. Exact matches above win;
    # this only runs when nothing matched with exact case.
    cls_low, short_low = cls.lower(), short.lower()
    ci: dict[str, str] = {}
    for fqcn, info in classes.items():
        if info.get("file") and (
            fqcn.lower() == cls_low or fqcn.rsplit("\\", 1)[-1].lower() == short_low
        ):
            ci[fqcn] = info["file"]
    for rel, node in php_nodes.items():
        fq = _fqcn(node)
        if fq.lower() == cls_low or (node.get("class") or "").lower() == short_low:
            ci.setdefault(fq or short, rel)
    if len(ci) == 1:
        fqcn, rel = next(iter(ci.items()))
        return rel, fqcn
    if len(ci) > 1:
        return None, sorted(ci)
    return None, []


def _related_articles_section(seed_ids: list[str], limit: int = 8) -> str:
    """Render a compact 'Related knowledge' section for the given graph seeds.

    Walks the unified graph two hops (undirected) from each seed node id
    (``file:...`` / ``class:...`` / ``symbol:...``) and lists the article
    nodes it reaches — the decision/why layer for this code. Bodies are not
    inlined; the agent calls ``get_articles`` for those. Always renders the
    heading (with an explicit 'none found' line) so callers can rely on it.
    """
    graph = _cache.get_unified_graph()
    nodes, edges = graph["nodes"], graph["edges"]
    seeds = [s for s in seed_ids if s in nodes]

    lines = ["## Related knowledge"]
    if not seeds:
        lines.append("_No graph node for this target — none found._")
        return "\n".join(lines)

    adj: dict[str, set[str]] = {}
    for e in edges:
        adj.setdefault(e["from"], set()).add(e["to"])
        adj.setdefault(e["to"], set()).add(e["from"])

    visited = set(seeds)
    frontier = set(seeds)
    for _ in range(2):
        nxt = {t for n in frontier for t in adj.get(n, set())} - visited
        visited |= nxt
        frontier = nxt

    articles = sorted(n for n in visited if nodes.get(n, {}).get("kind") == "article")
    if not articles:
        lines.append("_None found in the knowledge graph._")
        return "\n".join(lines)

    lines.append("_Call `get_articles` for full bodies._")
    for aid in articles[:limit]:
        n = nodes[aid]
        lines.append(
            f"- `{aid}` — **{n.get('label', '')}** "
            f"(type={n.get('type', '?')}, conf={n.get('confidence', '?')})"
        )
    if len(articles) > limit:
        lines.append(f"- … and {len(articles) - limit} more")
    return "\n".join(lines)


def _build_explore(target: str, max_depth: int = 4) -> str:
    """One call, one natural target, everything about it.

    ``target`` may be a file path (``src/Foo.php``, ``foo/x.html.twig``,
    ``bar_controller.js``), a route (``POST /api/x`` or a bare ``/api/x``),
    an FQCN (``App\\Service\\Foo``), a method symbol (``App\\Service\\Foo::bar``),
    or a bare class short-name (``Foo``). The output bundles the slices the
    granular tools return one at a time:

    - **file** → API surface + dependency/route/git/rationale view + related KB
    - **route** → call-tree trace + related KB for the controller file
    - **class/name** → resolved to its file, then the file view

    Deliberately budgeted, not exhaustive: it composes existing builders and
    appends the knowledge layer, so the agent trades ~3 calls (and ~3 permission
    prompts) for one. Drill deeper with the granular tools when a slice is
    truncated.
    """
    kind, *rest = _classify_target(target)

    if kind == "unknown":
        return (
            "# Explore\n\n"
            f"Could not tell what `{target}` refers to. Pass a file path "
            "(`src/Foo.php`), a route (`POST /api/x` or `/api/x`), an FQCN "
            "(`App\\Service\\Foo`), or a class name (`Foo`)."
        )

    if kind == "route":
        method, path = rest
        routes = _cache.get_route_map()
        entry = routes["routes"].get(path)
        if entry is None:
            return (
                f"# Explore (route)\n\nNo route found at path `{path}`. "
                "Check the exact Symfony path, or call `get_route_map(prefix)` "
                "to list candidates."
            )
        if method is None:
            method = entry["methods"][0] if entry["methods"] else "GET"
        elif entry["methods"] and method not in entry["methods"]:
            # Reject the verb up front — otherwise the success header names a
            # controller the trace then says does not handle this method.
            return (
                f"# Explore: `{method} {path}` (detected: route)\n\n"
                f"Route `{path}` does not handle `{method}` — it handles "
                f"{', '.join(entry['methods'])}. Re-run `explore` with one of those verbs."
            )

        header = (
            f"# Explore: `{method} {path}` (detected: route)\n\n"
            f"Controller `{entry['controller']}::{entry['action']}` "
            f"— `{entry['file']}`\n"
        )
        trace = _build_trace_route(method, path, max_depth=max_depth)
        related = _related_articles_section([f"file:{entry['file']}"])
        return f"{header}\n{trace}\n\n{related}"

    if kind in ("class", "name"):
        orig = target
        rel, matched = _resolve_class_to_file(orig)
        if rel is None:
            if matched:  # ambiguous short-name
                cand = "\n".join(f"- `{c}`" for c in matched)
                return (
                    f"# Explore\n\n`{orig}` is ambiguous — "
                    f"{len(matched)} classes share that name:\n{cand}\n\n"
                    "Re-run `explore` with the full namespace."
                )
            return (
                f"# Explore\n\nNo class or file found for `{orig}`. "
                "Try the exact FQCN, or `search_codebase(query)` to locate it."
            )
        note = f"_Resolved `{matched}` → `{rel}`._\n\n"
        # If a method symbol was given, verify the method actually exists —
        # otherwise a typo'd `::method` would look like a clean class hit.
        if "::" in orig:
            method_name = orig.rsplit("::", 1)[1]
            sym = f"{matched}::{method_name}"
            if method_name and sym not in _cache.get_call_graph().get("symbols", {}):
                note += (
                    f"_Note: method `{method_name}` was not found on `{matched}` "
                    "— showing the whole class._\n\n"
                )
        body = _explore_file(rel, header_kind="class")
        return note + body

    # kind == "file"
    return _explore_file(rest[0], header_kind="file")


def _explore_file(file_path: str, header_kind: str = "file") -> str:
    """Compose the one-call file view: API surface + deps + related knowledge."""
    file_path = file_path.replace("\\", "/").lstrip("./")
    header = f"# Explore: `{file_path}` (detected: {header_kind})\n"

    seeds = [f"file:{file_path}"]

    if file_path.endswith(".php"):
        api = _build_file_api(file_path)
        deps = _build_file_deps(file_path)
        # Seed related-articles from the file AND its classes so articles that
        # cite a class (not the bare file node) are still reached.
        graph = _cache.get_call_graph()
        for fqcn, info in graph.get("classes", {}).items():
            if info.get("file") == file_path:
                seeds.append(f"class:{fqcn}")
        related = _related_articles_section(seeds)
        return f"{header}\n{api}\n\n---\n\n{deps}\n\n{related}"

    # Twig: _build_file_deps keys templates by their `templates/`-prefixed
    # path, while the unified graph node is `template:<path-without-prefix>`.
    if file_path.endswith(".twig"):
        tpl = file_path if file_path.startswith("templates/") else f"templates/{file_path}"
        bare = tpl.removeprefix("templates/")
        deps = _build_file_deps(tpl)
        # An article may cite the template as either a `template:` node or a
        # `file:templates/...` node — seed both so neither citation is dropped.
        related = _related_articles_section([f"template:{bare}", f"file:{tpl}"])
        return f"{header}\n{deps}\n\n{related}"

    # Stimulus / other: deps view carries the useful structure.
    deps = _build_file_deps(file_path)
    related = _related_articles_section(seeds)
    return f"{header}\n{deps}\n\n{related}"


# -----------------------------------------------------------------------------
# FastMCP server bindings
# -----------------------------------------------------------------------------


def _make_server():
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("symfony-code-intel")

    @server.tool()
    def explore(target: str, max_depth: int = 4) -> str:
        """One call, one natural target, everything about it — start here.

        The single-call orienting tool. Give it what you already have in hand
        and it detects the kind and bundles what the granular tools return one
        slice at a time, so you trade ~3 calls (and ~3 permission prompts) for
        one:

        - **File path** (``src/Service/Foo.php``, ``arena/index.html.twig``,
          ``arena_controller.js``) → API surface + dependency/route/git/inline-
          rationale view + related knowledge articles.
        - **Route** (``POST /api/session/start`` or a bare ``/api/session/start``)
          → the controller-to-services call-tree trace + related knowledge.
        - **FQCN / method symbol / class name** (``App\\Service\\Foo``,
          ``App\\Service\\Foo::bar``, or just ``Foo``) → resolved to its file,
          then the file view. Ambiguous short-names list their candidates.

        Deliberately budgeted, not a dump: it composes existing builders. When a
        slice is truncated or you need one specific angle cheaply, reach for the
        granular tool (``get_file_api``, ``get_file_deps``, ``trace_route``,
        ``find_rationale``, ``impact_of_change``, ``get_articles``).

        Args:
            target: a file path, route, FQCN, method symbol, or class name.
            max_depth: call-tree depth for the route case (default 4).
        """
        return _build_explore(target, max_depth=max_depth)

    @server.tool()
    def get_codebase_overview() -> str:
        """Full overview: file counts by type, route counts, template counts, Stimulus stats, top git hotspots."""
        return _build_codebase_overview()

    @server.tool()
    def get_file_deps(file_path: str) -> str:
        """Dependencies for a specific file. Handles PHP, Twig, and Stimulus JS files."""
        return _build_file_deps(file_path)

    @server.tool()
    def get_file_api(file_path: str) -> str:
        """API surface of a file: class + method signatures, NO bodies — token-cheap.

        Read this before opening a PHP/JS file in full when you only need its
        shape (what methods exist, their params, return types, and — for
        controllers — the ``#[Route]`` binding). Covers PHP classes and Stimulus
        controller JS. For Twig use get_template_graph.
        """
        return _build_file_api(file_path)

    @server.tool()
    def get_route_map(prefix: str = "") -> str:
        """Symfony route -> controller -> service table, filtered by optional URL prefix."""
        return _build_route_map(prefix)

    @server.tool()
    def get_template_graph(template: str = "") -> str:
        """Twig template inheritance + includes. If `template` given, details for that file."""
        return _build_template_graph(template)

    @server.tool()
    def get_stimulus_map(controller: str = "") -> str:
        """Stimulus controller <-> template links. If `controller` given, details for that controller."""
        return _build_stimulus_map(controller)

    @server.tool()
    def get_hotspots(top_n: int = 10) -> str:
        """Top N hot files ranked by git churn score, with co-change partners and ownership."""
        return _build_hotspots(top_n)

    @server.tool()
    def get_unified_neighbors(node_id: str, depth: int = 1) -> str:
        """Neighbors of an article/file/class/symbol/template node in the unified graph.

        Node IDs use prefixes: ``article:concepts/foo``, ``file:src/Foo.php``,
        ``class:App\\Service\\Foo``, ``symbol:App\\Service\\Foo::bar``,
        ``template:foo/index.html.twig``. ``depth`` is clamped to ``[1, 3]``.

        Use this to answer questions like "what code does this article describe?"
        (article → cites → file → contains → class → defines → symbol — reachable
        within depth=3) without chaining ``search_codebase`` after ``get_article``.
        """
        return _build_unified_neighbors(node_id, depth)

    @server.tool()
    def get_neighborhood(
        node_id: str,
        depth: int = 1,
        include_source: bool = True,
        max_source_lines: int = 200,
    ) -> str:
        """Neighbors + source for ``symbol:`` nodes grouped by file, in one call.

        Combines ``get_unified_neighbors`` with source extraction for every
        ``symbol:`` node in the neighborhood, grouped under its owning file's
        heading. Bare ``file:`` nodes (Twig, YAML, plain-function PHP — no
        classes extracted) get a 40-line header excerpt. Articles in the
        neighborhood are listed (not inlined — use ``get_articles`` for those).

        Use when orienting in an unfamiliar area instead of chaining
        ``get_unified_neighbors`` → N × ``Read``.

        - ``depth``: 1–3 (clamped), same as ``get_unified_neighbors``.
        - ``include_source=False``: structure-only view, ~10× cheaper.
        - ``max_source_lines``: per-symbol cap; total output capped at 5×.
          Truncation is announced inline.
        """
        return _build_neighborhood(node_id, depth, include_source, max_source_lines)

    @server.tool()
    def trace_path(from_node: str, to_node: str, max_depth: int = 8) -> str:
        """Find the shortest connection between two nodes in the unified graph.

        Answers "how is X related to Y?" — e.g. how a controller symbol
        connects to a knowledge article, or how two articles relate through
        shared code. Complements ``get_unified_neighbors`` (which shows the
        radius around one node) by returning the actual chain *between* two.

        Node IDs use the same prefixes as ``get_unified_neighbors``:
        ``article:concepts/foo``, ``file:src/Foo.php``,
        ``class:App\\Service\\Foo``, ``symbol:App\\Service\\Foo::bar``,
        ``template:foo.html.twig``, ``note:src/Foo.php:42``.

        Args:
            from_node / to_node: fully-qualified node IDs.
            max_depth: hop budget, clamped to ``[1, 12]`` (default 8).

        Returns:
            A numbered hop-by-hop path with each step annotated by the edge
            kind (cites / wikilink / call / defines / annotates / …) and its
            relation or confidence, or a friendly "no path" message.
        """
        return _build_trace_path(from_node, to_node, max_depth)

    @server.tool()
    def get_communities(min_size: int = 3, top_n: int = 10) -> str:
        """List the top-N semantic communities in the unified knowledge graph.

        Communities are computed via Leiden modularity optimization over the
        union of article wikilinks + code call graph. Each community returns
        ``community_id``, ``size``, the highest-degree ``hub_node``, a
        deterministic ``label`` assembled from member node labels, and up to
        8 sample members.

        ``min_size`` filters out tiny clusters (default 3); ``top_n`` caps the
        result count (default 10). The cache lives at
        ``knowledge/communities.json`` and is invalidated automatically when
        the underlying graph changes.
        """
        return _build_communities(min_size, top_n)

    @server.tool()
    def find_community(node_id: str) -> str:
        """Find the semantic community a given node belongs to + list its siblings.

        ``node_id`` uses the same prefix convention as
        ``get_unified_neighbors`` (``article:...``, ``file:...``, ``class:...``,
        ``symbol:...``, ``template:...``). Returns a friendly error string
        if the node is not in the graph or is below the singleton threshold.
        """
        return _build_find_community(node_id)

    @server.tool()
    def find_entity(query: str, entity_type: str = "") -> str:
        """Find named things the KB talks about, and the notes that mention them.

        Entities — hosts, project commands, Symfony roles, env vars, URLs,
        services, source paths — are lifted from article PROSE by the extractor
        and joined to the articles that name them via ``mentions`` edges. This
        fuzzy-matches ``query`` (case-insensitive substring over node id +
        label) and lists, per match, the notes that reference it. It is the
        "show me every note about X" lookup that ``get_unified_neighbors`` needs
        an exact node id for.

        Searches minted ``entity:`` nodes AND the ``class:``/``file:`` code
        nodes that prose service/path mentions folded into, so a service with a
        real class is found too. ``entity_type`` (optional) — one of
        ``host``/``command``/``role``/``envvar``/``url``/``service``/``path`` —
        restricts to minted ``entity:`` nodes of that type; folded code targets
        appear only when ``entity_type`` is empty (reach those via the
        code-intel tools or ``get_unified_neighbors`` on the code node).
        """
        return _build_find_entity(query, entity_type or None)

    @server.tool()
    def get_salience(kind: str = "all", top_n: int = 10, scope: str = "all") -> str:
        """Rank the unified graph: what carries it, and which links are surprising.

        The complement to ``get_unified_neighbors`` / ``find_community``, both
        of which need you to already know a node id. This one tells you which
        node to ask about.

        Args:
            kind: ``all`` (default) or one section —
                ``hubs`` (highest degree — what the KB mostly talks about),
                ``brokers`` (highest betweenness — the load-bearing nodes;
                removing one disconnects parts of the graph),
                ``bridges`` (**surprising connections** — edges joining two
                communities that have almost no other link and whose
                endpoints share no common neighbour; cut one and the two
                clusters are nearly severed. Split into Article↔Article,
                Code↔Code and Article↔Code, each with its own density
                threshold),
                ``orphans`` (articles with degree ≤ 1 — written but
                unreachable by neighbour/community retrieval).
            top_n: Rows per table (default 10).
            scope: ``all``, ``article``, or ``code``. Centrality is always
                computed over the whole graph and only *presented* per scope —
                an article's brokerage often runs through the code it cites.

        Zero LLM cost. Cached at ``knowledge/salience.json`` and invalidated
        automatically when the graph changes.
        """
        return _build_salience(kind, top_n, scope)

    @server.tool()
    def find_rationale(tag: str | None = None, query: str | None = None) -> str:
        """Surface inline design-intent comments across the PHP codebase.

        The parser lifts rationale comments — ``// WHY:``, ``// HACK``,
        ``// TODO:``, ``// FIXME``, ``/** @deprecated */``, and similar — out
        of ``src/**/*.php`` and attaches them to the method or class they
        annotate. This tool lists them so you can answer "where are the known
        hacks / deprecations / TODOs?" without grepping, and understand *why*
        a piece of code is the way it is before changing it.

        Recognized tags: WHY, HACK, NOTE, TODO, FIXME, XXX, BUG, WARNING,
        OPTIMIZE, DEPRECATED (``@deprecated`` folds into DEPRECATED).

        Args:
            tag: restrict to one tag (case-insensitive), e.g. ``HACK`` or
                ``DEPRECATED``. Omit for all tags.
            query: case-insensitive substring filter over the comment text.

        Returns:
            Markdown grouped by file: ``L<line> **TAG** (owner): text``,
            with a per-tag count header.
        """
        return _build_find_rationale(tag=tag, query=query)

    @server.tool()
    def impact_of_change(
        file: str | None = None,
        since_ref: str = "HEAD",
        max_depth: int = 6,
        output_format: str = "text",
    ) -> str:
        """Reverse-walk the call graph from edited symbols to surface affected routes + risk score.

        Pass ``file`` to scope the diff to a single path. ``since_ref`` defaults
        to ``HEAD`` (working tree vs latest commit); use a branch name like
        ``main`` to see what your branch impacts.

        Set ``output_format='mermaid'`` for a flowchart rendering of the
        affected-route graph — easier to read at a glance than the default
        risk-sorted bullet list once more than 4-5 routes are affected.
        """
        return _build_impact_of_change(file, since_ref, max_depth, output_format)

    @server.tool()
    def trace_route(
        method: str,
        path: str,
        max_depth: int = 6,
        output_format: str = "text",
        collapse_accessors: bool = True,
    ) -> str:
        """Trace the call graph from a route's controller action down through services + repositories.

        Resolves constructor-injected services, static calls, typed locals, and
        Doctrine ``getRepository(X::class)`` chains. Templates rendered via
        ``$this->render()`` appear as leaves marked ``[render]``.

        ``collapse_accessors`` (default True) folds childless getter/setter
        calls to ``\\Entity\\`` classes (``User::setEmail`` …) into one summary
        line per parent, so the architecturally-meaningful service/repository
        hops aren't buried. Set False for the fully-expanded tree.

        Set ``output_format='mermaid'`` to receive a ``flowchart TD`` block
        instead of an indented bullet tree — the diagram form scales much
        better past depth 3.
        """
        return _build_trace_route(method, path, max_depth, output_format, collapse_accessors)

    @server.tool()
    def find_message_handlers(message: str | None = None) -> str:
        """Map Symfony Messenger messages to their handlers and dispatch sites.

        The async equivalent of ``trace_route`` for the message bus. Answers
        "which handler runs this message?", "who dispatches it?", and flags
        contract gaps a plain grep misses:

        - **Unhandled** — a message dispatched via ``->dispatch(new X())`` that
          no ``#[AsMessageHandler]`` consumes (a real wiring bug).
        - **Undispatched** — a handler with no literal dispatch site found
          (either genuinely unused, or dispatched through a variable).

        Args:
            message: substring of a message short name or FQCN (e.g.
                ``GradeSession``). Omit for the whole-bus overview + orphans.

        Returns:
            Markdown. Overview lists each handled message → handler(s) →
            dispatch-site count, then the orphan sections. Filtered mode lists
            every handler and every ``file:line`` dispatch site for the match.

        Detection is regex over ``src/**/*.php`` (class-level and method-level
        ``#[AsMessageHandler]``, explicit ``handles:`` args, and
        ``dispatch``/``dispatchAfterCurrentBus`` producer calls). Only messenger
        messages (``App\\Message\\`` namespace, or a class with a handler) are
        tracked, so EventDispatcher ``->dispatch(new App\\Event\\…)`` calls do
        not pollute the results.
        """
        return _build_find_message_handlers(message)

    @server.tool()
    def merge_order_risk(
        base: str = "main",
        branches: list[str] | None = None,
    ) -> str:
        """Assess which in-flight branches will collide, and in what order to merge.

        Purpose-built for this repo's reality: a chronically-dirty ``main`` and
        several feature branches / worktrees open at once. Surfaces two risks
        plain ``git`` can't rank for you:

        - **Direct file conflicts** — the same file changed on 2+ branches
          (git *will* conflict; merge those back-to-back).
        - **Community-overlap risk** — different files in the same semantic
          cluster (Leiden community over the unified graph): no textual
          conflict, but a coupling risk worth reviewing together. This is the
          signal that catches "these two branches both reshape the billing
          subsystem from different files."

        Args:
            base: branch to compare against (default ``main``). Use ``master``
                for the prod mainline in this repo if that's your target.
            branches: explicit branch names to compare. Omit to auto-detect
                every local branch with commits ahead of ``base`` (capped 12).

        Returns:
            A markdown report: per-branch summary, direct conflicts (merge
            these adjacent), and community-overlap pairs (review together,
            smaller change first).
        """
        return _build_merge_order_risk(base=base, branches=branches)

    @server.tool()
    def get_circular_dependencies(
        scope: str = "vendor-excluded",
        output_format: str = "text",
    ) -> str:
        """Detect strongly-connected components (cycles) in the call graph.

        Tarjan's algorithm over the resolved call graph. Useful before a
        refactor to find self-reinforcing dependencies that will trip
        Symfony's container compile pass — or to verify that a refactor
        actually broke a cycle you intended to break.

        Args:
            scope: ``all`` | ``php`` | ``js`` | ``vendor-excluded`` (default).
                ``vendor-excluded`` is the recommended scope: keeps only
                ``src/...`` PHP symbols so vendor-internal cycles
                (Symfony, Doctrine) don't drown out your own.
            output_format: ``text`` (default) — markdown listing per cycle
                with file:line per symbol — or ``mermaid`` — one
                ``flowchart LR`` per cycle.

        Returns:
            Markdown report. Cycles are sorted by size (largest first),
            then by leading symbol name.
        """
        return _build_circular_dependencies(scope, output_format)

    return server


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parent_watchdog.start()
    server = _make_server()
    server.run()


if __name__ == "__main__":
    main()
