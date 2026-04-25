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

from scripts.parsers import PROJECT_ROOT, php_graph, route_map, twig_graph, stimulus_map, git_intel
from scripts import parent_watchdog

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


# -----------------------------------------------------------------------------
# FastMCP server bindings
# -----------------------------------------------------------------------------


def _make_server():
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("symfony-code-intel")

    @server.tool()
    def get_codebase_overview() -> str:
        """Full overview: file counts by type, route counts, template counts, Stimulus stats, top git hotspots."""
        return _build_codebase_overview()

    @server.tool()
    def get_file_deps(file_path: str) -> str:
        """Dependencies for a specific file. Handles PHP, Twig, and Stimulus JS files."""
        return _build_file_deps(file_path)

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

    return server


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parent_watchdog.start()
    server = _make_server()
    server.run()


if __name__ == "__main__":
    main()
