"""Twig template dependency graph parser.

Scans templates/**/*.twig for extends/include/embed/render directives and
builds parent→child inheritance chains. Also detects Stimulus controllers
referenced via data-controller attributes and {{ stimulus_controller(...) }}.
"""
from __future__ import annotations

import re
from pathlib import Path

# {% extends 'base.html.twig' %}
_EXTENDS_RE = re.compile(r"{%\s*extends\s+['\"]([^'\"]+)['\"]\s*%}")

# {% include 'foo.html.twig' %}
_INCLUDE_TAG_RE = re.compile(r"{%\s*include\s+['\"]([^'\"]+)['\"]")

# {% embed 'foo.html.twig' %}
_EMBED_RE = re.compile(r"{%\s*embed\s+['\"]([^'\"]+)['\"]")

# {{ include('foo.html.twig') }}
_INCLUDE_FN_RE = re.compile(r"\{\{\s*include\s*\(\s*['\"]([^'\"]+)['\"]")

# data-controller="foo" or data-controller="foo bar"
_DATA_CONTROLLER_RE = re.compile(r'data-controller\s*=\s*["\']([^"\']+)["\']')

# {{ stimulus_controller('foo', { ... }) }}
_STIMULUS_CTRL_FN_RE = re.compile(r"stimulus_controller\s*\(\s*['\"]([^'\"]+)['\"]")


def _resolve_template_path(ref: str, all_templates: set[str]) -> str:
    """Map 'arena/index.html.twig' → 'templates/arena/index.html.twig' if it exists.

    Twig refs can also start with @Bundle/... — we ignore those as external.
    """
    if ref.startswith("@"):
        return ref  # external bundle — leave as-is
    candidate = f"templates/{ref}"
    if candidate in all_templates:
        return candidate
    return candidate  # return expected path even if not found (so we surface typos)


def parse(project_root: Path) -> dict:
    """Build the Twig template graph."""
    templates_dir = project_root / "templates"
    templates: dict[str, dict] = {}

    # First pass: discover all template files
    all_twig_files = list(templates_dir.rglob("*.twig"))
    all_template_paths: set[str] = {
        f.relative_to(project_root).as_posix() for f in all_twig_files
    }

    # Second pass: parse each template
    for twig_file in all_twig_files:
        rel_path = twig_file.relative_to(project_root).as_posix()
        try:
            content = twig_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        extends_match = _EXTENDS_RE.search(content)
        extends = (
            _resolve_template_path(extends_match.group(1), all_template_paths)
            if extends_match
            else None
        )

        includes: list[str] = []
        for rx in (_INCLUDE_TAG_RE, _EMBED_RE, _INCLUDE_FN_RE):
            for m in rx.finditer(content):
                resolved = _resolve_template_path(m.group(1), all_template_paths)
                if resolved not in includes:
                    includes.append(resolved)

        # Collect Stimulus controllers (direct attributes + Twig helpers)
        stim_controllers: list[str] = []
        for m in _DATA_CONTROLLER_RE.finditer(content):
            for name in m.group(1).split():
                if name and name not in stim_controllers:
                    stim_controllers.append(name)
        for m in _STIMULUS_CTRL_FN_RE.finditer(content):
            name = m.group(1)
            if name and name not in stim_controllers:
                stim_controllers.append(name)

        templates[rel_path] = {
            "extends": extends,
            "includes": includes,
            "included_by": [],   # filled below
            "rendered_by": [],   # filled by route_map integration (MCP-side)
            "stimulus_controllers": stim_controllers,
        }

    # Third pass: fill in included_by (reverse of includes)
    for path, info in templates.items():
        for child_path in info["includes"]:
            if child_path in templates:
                templates[child_path]["included_by"].append(path)

    # Build inheritance chains: parent → [children]
    inheritance: dict[str, list[str]] = {}
    for path, info in templates.items():
        parent = info["extends"]
        if parent:
            inheritance.setdefault(parent, []).append(path)

    return {
        "templates": templates,
        "inheritance_chains": inheritance,
        "stats": {
            "total_templates": len(templates),
        },
    }


def summary(project_root: Path) -> str:
    """Fast summary — count .twig files via glob."""
    templates_dir = project_root / "templates"
    if not templates_dir.is_dir():
        return "no templates/"
    count = sum(1 for _ in templates_dir.rglob("*.twig"))
    return f"{count} Twig files"
