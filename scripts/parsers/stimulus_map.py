"""Stimulus controller ↔ template bidirectional map.

Scans assets/controllers/*_controller.js for controller files and
templates/**/*.twig for references. Controller name is derived from
the filename: admin_cost_charts_controller.js → admin-cost-charts.
"""
from __future__ import annotations

import re
from pathlib import Path

# data-controller="foo" or data-controller="foo bar baz"
_DATA_CONTROLLER_RE = re.compile(r'data-controller\s*=\s*["\']([^"\']+)["\']')

# {{ stimulus_controller('foo', ...) }}
_STIMULUS_CTRL_FN_RE = re.compile(r"stimulus_controller\s*\(\s*['\"]([^'\"]+)['\"]")

# static values = { ... }    (declared on class)
_VALUES_RE = re.compile(r"static\s+values\s*=\s*\{([^}]*)\}", re.DOTALL)

# static targets = [ ... ]
_TARGETS_RE = re.compile(r"static\s+targets\s*=\s*\[([^\]]*)\]", re.DOTALL)

# static outlets = [ ... ]
_OUTLETS_RE = re.compile(r"static\s+outlets\s*=\s*\[([^\]]*)\]", re.DOTALL)

# Individual quoted items: "foo" or 'bar'
_QUOTED_ITEM_RE = re.compile(r"['\"]([\w-]+)['\"]")

# value keys: fooBar: String, ...
_VALUE_KEY_RE = re.compile(r"(\w+)\s*:\s*\w+")


def _filename_to_controller_name(filename: str) -> str:
    """admin_cost_charts_controller.js → admin-cost-charts."""
    stem = filename[:-len("_controller.js")] if filename.endswith("_controller.js") else filename
    return stem.replace("_", "-")


def parse(project_root: Path) -> dict:
    """Build the Stimulus controller ↔ template map."""
    controllers_dir = project_root / "assets" / "controllers"
    templates_dir = project_root / "templates"

    controllers: dict[str, dict] = {}

    # First: discover JS controller files
    if controllers_dir.is_dir():
        for js_file in controllers_dir.glob("*_controller.js"):
            name = _filename_to_controller_name(js_file.name)
            try:
                content = js_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                content = ""

            values: list[str] = []
            targets: list[str] = []
            outlets: list[str] = []

            vm = _VALUES_RE.search(content)
            if vm:
                values = list(dict.fromkeys(_VALUE_KEY_RE.findall(vm.group(1))))

            tm = _TARGETS_RE.search(content)
            if tm:
                targets = list(dict.fromkeys(_QUOTED_ITEM_RE.findall(tm.group(1))))

            om = _OUTLETS_RE.search(content)
            if om:
                outlets = list(dict.fromkeys(_QUOTED_ITEM_RE.findall(om.group(1))))

            controllers[name] = {
                "file": js_file.relative_to(project_root).as_posix(),
                "used_in": [],
                "values": values,
                "targets": targets,
                "outlets": outlets,
            }

    # Second: scan templates for references
    referenced: set[str] = set()
    if templates_dir.is_dir():
        for twig_file in templates_dir.rglob("*.twig"):
            rel_path = twig_file.relative_to(project_root).as_posix()
            try:
                content = twig_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            names_in_file: set[str] = set()
            for m in _DATA_CONTROLLER_RE.finditer(content):
                for name in m.group(1).split():
                    if name:
                        names_in_file.add(name)
            for m in _STIMULUS_CTRL_FN_RE.finditer(content):
                names_in_file.add(m.group(1))

            for name in names_in_file:
                referenced.add(name)
                if name in controllers:
                    controllers[name]["used_in"].append(rel_path)

    # Compute orphans (JS file, no template usage) and missing (template uses, no JS file)
    orphan = [name for name, info in controllers.items() if not info["used_in"]]
    missing = sorted(referenced - set(controllers.keys()))

    total_usages = sum(len(info["used_in"]) for info in controllers.values())

    return {
        "controllers": controllers,
        "orphan_controllers": sorted(orphan),
        "missing_controllers": missing,
        "stats": {
            "total_controllers": len(controllers),
            "total_usages": total_usages,
            "orphan_count": len(orphan),
            "missing_count": len(missing),
        },
    }


def summary(project_root: Path) -> str:
    """Fast summary — count JS controller files via glob."""
    controllers_dir = project_root / "assets" / "controllers"
    if not controllers_dir.is_dir():
        return "no Stimulus controllers"
    count = sum(1 for _ in controllers_dir.glob("*_controller.js"))
    return f"{count} controllers"
