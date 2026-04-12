"""PHP dependency graph parser.

Scans src/**/*.php, extracts namespace + class + `use App\\...` imports
via regex, builds a directed graph of internal dependencies, and
classifies files by Symfony domain (controller, service, entity, ...).
"""
from __future__ import annotations

import re
from pathlib import Path

# Matches: namespace App\Controller\Api;
_NAMESPACE_RE = re.compile(r"^\s*namespace\s+(App\\[\w\\]+)\s*;", re.MULTILINE)

# Matches: class FooController extends AbstractController {
# Also matches: final class FooController, abstract class FooController, readonly class ...
_CLASS_RE = re.compile(
    r"^\s*(?:final\s+|abstract\s+|readonly\s+)*class\s+(\w+)",
    re.MULTILINE,
)

# Matches: use App\Service\Session\SessionService;
# Also matches aliases: use App\Foo\Bar as Baz;
# We capture the FQCN only, not the alias.
_USE_RE = re.compile(
    r"^\s*use\s+(App\\[\w\\]+)(?:\s+as\s+\w+)?\s*;",
    re.MULTILINE,
)


def _classify_by_path(rel_path: str) -> str:
    """Classify a PHP file by its path segment."""
    p = rel_path.replace("\\", "/")
    if "/MessageHandler/" in p:
        return "handler"
    if "/Message/" in p:
        return "message"
    if "/Controller/" in p:
        return "controller"
    if "/Service/" in p or p.startswith("src/Service/"):
        return "service"
    if "/Entity/" in p:
        return "entity"
    if "/Repository/" in p:
        return "repository"
    if "/Command/" in p:
        return "command"
    if "/EventSubscriber/" in p:
        return "subscriber"
    if "/Form/" in p:
        return "form"
    if "/Validator/" in p:
        return "validator"
    if "/Enum/" in p:
        return "enum"
    if "/DTO/" in p:
        return "dto"
    if "/Security/" in p:
        return "security"
    if "/Twig/" in p:
        return "twig_extension"
    if "/DataFixtures/" in p:
        return "fixture"
    if "/Exception/" in p:
        return "exception"
    return "other"


def _fqcn_to_rel_path(fqcn: str) -> str:
    """Convert App\\Foo\\Bar → src/Foo/Bar.php (Symfony PSR-4 convention)."""
    # Drop leading "App\"
    without_prefix = fqcn[4:] if fqcn.startswith("App\\") else fqcn
    parts = without_prefix.split("\\")
    return "src/" + "/".join(parts) + ".php"


def parse(project_root: Path) -> dict:
    """Build the PHP dependency graph for src/**/*.php."""
    src_dir = project_root / "src"
    nodes: dict[str, dict] = {}
    raw_imports: dict[str, list[str]] = {}  # path → list of FQCNs

    for php_file in src_dir.rglob("*.php"):
        rel_path = php_file.relative_to(project_root).as_posix()
        try:
            content = php_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        ns_match = _NAMESPACE_RE.search(content)
        class_match = _CLASS_RE.search(content)
        uses = _USE_RE.findall(content)

        nodes[rel_path] = {
            "namespace": ns_match.group(1) if ns_match else "",
            "class": class_match.group(1) if class_match else "",
            "type": _classify_by_path(rel_path),
            "imports": uses,
            "in_degree": 0,
            "out_degree": 0,
        }
        raw_imports[rel_path] = uses

    # Build edges — translate FQCNs to rel paths, keep only internal hits
    edges: list[dict] = []
    for src_path, uses in raw_imports.items():
        for fqcn in uses:
            target_path = _fqcn_to_rel_path(fqcn)
            if target_path in nodes:
                edges.append({"from": src_path, "to": target_path, "names": [fqcn]})
                nodes[src_path]["out_degree"] += 1
                nodes[target_path]["in_degree"] += 1

    # Aggregate stats
    by_type: dict[str, int] = {}
    for node in nodes.values():
        t = node["type"]
        by_type[t] = by_type.get(t, 0) + 1

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_files": len(nodes),
            "by_type": by_type,
        },
    }


def summary(project_root: Path) -> str:
    """Fast summary — counts files via glob, no parsing."""
    src_dir = project_root / "src"
    if not src_dir.is_dir():
        return "no src/ directory"

    counts = {"controller": 0, "service": 0, "entity": 0, "repository": 0, "other": 0}
    total = 0
    for php_file in src_dir.rglob("*.php"):
        rel = php_file.relative_to(project_root).as_posix()
        t = _classify_by_path(rel)
        counts[t if t in counts else "other"] += 1
        total += 1

    return (
        f"{total} files "
        f"({counts['entity']} entities, {counts['controller']} controllers, "
        f"{counts['service']} services, {counts['repository']} repos)"
    )
