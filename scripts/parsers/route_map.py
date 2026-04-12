"""Symfony route map parser.

Extracts #[Route(...)] attributes from src/Controller/**/*.php, resolving
class-level prefixes and method-level suffixes. Links routes to the
templates they render and the services they inject.
"""
from __future__ import annotations

import re
from pathlib import Path

# Matches: #[Route('/path', name: 'foo', methods: ['GET','POST'])]
# Named arguments are optional and can appear in any order.
_ROUTE_ATTR_RE = re.compile(
    r"#\[Route\s*\(\s*(['\"])(?P<path>[^'\"]*)\1"
    r"(?P<rest>[^\]]*)"
    r"\]",
    re.DOTALL,
)

# Grab name: 'foo' from the rest of the attribute args
_NAME_RE = re.compile(r"name\s*:\s*['\"]([^'\"]+)['\"]")

# Grab methods: ['GET', 'POST'] or methods: 'GET'
_METHODS_LIST_RE = re.compile(r"methods\s*:\s*\[([^\]]*)\]")
_METHODS_SINGLE_RE = re.compile(r"methods\s*:\s*['\"]([A-Z]+)['\"]")
_METHOD_ITEM_RE = re.compile(r"['\"]([A-Z]+)['\"]")

# Matches: class FooController
_CLASS_RE = re.compile(r"^\s*(?:final\s+|abstract\s+|readonly\s+)*class\s+(\w+)", re.MULTILINE)

# Matches: namespace App\Controller\Api;
_NAMESPACE_RE = re.compile(r"^\s*namespace\s+(App\\[\w\\]+)\s*;", re.MULTILINE)

# Matches: public function foo(
_FUNCTION_RE = re.compile(r"public\s+function\s+(\w+)\s*\(")

# Matches: $this->render('path/to/template.html.twig'
_RENDER_RE = re.compile(r"\$this->render\s*\(\s*['\"]([^'\"]+\.html\.twig)['\"]")

# Matches constructor argument types: private readonly FooService $foo
_CTOR_PARAM_RE = re.compile(
    r"(?:private|protected|public)?\s*(?:readonly\s+)?(\\?[\w\\]+)\s+\$\w+",
)


def _extract_methods(attr_rest: str) -> list[str]:
    """Parse the methods argument from a route attribute body."""
    list_match = _METHODS_LIST_RE.search(attr_rest)
    if list_match:
        return [m.group(1) for m in _METHOD_ITEM_RE.finditer(list_match.group(1))]
    single_match = _METHODS_SINGLE_RE.search(attr_rest)
    if single_match:
        return [single_match.group(1)]
    return ["GET"]  # Symfony default


def _extract_services_from_constructor(content: str) -> list[str]:
    """Pull injected service short names from the __construct block."""
    ctor_match = re.search(
        r"public\s+function\s+__construct\s*\(([^)]*)\)",
        content,
        re.DOTALL,
    )
    if not ctor_match:
        return []
    params = ctor_match.group(1)
    services: list[str] = []
    for type_match in _CTOR_PARAM_RE.finditer(params):
        typ = type_match.group(1).lstrip("\\")
        short = typ.split("\\")[-1]
        # Filter out framework types by convention: keep only App\... services
        if "App\\" in typ or short.endswith(("Service", "Repository", "Manager", "Factory", "Provider", "Handler")):
            services.append(short)
    return services


def parse(project_root: Path) -> dict:
    """Scan src/Controller/**/*.php and build the route table."""
    controller_dir = project_root / "src" / "Controller"
    routes: dict[str, dict] = {}

    for php_file in controller_dir.rglob("*.php"):
        rel_path = php_file.relative_to(project_root).as_posix()
        try:
            content = php_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        ns_match = _NAMESPACE_RE.search(content)
        class_match = _CLASS_RE.search(content)
        if not ns_match or not class_match:
            continue
        namespace = ns_match.group(1)
        class_name = class_match.group(1)
        fqcn = f"{namespace}\\{class_name}"

        services = _extract_services_from_constructor(content)

        # Find class-level route prefix (if any) — it appears before `class ...`
        class_start = class_match.start()
        pre_class = content[:class_start]
        class_prefix = ""
        class_route = _ROUTE_ATTR_RE.search(pre_class)
        if class_route:
            class_prefix = class_route.group("path")

        # Find all method-level route attributes
        # For each, find the method name that follows it
        for m in _ROUTE_ATTR_RE.finditer(content, pos=class_start):
            method_path = m.group("path")
            rest = m.group("rest")

            # Skip the class-level route (already captured above)
            if class_route and m.start() == class_route.start():
                continue

            # Find the next `public function X(` after this attribute
            after_attr = content[m.end():]
            fn_match = _FUNCTION_RE.search(after_attr)
            if not fn_match:
                continue
            action = fn_match.group(1)

            # Find the $this->render call inside the method body (rough — first match after fn)
            fn_body_start = m.end() + fn_match.end()
            fn_body = after_attr[fn_match.end():fn_match.end() + 4000]  # scan next 4KB
            render_match = _RENDER_RE.search(fn_body)
            template = render_match.group(1) if render_match else None

            # Compose full path: class prefix + method path (avoid double slashes)
            full_path = (class_prefix.rstrip("/") + "/" + method_path.lstrip("/")).rstrip("/")
            if not full_path:
                full_path = "/"

            name_match = _NAME_RE.search(rest)
            route_name = name_match.group(1) if name_match else None
            methods = _extract_methods(rest)

            routes[full_path] = {
                "name": route_name,
                "methods": methods,
                "controller": fqcn,
                "action": action,
                "file": rel_path,
                "template": template,
                "services": services,
            }

    # Aggregate prefix stats
    by_prefix: dict[str, int] = {}
    for path in routes:
        segments = path.split("/")
        prefix = "/" + segments[1] if len(segments) > 1 and segments[1] else "/"
        by_prefix[prefix] = by_prefix.get(prefix, 0) + 1

    return {
        "routes": routes,
        "stats": {
            "total_routes": len(routes),
            "by_prefix": by_prefix,
        },
    }


def summary(project_root: Path) -> str:
    """Fast summary — scans controllers, counts #[Route attributes via regex."""
    controller_dir = project_root / "src" / "Controller"
    if not controller_dir.is_dir():
        return "no controllers"

    route_count = 0
    file_count = 0
    for php_file in controller_dir.rglob("*.php"):
        file_count += 1
        try:
            content = php_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        route_count += len(_ROUTE_ATTR_RE.findall(content))

    # Subtract class-level route attributes (rough — assume 1 per controller file)
    method_routes = max(0, route_count - file_count)
    return f"{method_routes} routes across {file_count} controllers"
