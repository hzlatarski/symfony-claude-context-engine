"""PHP call-graph parser.

Builds a symbol-level call graph for src/**/*.php using tree-sitter:
each method is a node, each `$this->foo->bar()`, `Foo::bar()`,
`(new Foo)->bar()`, etc., is an edge tagged with a confidence score
based on how the receiver type was resolved.

Confidence scale:
    1.0  static call (Foo::bar) or constructor-injected typed property
    0.7  typed local variable (Foo $x = ...; $x->bar())
    0.4  inferred from method-return-type chain
    skipped — no resolution possible (dynamic dispatch, unbound $this)

This is the structural foundation for `trace_route` and
`impact_of_change` MCP tools — the parser produces a deterministic JSON
graph; the tools traverse it.
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_PARSER: Any = None
_JS_PARSER: Any = None


def _get_parser() -> Any:
    """Return a cached tree-sitter PHP Parser, or None if deps are missing."""
    global _PARSER
    if _PARSER is not None:
        return _PARSER if _PARSER is not False else None

    try:
        from tree_sitter import Language, Parser
        import tree_sitter_php
    except (ImportError, AttributeError):
        _PARSER = False
        return None

    language = Language(tree_sitter_php.language_php())
    _PARSER = Parser(language)
    return _PARSER


def _get_js_parser() -> Any:
    """Return a cached tree-sitter JavaScript Parser, or None if deps are missing."""
    global _JS_PARSER
    if _JS_PARSER is not None:
        return _JS_PARSER if _JS_PARSER is not False else None

    try:
        from tree_sitter import Language, Parser
        import tree_sitter_javascript
    except (ImportError, AttributeError):
        _JS_PARSER = False
        return None

    language = Language(tree_sitter_javascript.language())
    _JS_PARSER = Parser(language)
    return _JS_PARSER


def _walk(node):
    """Yield every descendant node, depth-first."""
    yield node
    for child in node.children:
        yield from _walk(child)


def _text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _find_child(node, type_name: str):
    for child in node.children:
        if child.type == type_name:
            return child
    return None


def _find_children(node, type_name: str) -> list:
    return [c for c in node.children if c.type == type_name]


def _extract_namespace(root, source: bytes) -> str:
    """Return the FQCN namespace declared at the top of the file (or '')."""
    for node in root.children:
        if node.type == "namespace_definition":
            name = _find_child(node, "namespace_name")
            if name is not None:
                return _text(name, source)
        if node.type == "class_declaration":
            break
    return ""


def _extract_use_imports(root, source: bytes) -> dict[str, str]:
    """Build a {short_name: FQCN} map from `use` statements.

    Handles aliases (``use Foo\\Bar as Baz``) so the alias maps to the FQCN.
    """
    imports: dict[str, str] = {}
    for node in root.children:
        if node.type != "namespace_use_declaration":
            continue
        for clause in _find_children(node, "namespace_use_clause"):
            qualified = _find_child(clause, "qualified_name")
            if qualified is None:
                continue
            fqcn = _text(qualified, source).lstrip("\\")
            short = fqcn.rsplit("\\", 1)[-1]
            # Look for `as Alias`
            saw_as = False
            for child in clause.children:
                if child.type == "as":
                    saw_as = True
                    continue
                if saw_as and child.type == "name":
                    short = _text(child, source)
                    break
            imports[short] = fqcn
    return imports


def _resolve_type_name(short_name: str, imports: dict[str, str], current_namespace: str) -> str:
    """Resolve a short type name to FQCN using `use` imports + current namespace.

    Falls back to ``<current_namespace>\\<short_name>`` if not in imports —
    this matches PHP's name-resolution rules for an unqualified type.
    Already-qualified (contains ``\\``) names pass through unchanged.
    """
    if not short_name:
        return ""
    if "\\" in short_name:
        return short_name.lstrip("\\")
    if short_name in imports:
        return imports[short_name]
    if current_namespace:
        return f"{current_namespace}\\{short_name}"
    return short_name


def _method_visibility(method_node, source: bytes) -> str:
    """Find the public/protected/private modifier on a method declaration."""
    for child in method_node.children:
        if child.type == "visibility_modifier":
            return _text(child, source)
    return "public"


def _extract_property_types_from_class(
    class_node,
    source: bytes,
    imports: dict[str, str],
    namespace: str,
) -> dict[str, str]:
    """Build {property_name: FQCN} from typed properties + ctor promotion + ctor assignments.

    The mapping is what `$this->name` resolves to inside any method on this class.
    """
    props: dict[str, str] = {}
    decl_list = _find_child(class_node, "declaration_list")
    if decl_list is None:
        return props

    # Pass A: typed `property_declaration` (e.g. ``private SessionService $session;``)
    for member in decl_list.children:
        if member.type != "property_declaration":
            continue
        type_node = _find_child(member, "named_type")
        if type_node is None:
            continue
        type_name = _resolve_type_name(_text(type_node, source).lstrip("?"), imports, namespace)
        for elem in _find_children(member, "property_element"):
            var_name = _find_child(elem, "variable_name")
            if var_name is None:
                continue
            prop_name = _text(_find_child(var_name, "name") or var_name, source)
            props[prop_name] = type_name

    # Pass B: ``__construct`` — promoted params + body assignments
    ctor = None
    for member in decl_list.children:
        if member.type == "method_declaration":
            name_node = _find_child(member, "name")
            if name_node and _text(name_node, source) == "__construct":
                ctor = member
                break
    if ctor is None:
        return props

    formal = _find_child(ctor, "formal_parameters")
    classic_param_types: dict[str, str] = {}
    if formal is not None:
        for param in formal.children:
            type_node = _find_child(param, "named_type")
            if type_node is None:
                continue
            type_name = _resolve_type_name(_text(type_node, source).lstrip("?"), imports, namespace)
            var_name_node = _find_child(param, "variable_name")
            if var_name_node is None:
                continue
            param_name = _text(_find_child(var_name_node, "name") or var_name_node, source)
            if param.type == "property_promotion_parameter":
                props[param_name] = type_name
            else:  # simple_parameter — track for body-assignment resolution
                classic_param_types[param_name] = type_name

    body = _find_child(ctor, "compound_statement")
    if body is not None and classic_param_types:
        for node in _walk(body):
            if node.type != "assignment_expression":
                continue
            left = node.children[0] if node.children else None
            right = node.children[-1] if len(node.children) >= 3 else None
            if left is None or right is None:
                continue
            if left.type != "member_access_expression":
                continue
            obj = _find_child(left, "variable_name")
            if obj is None or _text(_find_child(obj, "name") or obj, source) != "this":
                continue
            prop_name_node = None
            # member_access_expression children: variable_name, '->', name
            for child in left.children:
                if child.type == "name":
                    prop_name_node = child
            if prop_name_node is None:
                continue
            prop_name = _text(prop_name_node, source)
            if right.type == "variable_name":
                src_name = _text(_find_child(right, "name") or right, source)
                if src_name in classic_param_types:
                    props[prop_name] = classic_param_types[src_name]

    return props


def _extract_member_call_chain(call_node, source: bytes) -> tuple[str, list[str]]:
    """Decompose a ``$root->prop1->prop2->method()`` chain.

    Returns (root_text, [prop1, prop2, ..., method]) where ``root_text`` is the
    raw text of the leftmost expression — typically ``$this``, a variable
    name, or a more complex expression we won't try to resolve.
    """
    method_name_node = None
    for child in call_node.children:
        if child.type == "name":
            method_name_node = child
    if method_name_node is None:
        return "", []
    method_name = _text(method_name_node, source)

    # The receiver is the first child (object).
    receiver = call_node.children[0]
    chain: list[str] = []
    cur = receiver
    while cur is not None and cur.type == "member_access_expression":
        prop_name_node = None
        for child in cur.children:
            if child.type == "name":
                prop_name_node = child
        if prop_name_node is None:
            break
        chain.insert(0, _text(prop_name_node, source))
        cur = cur.children[0] if cur.children else None

    root_text = _text(cur, source) if cur is not None else ""
    chain.append(method_name)
    return root_text, chain


def _resolve_static_receiver(
    receiver_node,
    source: bytes,
    imports: dict[str, str],
    namespace: str,
    current_class_fqcn: str,
    parent_class_fqcn: str = "",
) -> str:
    """Return the FQCN of a ``Receiver::method()`` style scope, or ''."""
    if receiver_node.type == "name":
        return _resolve_type_name(_text(receiver_node, source), imports, namespace)
    if receiver_node.type == "qualified_name":
        return _text(receiver_node, source).lstrip("\\")
    if receiver_node.type == "relative_scope":
        scope = _text(receiver_node, source)
        if scope in {"self", "static"}:
            return current_class_fqcn
        if scope == "parent":
            return parent_class_fqcn
    return ""


def _entity_fqcn_from_get_repository_args(call_node, source: bytes, imports: dict[str, str], namespace: str) -> str:
    """For a ``getRepository(X::class)`` call node, return X's FQCN, or ''."""
    args = _find_child(call_node, "arguments")
    if args is None:
        return ""
    for arg in args.children:
        if arg.type != "argument":
            continue
        for child in arg.children:
            if child.type == "class_constant_access_expression":
                # children: <name>, '::', name(class)
                receiver = child.children[0] if child.children else None
                if receiver is None:
                    continue
                if receiver.type == "name":
                    return _resolve_type_name(_text(receiver, source), imports, namespace)
                if receiver.type == "qualified_name":
                    return _text(receiver, source).lstrip("\\")
    return ""


def _entity_fqcn_to_repo_fqcn(entity_fqcn: str) -> str:
    """Apply the Symfony convention: ``App\\Entity\\X`` -> ``App\\Repository\\XRepository``.

    For non-App or non-Entity namespaces, fall back to derivation by basename:
    take the short class name and prepend ``App\\Repository\\``. This is wrong
    for some libraries but right for the project's own entities — which is
    where the repository pattern actually applies.
    """
    if not entity_fqcn:
        return ""
    short = entity_fqcn.rsplit("\\", 1)[-1]
    repo_short = f"{short}Repository"
    if entity_fqcn.startswith("App\\Entity\\"):
        return f"App\\Repository\\{repo_short}"
    return f"App\\Repository\\{repo_short}"


def _is_get_repository_call(node, source: bytes) -> bool:
    """True iff ``node`` is a ``->getRepository(...)`` member_call_expression."""
    if node.type != "member_call_expression":
        return False
    method_name = None
    for child in node.children:
        if child.type == "name":
            method_name = _text(child, source)
    return method_name == "getRepository"


_RENDER_METHODS = frozenset({"render", "renderView", "renderForm"})


def _first_string_arg(call_node, source: bytes) -> str:
    """Return the literal string value of the first argument, or '' if non-literal."""
    args = _find_child(call_node, "arguments")
    if args is None:
        return ""
    for arg in args.children:
        if arg.type != "argument":
            continue
        for child in arg.children:
            if child.type == "string":
                content = _find_child(child, "string_content")
                if content is not None:
                    return _text(content, source)
                # Fallback: strip surrounding quotes from raw text
                raw = _text(child, source)
                if len(raw) >= 2 and raw[0] in "'\"":
                    return raw[1:-1]
                return raw
        return ""
    return ""


def _build_local_types(
    method_node,
    source: bytes,
    imports: dict[str, str],
    namespace: str,
) -> dict[str, str]:
    """Build {var_name: FQCN} for typed locals: typed params, ``new X()``, and Doctrine ``getRepository(X::class)`` assignments."""
    locals_: dict[str, tuple[str, float]] = {}
    formal = _find_child(method_node, "formal_parameters")
    if formal is not None:
        for param in formal.children:
            if param.type != "simple_parameter":
                continue
            type_node = _find_child(param, "named_type")
            if type_node is None:
                continue
            type_name = _resolve_type_name(_text(type_node, source).lstrip("?"), imports, namespace)
            var_name = _find_child(param, "variable_name")
            if var_name is None:
                continue
            name = _text(_find_child(var_name, "name") or var_name, source)
            locals_[name] = (type_name, 0.7)

    body = _find_child(method_node, "compound_statement")
    if body is not None:
        for node in _walk(body):
            if node.type != "assignment_expression":
                continue
            children = node.children
            if len(children) < 3:
                continue
            left, _, right = children[0], children[1], children[-1]
            if left.type != "variable_name":
                continue
            var_name = _text(_find_child(left, "name") or left, source)
            if right.type == "object_creation_expression":
                type_node = None
                for child in right.children:
                    if child.type in ("name", "qualified_name"):
                        type_node = child
                        break
                if type_node is None:
                    continue
                type_text = _text(type_node, source)
                locals_[var_name] = (_resolve_type_name(type_text, imports, namespace), 0.7)
                continue
            if _is_get_repository_call(right, source):
                entity = _entity_fqcn_from_get_repository_args(right, source, imports, namespace)
                if entity:
                    locals_[var_name] = (_entity_fqcn_to_repo_fqcn(entity), 1.0)
    return locals_


def _walk_method_for_calls(
    method_node,
    source: bytes,
    from_id: str,
    rel_path: str,
    property_types: dict[str, str],
    imports: dict[str, str],
    namespace: str,
    current_class_fqcn: str,
    parent_class_fqcn: str = "",
) -> list[dict]:
    """Emit edges from a single method body."""
    edges: list[dict] = []
    body = _find_child(method_node, "compound_statement")
    if body is None:
        return edges

    local_types = _build_local_types(method_node, source, imports, namespace)

    for node in _walk(body):
        if node.type == "member_call_expression":
            # Chained Doctrine: $whatever->getRepository(X::class)->method()
            receiver = node.children[0] if node.children else None
            method_name_node = None
            for child in node.children:
                if child.type == "name":
                    method_name_node = child
            outer_method = _text(method_name_node, source) if method_name_node else ""
            if (
                receiver is not None
                and _is_get_repository_call(receiver, source)
                and outer_method
            ):
                entity = _entity_fqcn_from_get_repository_args(
                    receiver, source, imports, namespace,
                )
                repo_fqcn = _entity_fqcn_to_repo_fqcn(entity)
                if repo_fqcn:
                    edges.append({
                        "from": from_id,
                        "to": f"{repo_fqcn}::{outer_method}",
                        "kind": "call",
                        "confidence": 1.0,
                        "evidence": f"{rel_path}:{node.start_point[0] + 1}",
                    })
                continue

            # Suppress noise: $em->getRepository(X::class) standalone is
            # uninteresting (vendor target). The repo edge fires when it's
            # actually used (chained or via local var assignment).
            if outer_method == "getRepository":
                continue

            root_text, chain = _extract_member_call_chain(node, source)
            if not chain:
                continue
            # $this->render('x.html.twig', ...) — emit render edge instead of self-call.
            if (
                root_text == "$this"
                and len(chain) == 1
                and chain[0] in _RENDER_METHODS
            ):
                template = _first_string_arg(node, source)
                if template:
                    edges.append({
                        "from": from_id,
                        "to": f"template:{template}",
                        "kind": "render",
                        "confidence": 1.0,
                        "evidence": f"{rel_path}:{node.start_point[0] + 1}",
                    })
                continue
            # $this->method() — call on self (resolved to inheritance chain by post-pass).
            if root_text == "$this" and len(chain) == 1:
                edges.append({
                    "from": from_id,
                    "to": f"{current_class_fqcn}::{chain[0]}",
                    "kind": "call",
                    "confidence": 1.0,
                    "evidence": f"{rel_path}:{node.start_point[0] + 1}",
                })
                continue
            # $this->prop->method() with one hop — resolve via property table.
            if root_text == "$this" and len(chain) == 2:
                prop = chain[0]
                method = chain[-1]
                target_class = property_types.get(prop, "")
                if target_class:
                    edges.append({
                        "from": from_id,
                        "to": f"{target_class}::{method}",
                        "kind": "call",
                        "confidence": 1.0,
                        "evidence": f"{rel_path}:{node.start_point[0] + 1}",
                    })
                continue
            # $var->method() — typed-local resolution.
            if root_text.startswith("$") and len(chain) == 1:
                var_name = root_text[1:]
                local_entry = local_types.get(var_name)
                if local_entry:
                    target_class, confidence = local_entry
                    if target_class:
                        edges.append({
                            "from": from_id,
                            "to": f"{target_class}::{chain[-1]}",
                            "kind": "call",
                            "confidence": confidence,
                            "evidence": f"{rel_path}:{node.start_point[0] + 1}",
                        })
            continue

        if node.type == "scoped_call_expression":
            # Children: <receiver>, '::', name, arguments
            receiver = node.children[0] if node.children else None
            if receiver is None:
                continue
            method_name_node = None
            for child in node.children:
                if child.type == "name" and child is not receiver:
                    method_name_node = child
            if method_name_node is None:
                continue
            target_class = _resolve_static_receiver(
                receiver, source, imports, namespace, current_class_fqcn, parent_class_fqcn,
            )
            if not target_class:
                continue
            method_name = _text(method_name_node, source)
            edges.append({
                "from": from_id,
                "to": f"{target_class}::{method_name}",
                "kind": "call",
                "confidence": 1.0,
                "evidence": f"{rel_path}:{node.start_point[0] + 1}",
            })
    return edges


def _extract_extends(class_node, source: bytes, imports: dict[str, str], namespace: str) -> str:
    """Resolve the parent class FQCN from a ``extends Foo`` clause, or ''."""
    base = _find_child(class_node, "base_clause")
    if base is None:
        return ""
    for child in base.children:
        if child.type == "name":
            return _resolve_type_name(_text(child, source), imports, namespace)
        if child.type == "qualified_name":
            return _text(child, source).lstrip("\\")
    return ""


def _parse_file(
    path: Path,
    rel_path: str,
) -> tuple[dict[str, dict], list[dict], dict[str, dict]]:
    """Parse a single PHP file, returning (symbols, edges, classes)."""
    parser = _get_parser()
    if parser is None:
        return {}, [], {}

    try:
        source = path.read_bytes()
    except OSError:
        return {}, [], {}

    tree = parser.parse(source)
    root = tree.root_node

    namespace = _extract_namespace(root, source)
    imports = _extract_use_imports(root, source)
    symbols: dict[str, dict] = {}
    edges: list[dict] = []
    classes: dict[str, dict] = {}

    for node in _walk(root):
        if node.type != "class_declaration":
            continue
        name_node = _find_child(node, "name")
        if name_node is None:
            continue
        class_name = _text(name_node, source)
        fqcn = f"{namespace}\\{class_name}" if namespace else class_name

        property_types = _extract_property_types_from_class(node, source, imports, namespace)
        parent_fqcn = _extract_extends(node, source, imports, namespace)
        classes[fqcn] = {"file": rel_path, "extends": parent_fqcn}

        decl_list = _find_child(node, "declaration_list")
        if decl_list is None:
            continue

        for member in decl_list.children:
            if member.type != "method_declaration":
                continue
            method_name_node = _find_child(member, "name")
            if method_name_node is None:
                continue
            method_name = _text(method_name_node, source)
            symbol_id = f"{fqcn}::{method_name}"
            symbols[symbol_id] = {
                "file": rel_path,
                "line": member.start_point[0] + 1,
                "end_line": member.end_point[0] + 1,
                "kind": "method",
                "visibility": _method_visibility(member, source),
            }
            edges.extend(_walk_method_for_calls(
                member, source, symbol_id, rel_path, property_types,
                imports, namespace, fqcn, parent_fqcn,
            ))

    return symbols, edges, classes


def _stimulus_name_from_path(rel_path: str) -> str:
    """``assets/controllers/chat_arena_controller.js`` -> ``chat-arena``.

    Mirrors Stimulus's snake_case → kebab-case naming convention for
    HTML ``data-controller`` references.
    """
    name = Path(rel_path).stem
    if name.endswith("_controller"):
        name = name[: -len("_controller")]
    return name.replace("_", "-")


def _extract_fetch_url(arg_node, source: bytes) -> tuple[str, float]:
    """Pull the URL string from a fetch first-arg, with confidence.

    - ``'literal'`` → (url, 1.0)
    - ``\\`prefix${x}/suffix\\``` → ('prefix*/suffix', 0.7) — substitutions
      collapse to ``*`` so the resolver can wildcard-match Symfony route
      placeholders (``{id}`` etc.).
    - anything else (variable, member expression, computed) → ('', 0.0)
    """
    if arg_node.type == "string":
        frag = _find_child(arg_node, "string_fragment")
        if frag is not None:
            return _text(frag, source), 1.0
        return "", 0.0
    if arg_node.type == "template_string":
        parts: list[str] = []
        for child in arg_node.children:
            if child.type == "string_fragment":
                parts.append(_text(child, source))
            elif child.type == "template_substitution":
                parts.append("*")
        joined = "".join(parts)
        return (joined, 0.7) if joined else ("", 0.0)
    return "", 0.0


def _extract_fetch_method(obj_node, source: bytes) -> str:
    """Find ``method: 'POST'`` in a fetch options object. Default ``GET``."""
    for pair in obj_node.children:
        if pair.type != "pair":
            continue
        key = _find_child(pair, "property_identifier")
        if key is None or _text(key, source) != "method":
            continue
        for child in pair.children:
            if child.type == "string":
                frag = _find_child(child, "string_fragment")
                if frag is not None:
                    return _text(frag, source).upper()
    return "GET"


def _walk_js_method_for_fetch(
    method_node,
    source: bytes,
    from_id: str,
    rel_path: str,
) -> list[dict]:
    """Emit a ``fetch:<METHOD> <url>`` edge for each fetch() call in a method body."""
    edges: list[dict] = []
    body = _find_child(method_node, "statement_block")
    if body is None:
        return edges
    for node in _walk(body):
        if node.type != "call_expression":
            continue
        if not node.children:
            continue
        callee = node.children[0]
        if callee.type != "identifier" or _text(callee, source) != "fetch":
            continue
        args = _find_child(node, "arguments")
        if args is None:
            continue
        first_arg = None
        options = None
        seen_first = False
        for child in args.children:
            if child.type in {"(", ")", ","}:
                continue
            if not seen_first:
                first_arg = child
                seen_first = True
                continue
            if child.type == "object":
                options = child
                break
        if first_arg is None:
            continue
        url, confidence = _extract_fetch_url(first_arg, source)
        if not url:
            continue
        http_method = _extract_fetch_method(options, source) if options is not None else "GET"
        edges.append({
            "from": from_id,
            "to": f"fetch:{http_method} {url}",
            "kind": "fetch",
            "confidence": confidence,
            "evidence": f"{rel_path}:{node.start_point[0] + 1}",
        })
    return edges


def _parse_js_file(path: Path, rel_path: str) -> tuple[dict[str, dict], list[dict]]:
    """Parse a Stimulus controller file into ``(symbols, edges)``.

    Symbols use the form ``js:<stimulus-name>::<method>``. Edges target
    placeholder strings like ``fetch:POST /api/foo`` that ``resolve_fetch_edges``
    rewrites to PHP controller symbols using the route map.
    """
    parser = _get_js_parser()
    if parser is None:
        return {}, []
    try:
        source = path.read_bytes()
    except OSError:
        return {}, []
    tree = parser.parse(source)
    stim_name = _stimulus_name_from_path(rel_path)
    symbols: dict[str, dict] = {}
    edges: list[dict] = []
    for node in _walk(tree.root_node):
        if node.type != "method_definition":
            continue
        name_node = _find_child(node, "property_identifier")
        if name_node is None:
            continue
        method_name = _text(name_node, source)
        symbol_id = f"js:{stim_name}::{method_name}"
        symbols[symbol_id] = {
            "file": rel_path,
            "line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "kind": "stimulus_method",
        }
        edges.extend(_walk_js_method_for_fetch(node, source, symbol_id, rel_path))
    return symbols, edges


def resolve_fetch_edges(graph: dict, routes: dict) -> None:
    """Rewrite ``fetch:METHOD /url`` placeholders in-place to PHP controller symbols.

    Matches ``METHOD`` against ``route['methods']`` and matches the URL
    against ``route['path']`` after normalizing both sides — the JS side
    already uses ``*`` for substitutions; the route side uses ``{name}``
    placeholders that we collapse to ``*`` for comparison.

    Edges that don't match any route are left as-is so users see the
    attempted target instead of a silent drop.
    """
    # Pre-normalize the route paths once.
    normalized: list[tuple[str, str, dict]] = []
    placeholder_re = re.compile(r"\{[^}]+\}")
    for path, route in routes.get("routes", {}).items():
        norm = placeholder_re.sub("*", path)
        normalized.append((norm, path, route))

    for edge in graph["edges"]:
        target = edge.get("to", "")
        if not target.startswith("fetch:"):
            continue
        rest = target[len("fetch:"):]
        space = rest.find(" ")
        if space == -1:
            continue
        method = rest[:space]
        url = rest[space + 1:]
        for norm_path, _orig, route in normalized:
            if method in route["methods"] and norm_path == url:
                edge["to"] = f"{route['controller']}::{route['action']}"
                break


def parse(project_root: Path) -> dict:
    """Build the call graph for ``src/**/*.php`` and ``assets/controllers/**/*.js``.

    Returns a dict with four top-level keys:
        symbols: {symbol_id: {file, line, end_line, kind, visibility?}}
        edges:   [{from, to, kind, confidence, evidence}, ...]
        classes: {fqcn: {file, extends}}
        stats:   {total_files, total_symbols, total_edges, ..., by_confidence}

    Stimulus controllers are parsed separately and produce ``js:<name>::<method>``
    symbols plus ``fetch:METHOD /url`` placeholder edges. Call ``resolve_fetch_edges``
    afterwards to map those placeholders to PHP controller symbols using a route map.
    """
    src_dir = project_root / "src"
    php_files = sorted(src_dir.rglob("*.php")) if src_dir.is_dir() else []

    symbols: dict[str, dict] = {}
    edges: list[dict] = []
    classes: dict[str, dict] = {}

    for path in php_files:
        rel_path = str(path.relative_to(project_root)).replace("\\", "/")
        file_symbols, file_edges, file_classes = _parse_file(path, rel_path)
        symbols.update(file_symbols)
        edges.extend(file_edges)
        classes.update(file_classes)

    _resolve_inherited_targets(edges, symbols, classes)

    js_dir = project_root / "assets" / "controllers"
    js_files = sorted(js_dir.rglob("*_controller.js")) if js_dir.is_dir() else []
    for path in js_files:
        rel_path = str(path.relative_to(project_root)).replace("\\", "/")
        js_symbols, js_edges = _parse_js_file(path, rel_path)
        symbols.update(js_symbols)
        edges.extend(js_edges)

    by_confidence: dict[str, int] = {}
    for edge in edges:
        key = f"{edge['confidence']:.1f}"
        by_confidence[key] = by_confidence.get(key, 0) + 1

    return {
        "symbols": symbols,
        "edges": edges,
        "classes": classes,
        "stats": {
            "total_files": len(php_files) + len(js_files),
            "total_php_files": len(php_files),
            "total_js_files": len(js_files),
            "total_symbols": len(symbols),
            "total_edges": len(edges),
            "total_classes": len(classes),
            "by_confidence": by_confidence,
        },
    }


def _max_mtime(php_files: list[Path]) -> float:
    """Largest mtime across the file list, or 0.0 if empty."""
    return max((p.stat().st_mtime for p in php_files), default=0.0)


def _git_head(project_root: Path) -> str:
    """Return current git HEAD SHA, or '' if not a git repo / git unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        ).decode("ascii").strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""


def load_or_parse(project_root: Path, cache_file: Path | None = None) -> dict:
    """Return cached call graph if mtime + HEAD match, else rebuild and cache.

    Cache key = (max mtime across src/**/*.php, git HEAD). Either changing
    invalidates. ``cache_file`` defaults to ``knowledge/call-graph.json``.
    """
    if cache_file is None:
        cache_file = project_root / "knowledge" / "call-graph.json"

    src_dir = project_root / "src"
    php_files = sorted(src_dir.rglob("*.php")) if src_dir.is_dir() else []
    max_mtime = _max_mtime(php_files)
    head = _git_head(project_root)

    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            cache_key = cached.get("cache_key", {})
            if (
                cache_key.get("max_mtime") == max_mtime
                and cache_key.get("head") == head
            ):
                return cached
        except (json.JSONDecodeError, OSError):
            pass

    result = parse(project_root)
    result["cache_key"] = {
        "max_mtime": max_mtime,
        "head": head,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


_DIFF_FILE_RE = re.compile(
    r"^diff --git a/(?P<old>[^\s]+) b/(?P<new>[^\s]+)$",
    re.MULTILINE,
)
_DIFF_HUNK_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@",
    re.MULTILINE,
)


def parse_diff_hunks(diff_text: str) -> dict[str, list[tuple[int, int]]]:
    """Parse ``git diff -U0`` output into ``{file_path: [(start, end), ...]}``.

    Returns inclusive line ranges referring to the **new** file (post-diff).
    Pure-deletion hunks (new_count=0) collapse to a single anchor line so a
    method straddling the deletion still gets flagged. Only PHP files are
    returned — all other extensions filter out.
    """
    out: dict[str, list[tuple[int, int]]] = {}
    # Walk diffs file by file. Each file's body extends to the next
    # ``diff --git`` header (or EOF), so we slice on those boundaries.
    file_matches = list(_DIFF_FILE_RE.finditer(diff_text))
    for i, match in enumerate(file_matches):
        rel_path = match.group("new")
        if not rel_path.endswith(".php"):
            continue
        body_start = match.end()
        body_end = file_matches[i + 1].start() if i + 1 < len(file_matches) else len(diff_text)
        body = diff_text[body_start:body_end]
        ranges: list[tuple[int, int]] = []
        for hunk in _DIFF_HUNK_RE.finditer(body):
            new_start = int(hunk.group("new_start"))
            new_count = int(hunk.group("new_count")) if hunk.group("new_count") else 1
            if new_count == 0:
                ranges.append((new_start, new_start))
            else:
                ranges.append((new_start, new_start + new_count - 1))
        if ranges:
            out[rel_path] = ranges
    return out


def find_changed_symbols(
    graph: dict,
    file_ranges: dict[str, list[tuple[int, int]]],
) -> set[str]:
    """Return symbol IDs whose ``[line, end_line]`` overlaps any provided range.

    ``file_ranges`` maps a project-relative file path to a list of
    ``(start_line, end_line)`` tuples (inclusive). Files not present in the
    graph's symbol table are silently skipped — caller may have edited
    non-PHP files we don't track.
    """
    by_file: dict[str, list[tuple[int, int, str]]] = {}
    for symbol_id, info in graph["symbols"].items():
        f = info.get("file")
        if not f:
            continue
        by_file.setdefault(f, []).append((info["line"], info.get("end_line", info["line"]), symbol_id))

    out: set[str] = set()
    for file, ranges in file_ranges.items():
        symbols_in_file = by_file.get(file, [])
        for r_start, r_end in ranges:
            for s_start, s_end, sid in symbols_in_file:
                # Inclusive overlap on both sides.
                if r_end >= s_start and r_start <= s_end:
                    out.add(sid)
    return out


def reverse_callers(
    graph: dict,
    target: str,
    max_depth: int = 6,
) -> list[dict]:
    """BFS the reverse-edge graph starting at ``target``.

    Returns a list of ``{symbol, depth}`` for every upstream caller within
    ``max_depth`` hops, where depth=1 means a direct caller. Each symbol
    appears at most once with its minimum depth — useful for routing risk
    scores by proximity to the change.
    """
    from collections import deque

    if target not in graph["symbols"]:
        return []

    edges_by_to: dict[str, list[str]] = {}
    for edge in graph["edges"]:
        edges_by_to.setdefault(edge["to"], []).append(edge["from"])

    visited: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque([(target, 0)])
    while queue:
        sym, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for caller in edges_by_to.get(sym, []):
            if caller in visited:
                continue
            visited[caller] = depth + 1
            queue.append((caller, depth + 1))

    return [{"symbol": s, "depth": d} for s, d in visited.items()]


def trace(graph: dict, from_id: str, max_depth: int = 6) -> dict:
    """Walk the call graph from ``from_id`` and return an ordered tree.

    Each node is ``{symbol, kind, confidence, evidence, children, ...}``.
    The root carries ``kind="root"`` and no edge metadata; descendant nodes
    carry the metadata from the edge that reached them.

    Cycles are truncated: the second visit to a symbol on the current path
    is recorded as a leaf with ``truncated="cycle"`` instead of recursing.
    Symbols not present in ``graph["symbols"]`` are emitted with
    ``missing=True`` (useful for vendor or unresolved targets).
    """
    edges_by_from: dict[str, list[dict]] = {}
    for edge in graph["edges"]:
        edges_by_from.setdefault(edge["from"], []).append(edge)

    def visit(symbol: str, depth: int, path: set[str]) -> dict:
        node: dict = {"symbol": symbol, "children": []}
        if symbol not in graph["symbols"] and not symbol.startswith("template:"):
            node["missing"] = True
        if depth >= max_depth:
            return node
        outgoing = edges_by_from.get(symbol, [])
        new_path = path | {symbol}
        for edge in outgoing:
            child_symbol = edge["to"]
            child: dict = {
                "symbol": child_symbol,
                "kind": edge["kind"],
                "confidence": edge["confidence"],
                "evidence": edge["evidence"],
                "children": [],
            }
            if child_symbol in new_path:
                child["truncated"] = "cycle"
            elif edge["kind"] == "render":
                # Templates are leaves — nothing to traverse further.
                pass
            else:
                deeper = visit(child_symbol, depth + 1, new_path)
                child["children"] = deeper["children"]
                if deeper.get("missing"):
                    child["missing"] = True
            node["children"].append(child)
        return node

    return visit(from_id, 0, set())


def _resolve_inherited_targets(
    edges: list[dict],
    symbols: dict[str, dict],
    classes: dict[str, dict],
) -> None:
    """Rewrite edges whose target class doesn't define the called method but a parent does.

    Walks the inheritance chain via ``classes[cls]['extends']``. Only rewrites
    when the target symbol is missing AND a parent class defines it — leaves
    edges to vendor or unknown classes unchanged. Mutates ``edges`` in place.
    """
    for edge in edges:
        target = edge["to"]
        if target in symbols:
            continue
        if "::" not in target:
            continue
        cls, method = target.rsplit("::", 1)
        parent = classes.get(cls, {}).get("extends", "")
        seen = {cls}
        while parent and parent not in seen:
            candidate = f"{parent}::{method}"
            if candidate in symbols:
                edge["to"] = candidate
                break
            seen.add(parent)
            parent = classes.get(parent, {}).get("extends", "")
