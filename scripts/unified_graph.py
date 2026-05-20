"""Unified knowledge graph: articles + call graph + src-anchor citations.

Fuses three data sources into a single ``{nodes, edges}`` dict:

* Articles in ``knowledge/concepts/``, ``knowledge/connections/``,
  ``knowledge/qa/`` become ``article:<rel-path-no-ext>`` nodes.
* Symbols and classes from ``parsers.call_graph.parse(...)`` become
  ``symbol:<FQCN>::<method>`` and ``class:<FQCN>`` nodes (the
  call_graph already uses these IDs in its ``symbols`` map).
* File-path tokens like ``src/Foo/Bar.php`` referenced via
  ``[src:src/Foo/Bar.php]`` anchors become ``file:<rel-path>`` nodes.

Edges:

* ``article -> article`` via ``[[wikilink]]`` extraction (kind=``wikilink``,
  optional ``relation`` field carrying the ``{relation}`` annotation).
* ``article -> file`` via ``[src:]`` anchor extraction (kind=``cites``).
* ``article -> symbol`` is NOT emitted directly — the path lookup goes
  through the ``file`` node (article cites file, file owns class, class
  defines symbol). Keeps the graph shape narrow.
* ``symbol -> symbol`` copied verbatim from the call graph (kind=``call``
  or ``render`` per the call_graph's existing kind tags).
* ``file -> class -> symbol`` materialized from the call graph's
  ``classes`` map so file-level traversal works.

Node ID prefixes are non-overlapping (``article:``, ``file:``, ``class:``,
``symbol:``, ``template:``) so a single ID space is unambiguous.
"""
from __future__ import annotations

from pathlib import Path
import re

_TYPED_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\](?:\{([a-z0-9_]+)\})?")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_SRC_ANCHOR_RE = re.compile(r"\[src:([^\]]+)\]")


def build(call_graph: dict, knowledge_root: Path) -> dict:
    """Return ``{nodes, edges}`` for the unified graph.

    Args:
        call_graph: Output of ``parsers.call_graph.parse(project_root)``.
            Expected keys: ``symbols`` (dict), ``edges`` (list),
            ``classes`` (dict).
        knowledge_root: Directory containing ``concepts/``, ``connections/``,
            and ``qa/`` subdirectories of article markdown files.
            Missing subdirs are treated as empty.

    Returns:
        ``{"nodes": {id: {label, kind, ...}}, "edges": [{from, to, kind, ...}]}``
    """
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    article_contents: dict[str, str] = {}

    # Pass 0: materialize call-graph nodes + structural edges.
    for class_fqcn, class_info in call_graph.get("classes", {}).items():
        class_id = f"class:{class_fqcn}"
        file_path = class_info.get("file", "")
        nodes[class_id] = {"kind": "class", "label": class_fqcn.rsplit("\\", 1)[-1]}
        if file_path:
            file_id = f"file:{file_path}"
            if file_id not in nodes:
                nodes[file_id] = {"kind": "file", "label": file_path}
            edges.append({"from": file_id, "to": class_id, "kind": "contains"})

    for symbol_id_raw, sym_info in call_graph.get("symbols", {}).items():
        symbol_id = f"symbol:{symbol_id_raw}"
        nodes[symbol_id] = {
            "kind": "symbol",
            "label": symbol_id_raw.rsplit("::", 1)[-1] if "::" in symbol_id_raw else symbol_id_raw,
        }
        cls = sym_info.get("class", "")
        if cls:
            class_id = f"class:{cls}"
            if class_id in nodes:
                edges.append({"from": class_id, "to": symbol_id, "kind": "defines"})

    for edge in call_graph.get("edges", []):
        dst = edge["to"]
        # Unresolved JS fetch() placeholders ("fetch:POST /api/x") point at an
        # endpoint with no matching route. resolve_fetch_edges() rewrites the
        # resolvable ones to real PHP symbols upstream; whatever still carries
        # the fetch: prefix here is dead — skip it rather than mint a malformed
        # "symbol:fetch:POST /api/x" node.
        if dst.startswith("fetch:"):
            continue
        src = f"symbol:{edge['from']}"
        if dst.startswith("template:"):
            dst_id = dst  # already prefixed
            if dst_id not in nodes:
                nodes[dst_id] = {"kind": "template", "label": dst.split(":", 1)[1]}
        else:
            dst_id = f"symbol:{dst}"
            if dst_id not in nodes:
                # Vendor / unresolved — emit a stub so the edge has a target.
                nodes[dst_id] = {"kind": "symbol", "label": dst.rsplit("::", 1)[-1], "missing": True}
        new_edge = {"from": src, "to": dst_id, "kind": edge.get("kind", "call")}
        if "confidence" in edge:
            new_edge["confidence"] = edge["confidence"]
        if "evidence" in edge:
            new_edge["evidence"] = edge["evidence"]
        edges.append(new_edge)

    for subdir in ("concepts", "connections", "qa"):
        root = knowledge_root / subdir
        if not root.exists():
            continue
        for md in sorted(root.glob("*.md")):
            slug = f"{subdir}/{md.stem}"
            node_id = f"article:{slug}"
            content = md.read_text(encoding="utf-8")
            meta = _parse_article_frontmatter(content)
            nodes[node_id] = {
                "kind": "article",
                "label": meta.get("title") or md.stem,
                "type": meta.get("type", "unknown"),
                "confidence": meta.get("confidence"),
            }
            article_contents[node_id] = content

    # Pass 2: emit edges that reference other nodes.
    for src_id, content in article_contents.items():
        stripped = _HTML_COMMENT_RE.sub("", content)

        for target, relation in [(m.group(1), m.group(2)) for m in _TYPED_WIKILINK_RE.finditer(stripped)]:
            target_id = f"article:{target}"
            if target_id not in nodes:
                continue
            edge: dict = {"from": src_id, "to": target_id, "kind": "wikilink"}
            if relation is not None:
                edge["relation"] = relation
            edges.append(edge)

        seen_anchors: set[str] = set()
        for anchor in _SRC_ANCHOR_RE.findall(stripped):
            if anchor in seen_anchors:
                continue
            seen_anchors.add(anchor)
            file_id = f"file:{anchor}"
            if file_id not in nodes:
                nodes[file_id] = {"kind": "file", "label": anchor}
            edges.append({"from": src_id, "to": file_id, "kind": "cites"})

    return {"nodes": nodes, "edges": edges}


def build_for_project(project_root: Path, knowledge_root: Path) -> dict:
    """Parse the call graph + route map, resolve JS ``fetch()`` edges, then fuse.

    Single source of truth for callers that lack a warm ``ParseCache`` —
    ``compile_truth`` and ``kb_health``. Without this, each consumer would
    call ``call_graph.parse()`` independently and *skip* fetch resolution,
    so their graphs would silently disagree with the MCP server's (which
    resolves fetch edges via ``ParseCache.get_call_graph``).

    The ``parsers`` package resolves under two import regimes depending on
    how the calling entrypoint set up ``sys.path`` — try both.
    """
    try:
        from scripts.parsers import call_graph, route_map
    except ImportError:
        from parsers import call_graph, route_map

    cg = call_graph.parse(project_root)
    call_graph.resolve_fetch_edges(cg, route_map.parse(project_root))
    return build(call_graph=cg, knowledge_root=knowledge_root)


def _parse_article_frontmatter(content: str) -> dict:
    """Minimal YAML frontmatter parser — reuses compile_truth's conventions."""
    if not content.startswith("---"):
        return {}
    end = content.find("---", 3)
    if end == -1:
        return {}
    result: dict = {}
    for line in content[3:end].split("\n"):
        line = line.strip()
        if ":" not in line or line.startswith("-") or line.startswith("#"):
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key == "title":
            result["title"] = value
        elif key == "type":
            result["type"] = value
        elif key == "confidence":
            try:
                result["confidence"] = float(value)
            except ValueError:
                pass
    return result
