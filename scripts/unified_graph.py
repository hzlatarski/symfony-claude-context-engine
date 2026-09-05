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
``symbol:``, ``template:``, ``note:``, ``entity:``) so a single ID space is
unambiguous.

Entity nodes (``entity:<type>/<value>``) are lifted from article PROSE by
``entities.extract`` (hosts, commands, roles, env vars, URLs, paths, services)
and joined to the articles that name them via ``mentions`` edges — the same
build-time-only pattern as ``note:`` rationale nodes, so no article file is
ever written. A prose ``path``/``service`` that matches an existing
``file:``/``class:`` node folds into it instead of minting a parallel node.
"""
from __future__ import annotations

from pathlib import Path
import re

# ``entities`` (prose entity extraction) resolves under two import regimes
# depending on how the entrypoint set up sys.path — mirror the parsers pattern.
try:
    from scripts import entities
except ImportError:  # pragma: no cover - exercised only via CLI entrypoint
    import entities

_TYPED_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\](?:\{([a-z0-9_]+)\})?")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_SRC_ANCHOR_RE = re.compile(r"\[src:([^\]]+)\]")


def _resolve_entity_target(
    canonical_id: str, entity_type: str, nodes: dict, class_by_shortname: dict
) -> str | None:
    """Map a prose entity to an EXISTING code node when one is unambiguous.

    A ``path`` entity folds into the ``file:`` node of the same repo path; a
    ``service`` entity folds into a ``class:`` node when exactly one class
    carries that short name. Every other type (host/command/role/envvar/url),
    and any path/service with no match (or an ambiguous service short-name),
    returns ``None`` so the caller mints a standalone ``entity:`` node. This
    stops one real thing being split across a code node and a parallel entity.
    """
    _, _, value = canonical_id.partition("/")
    if entity_type == "path":
        file_id = f"file:{value}"
        return file_id if file_id in nodes else None
    if entity_type == "service":
        candidates = class_by_shortname.get(value, [])
        return candidates[0] if len(candidates) == 1 else None
    return None


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

    def _emit_rationale(owner_id: str, file_path: str, notes: list[dict]) -> None:
        """Materialize ``note:`` leaf nodes for one symbol/class's rationale.

        Each note is a degree-1 leaf keyed by ``note:<file>:<line>:<tag>`` so
        it clusters into its owner's Leiden community without adding hub
        noise. The ``annotates`` edge points owner → note. Notes with no owning
        file are skipped (no ``note::N`` orphans), and a repeated id (two
        identically-tagged comments on one line) is emitted once — no dangling
        duplicate edge.
        """
        if not file_path:
            return
        for n in notes or []:
            line = n.get("line", 0)
            tag = n.get("tag", "NOTE")
            note_id = f"note:{file_path}:{line}:{tag}"
            if note_id in nodes:
                continue
            text = (n.get("text") or "").strip()
            label = f"[{tag}] {text[:60]}" if text else f"[{tag}]"
            nodes[note_id] = {
                "kind": "rationale",
                "label": label,
                "tag": tag,
                "text": text,
                "line": line,
                "file": file_path,
            }
            edges.append({"from": owner_id, "to": note_id, "kind": "annotates"})

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
        _emit_rationale(class_id, file_path, class_info.get("rationale", []))

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
        _emit_rationale(symbol_id, sym_info.get("file", ""), sym_info.get("rationale", []))

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

    # Short-name → class node id(s), for folding ``service`` entities into
    # existing code nodes. Built once, after Pass 0 has minted class nodes.
    class_by_shortname: dict[str, list[str]] = {}
    for nid, ndata in nodes.items():
        if ndata.get("kind") == "class":
            class_by_shortname.setdefault(ndata["label"], []).append(nid)

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
        cited_files: set[str] = set()
        for anchor in _SRC_ANCHOR_RE.findall(stripped):
            if anchor in seen_anchors:
                continue
            seen_anchors.add(anchor)
            file_id = f"file:{anchor}"
            if file_id not in nodes:
                nodes[file_id] = {"kind": "file", "label": anchor}
            edges.append({"from": src_id, "to": file_id, "kind": "cites"})
            cited_files.add(file_id)

        # Entity mentions (Slice 2): recurring named things in prose. Fold into
        # an existing code node when unambiguous, else mint an ``entity:`` node.
        for ent in entities.extract(stripped):
            target_id = _resolve_entity_target(
                ent.canonical_id, ent.type, nodes, class_by_shortname
            )
            if target_id is None:
                target_id = ent.canonical_id
                if target_id not in nodes:
                    nodes[target_id] = {
                        "kind": "entity",
                        "entity_type": ent.type,
                        "label": ent.surface,
                    }
            elif target_id in cited_files:
                # A prose path the article already cites via [src:] — the cites
                # edge already represents it; skip the redundant mention.
                continue
            edges.append({"from": src_id, "to": target_id, "kind": "mentions"})

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
