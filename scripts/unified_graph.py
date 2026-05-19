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

    return {"nodes": nodes, "edges": edges}


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
