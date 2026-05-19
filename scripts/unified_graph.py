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
    return {"nodes": {}, "edges": []}
