"""External graph exports: GraphML / Neo4j Cypher / interactive HTML / JSON.

The engine builds a unified knowledge graph (see ``unified_graph.build`` /
``unified_graph.build_for_project``). This module serializes that
``{nodes, edges}`` dict into formats consumable by external tools:

* **GraphML** — for Gephi / yEd (valid XML, parses with ``xml.etree``).
* **Cypher** — ``MERGE`` statements for importing into Neo4j.
* **HTML** — a single self-contained file rendering an interactive
  force-directed graph via the CDN-hosted ``vis-network`` library.
* **JSON** — the raw graph, pretty-printed and key-sorted.

All emitters are *pure*: a graph dict in, a ``str`` out. No I/O, no
network, no Chroma — so they are trivially testable. Only the CLI at the
bottom touches the filesystem and the live project.

CLI::

    uv run python scripts/export_graph.py --format graphml|cypher|html|json [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.sax.saxutils as saxutils
from pathlib import Path

# ── Node attributes serialized into GraphML/Cypher (in declaration order) ──
# Edges carry kind/confidence/relation; nodes carry these.
_NODE_ATTRS = ("label", "kind", "type", "confidence")
_EDGE_ATTRS = ("kind", "confidence", "relation")

# Stable color-by-kind palette for the HTML view.
_KIND_COLORS = {
    "article": "#4f9cff",
    "file": "#6fcf97",
    "class": "#f2c94c",
    "symbol": "#bb6bd9",
    "template": "#eb5757",
}
_DEFAULT_COLOR = "#9aa5b1"


# ── GraphML ────────────────────────────────────────────────────────────


def to_graphml(graph: dict) -> str:
    """Serialize ``graph`` to a GraphML XML document string.

    Declares ``<key>`` elements for node attrs (label, kind, type,
    confidence) and edge attrs (kind, confidence, relation), then one
    ``<node>`` per node and one ``<edge>`` per edge. All attribute values
    are XML-escaped. The output parses with ``xml.etree.ElementTree``.
    """
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns '
        'http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">'
    )

    # <key> declarations. id namespace: n_<attr> for nodes, e_<attr> for edges.
    for attr in _NODE_ATTRS:
        type_ = "double" if attr == "confidence" else "string"
        lines.append(
            f'  <key id="n_{attr}" for="node" attr.name="{attr}" attr.type="{type_}"/>'
        )
    for attr in _EDGE_ATTRS:
        type_ = "double" if attr == "confidence" else "string"
        lines.append(
            f'  <key id="e_{attr}" for="edge" attr.name="{attr}" attr.type="{type_}"/>'
        )

    lines.append('  <graph id="G" edgedefault="directed">')

    for node_id in sorted(nodes.keys()):
        data = nodes[node_id]
        lines.append(f'    <node id="{saxutils.quoteattr(node_id)[1:-1]}">')
        for attr in _NODE_ATTRS:
            value = data.get(attr)
            if value is None:
                continue
            lines.append(
                f'      <data key="n_{attr}">{saxutils.escape(_stringify(value))}</data>'
            )
        lines.append("    </node>")

    for idx, edge in enumerate(edges):
        src = saxutils.quoteattr(edge["from"])[1:-1]
        dst = saxutils.quoteattr(edge["to"])[1:-1]
        lines.append(f'    <edge id="e{idx}" source="{src}" target="{dst}">')
        for attr in _EDGE_ATTRS:
            value = edge.get(attr)
            if value is None:
                continue
            lines.append(
                f'      <data key="e_{attr}">{saxutils.escape(_stringify(value))}</data>'
            )
        lines.append("    </edge>")

    lines.append("  </graph>")
    lines.append("</graphml>")
    return "\n".join(lines) + "\n"


# ── Cypher ─────────────────────────────────────────────────────────────


def to_cypher(graph: dict) -> str:
    """Serialize ``graph`` to a Neo4j Cypher import script string.

    Emits ``MERGE (n:Node {id: "...", label: "...", ...})`` per node, then
    a ``MATCH ... MERGE (a)-[:REL {kind:"..."}]->(b)`` per edge. String
    values have double-quotes and backslashes escaped. Ordering is
    deterministic (nodes sorted by id, edges in input order).
    """
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])

    lines: list[str] = []

    for node_id in sorted(nodes.keys()):
        data = nodes[node_id]
        props = [f'id: "{_cypher_escape(node_id)}"']
        for attr in _NODE_ATTRS:
            value = data.get(attr)
            if value is None:
                continue
            props.append(f"{attr}: {_cypher_value(value)}")
        lines.append(f"MERGE (n:Node {{{', '.join(props)}}});")

    for edge in edges:
        rel_props = []
        for attr in _EDGE_ATTRS:
            value = edge.get(attr)
            if value is None:
                continue
            rel_props.append(f"{attr}:{_cypher_value(value)}")
        rel_clause = f" {{{', '.join(rel_props)}}}" if rel_props else ""
        from_lit = f'"{_cypher_escape(edge["from"])}"'
        to_lit = f'"{_cypher_escape(edge["to"])}"'
        lines.append(
            f"MATCH (a:Node {{id: {from_lit}}}), (b:Node {{id: {to_lit}}}) "
            f"MERGE (a)-[:REL{rel_clause}]->(b);"
        )

    return "\n".join(lines) + ("\n" if lines else "")


# ── HTML ───────────────────────────────────────────────────────────────


def to_html(graph: dict) -> str:
    """Serialize ``graph`` to a single self-contained interactive HTML page.

    Uses the CDN-hosted ``vis-network`` library to draw a force-directed
    graph. Node/edge data is embedded as a JSON blob in a ``<script>``
    tag, with nodes colored by ``kind``. The embedded JSON contains every
    node id verbatim, so callers can assert ids appear in the output.
    """
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])

    vis_nodes = []
    for node_id in sorted(nodes.keys()):
        data = nodes[node_id]
        kind = data.get("kind", "")
        vis_nodes.append(
            {
                "id": node_id,
                "label": data.get("label", node_id),
                "group": kind,
                "color": _KIND_COLORS.get(kind, _DEFAULT_COLOR),
                "title": f"{kind}: {node_id}",
            }
        )

    vis_edges = []
    for edge in edges:
        e = {"from": edge["from"], "to": edge["to"], "label": edge.get("kind", "")}
        vis_edges.append(e)

    # json.dumps without ensure_ascii=False keeps the blob ASCII-safe and,
    # crucially, escapes backslashes so node ids like "class:App\Foo" appear
    # as "class:App\\Foo" — valid inside a <script> JSON literal.
    nodes_json = json.dumps(vis_nodes, indent=2, sort_keys=True)
    edges_json = json.dumps(vis_edges, indent=2, sort_keys=True)

    # Guard against an accidental </script> inside any embedded string.
    nodes_json = nodes_json.replace("</", "<\\/")
    edges_json = edges_json.replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Knowledge Graph</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    html, body {{ margin: 0; height: 100%; background: #11161c; color: #e4e7eb; font-family: system-ui, sans-serif; }}
    #header {{ padding: 8px 16px; font-size: 14px; border-bottom: 1px solid #2a3340; }}
    #graph {{ width: 100%; height: calc(100% - 38px); }}
  </style>
</head>
<body>
  <div id="header">Knowledge Graph &mdash; {len(vis_nodes)} nodes, {len(vis_edges)} edges</div>
  <div id="graph"></div>
  <script type="application/json" id="graph-data">
{{"nodes": {nodes_json}, "edges": {edges_json}}}
  </script>
  <script>
    (function () {{
      var raw = JSON.parse(document.getElementById('graph-data').textContent);
      var container = document.getElementById('graph');
      var data = {{
        nodes: new vis.DataSet(raw.nodes),
        edges: new vis.DataSet(raw.edges)
      }};
      var options = {{
        nodes: {{ shape: 'dot', size: 12, font: {{ color: '#e4e7eb', size: 12 }} }},
        edges: {{ arrows: 'to', color: {{ color: '#46505c' }}, font: {{ color: '#8a94a0', size: 10 }} }},
        physics: {{ stabilization: true, barnesHut: {{ gravitationalConstant: -3000 }} }}
      }};
      new vis.Network(container, data, options);
    }})();
  </script>
</body>
</html>
"""


# ── JSON ───────────────────────────────────────────────────────────────


def to_json(graph: dict) -> str:
    """Pretty-print the raw graph dict (indent=2, key-sorted, round-trippable)."""
    return json.dumps(graph, indent=2, sort_keys=True)


# ── Helpers ────────────────────────────────────────────────────────────


def _stringify(value) -> str:
    """Render a scalar attribute value for XML text content."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _cypher_escape(value: str) -> str:
    """Escape backslashes then double-quotes for a Cypher double-quoted string."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _cypher_value(value) -> str:
    """Render a property value as a Cypher literal (number bare, else quoted)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return f'"{_cypher_escape(str(value))}"'


# ── CLI ────────────────────────────────────────────────────────────────

_FORMATS = {
    "graphml": (to_graphml, "graphml"),
    "cypher": (to_cypher, "cypher"),
    "html": (to_html, "html"),
    "json": (to_json, "json"),
}


def _build_live_graph():
    """Build the live unified graph for this project. May raise on parse errors."""
    # Mirror unified_graph.build_for_project's dual-import regime.
    try:
        from scripts import config, unified_graph
    except ImportError:
        import config  # type: ignore
        import unified_graph  # type: ignore
    return unified_graph.build_for_project(config.PROJECT_ROOT, config.KNOWLEDGE_DIR)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export the unified knowledge graph to an external format."
    )
    parser.add_argument(
        "--format",
        choices=sorted(_FORMATS.keys()),
        default="html",
        help="Output format (default: html).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path. Default: knowledge/exports/graph.<ext>.",
    )
    args = parser.parse_args(argv)

    emitter, ext = _FORMATS[args.format]

    try:
        graph = _build_live_graph()
    except Exception as exc:  # noqa: BLE001 — surface any parser failure cleanly.
        print(f"error: failed to build the knowledge graph: {exc}", file=sys.stderr)
        return 1

    try:
        from scripts import config
    except ImportError:
        import config  # type: ignore

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = config.KNOWLEDGE_DIR / "exports" / f"graph.{ext}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(emitter(graph), encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
