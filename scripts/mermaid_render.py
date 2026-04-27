"""Render call-graph subgraphs as Mermaid flowcharts.

LLMs (and humans) read mermaid diagrams much faster than nested
markdown trees once the call chain branches more than two or three
levels. The MCP tools that touch the graph (``trace_route``,
``impact_of_change``) accept ``format="mermaid"`` and delegate here.

This module is deliberately string-based — it does not import the
parsers. The caller passes already-resolved tree/edge dicts; the
renderer just turns them into mermaid syntax.
"""
from __future__ import annotations

from typing import Any


def _safe_id(seq: int) -> str:
    """Mermaid node IDs must be alphanumeric. Anchor on a counter."""
    return f"n{seq}"


def _label(text: str) -> str:
    """Escape text for inclusion in a mermaid node label.

    Mermaid uses ``[" "]`` for free-text labels. Inside, double quotes
    must be escaped with ``#quot;``, backslashes are fine, and the
    closing bracket is fine because we always wrap in quotes.
    """
    return text.replace('"', "#quot;")


def render_trace_tree(tree: dict[str, Any], root_label: str | None = None) -> str:
    """Render a trace tree (as produced by ``call_graph.trace``) as Mermaid.

    Tree shape: ``{symbol, kind?, confidence?, evidence?, truncated?, children: [...]}``.
    Root node has no ``kind``; children carry one (``call``, ``render``, etc.).
    Cycles are surfaced as a ``(cycle)`` annotation on the truncated child.
    """
    lines = ["flowchart TD"]
    counter = [0]
    nodes_seen: set[str] = set()

    def _walk(node: dict, parent_id: str | None) -> None:
        counter[0] += 1
        nid = _safe_id(counter[0])
        sym = node.get("symbol", "?")
        kind = node.get("kind")
        if parent_id is None:
            label = root_label or f"**{sym}**"
        else:
            conf = node.get("confidence")
            cycle = node.get("truncated") == "cycle"
            parts = [f"[{kind or 'call'}]" if kind else "", sym]
            if conf is not None:
                parts.append(f"c={conf}")
            if cycle:
                parts.append("(cycle)")
            label = " ".join(p for p in parts if p)
        lines.append(f'  {nid}["{_label(label)}"]')
        nodes_seen.add(nid)
        if parent_id is not None:
            lines.append(f"  {parent_id} --> {nid}")
        for child in node.get("children", []) or []:
            _walk(child, nid)

    _walk(tree, parent_id=None)
    return "\n".join(lines)


def render_impact_graph(
    *,
    changed_symbols: list[str],
    affected_routes: list[tuple[str, dict]],
    js_reaches: dict[str, list[dict]] | None = None,
) -> str:
    """Render impact_of_change as a top-down two-tier flowchart.

    Top tier: affected routes (by HTTP path). Middle tier: changed methods
    they reach. JS reaches (Stimulus controllers) get their own subgraph.
    Risk score appears in the route label.
    """
    js_reaches = js_reaches or {}
    lines = ["flowchart TD"]
    counter = [0]

    def _id_for(label: str, registry: dict[str, str]) -> str:
        if label in registry:
            return registry[label]
        counter[0] += 1
        nid = _safe_id(counter[0])
        registry[label] = nid
        return nid

    method_ids: dict[str, str] = {}
    route_ids: dict[str, str] = {}
    js_ids: dict[str, str] = {}

    # Pre-create method nodes so they cluster nicely.
    for sym in sorted(changed_symbols):
        nid = _id_for(sym, method_ids)
        lines.append(f'  {nid}["{_label(sym)}"]:::changed')

    for path, info in affected_routes:
        route = info["route"]
        methods = ",".join(route.get("methods") or [])
        risk = info.get("risk", 0)
        ctrl_short = route.get("controller", "?").rsplit("\\", 1)[-1]
        action = route.get("action", "?")
        label = f"{methods} {path}\\n→ {ctrl_short}::{action}\\nrisk {risk}"
        rid = _id_for(path, route_ids)
        lines.append(f'  {rid}["{_label(label)}"]:::route')
        for reach in info.get("reaches", []):
            mid = method_ids.get(reach["symbol"])
            if mid is None:
                # Unknown changed symbol — synthesize a node so the edge exists.
                mid = _id_for(reach["symbol"], method_ids)
                lines.append(f'  {mid}["{_label(reach["symbol"])}"]:::changed')
            lines.append(f"  {rid} --> {mid}")

    if js_reaches:
        for js_symbol, reaches in sorted(js_reaches.items()):
            jid = _id_for(js_symbol, js_ids)
            lines.append(f'  {jid}["{_label(js_symbol)}"]:::stimulus')
            for reach in reaches:
                mid = method_ids.get(reach["symbol"])
                if mid is None:
                    mid = _id_for(reach["symbol"], method_ids)
                    lines.append(f'  {mid}["{_label(reach["symbol"])}"]:::changed')
                lines.append(f"  {jid} --> {mid}")

    # Class definitions for visual differentiation.
    lines.append("  classDef changed fill:#fef08a,stroke:#a16207,color:#1f2937")
    lines.append("  classDef route fill:#bfdbfe,stroke:#1d4ed8,color:#1e3a8a")
    lines.append("  classDef stimulus fill:#fbcfe8,stroke:#be185d,color:#831843")
    return "\n".join(lines)


def render_cycles(cycles: list[list[str]]) -> str:
    """Render circular dependencies as a series of small mermaid graphs.

    Each cycle becomes its own ``flowchart LR`` block (one cycle = one
    SCC of size > 1). When there are no cycles, returns a single line
    so the LLM still sees a parseable mermaid block.
    """
    if not cycles:
        return "flowchart LR\n  empty[\"No circular dependencies\"]"
    blocks: list[str] = []
    for idx, cycle in enumerate(cycles):
        lines = [f"%% Cycle #{idx + 1} ({len(cycle)} symbols)", "flowchart LR"]
        ids = {sym: _safe_id(i + 1) for i, sym in enumerate(cycle)}
        for sym, nid in ids.items():
            lines.append(f'  {nid}["{_label(sym)}"]')
        for i, sym in enumerate(cycle):
            nxt = cycle[(i + 1) % len(cycle)]
            lines.append(f"  {ids[sym]} --> {ids[nxt]}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
