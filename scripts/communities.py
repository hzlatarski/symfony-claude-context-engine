"""Leiden community detection over the unified knowledge graph.

Runs `leidenalg.find_partition` with `ModularityVertexPartition` on the
undirected projection of the unified graph (edge directions are
informational — communities are about connectivity, not flow).
Communities below `min_size` are dropped. Each surviving community gets
a deterministic label assembled from its highest-degree node labels.

Public surface:

    detect(graph, *, seed=42, min_size=2) -> list[dict]
    label_for(node_labels) -> str
"""
from __future__ import annotations


def detect(graph: dict, *, seed: int = 42, min_size: int = 2) -> list[dict]:
    """Return one record per community in the unified graph.

    Each record:
        ``{community_id: int, members: list[str], hub_node: str,
           size: int, label: str}``

    Communities are sorted by size (descending), then by ``hub_node`` id
    for ties — making the output deterministic across runs given the
    same seed.

    Args:
        graph: ``{nodes, edges}`` from ``unified_graph.build``.
        seed: Leiden's RNG seed. Same seed + same graph = same partition.
        min_size: Drop communities smaller than this. ``2`` is the
            default because singletons are noise.
    """
    import igraph as ig
    import leidenalg as la

    node_ids = list(graph["nodes"].keys())
    if not node_ids:
        return []

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    ig_edges = []
    for e in graph["edges"]:
        src = id_to_idx.get(e["from"])
        dst = id_to_idx.get(e["to"])
        if src is None or dst is None or src == dst:
            continue
        ig_edges.append((src, dst))

    g = ig.Graph(n=len(node_ids), edges=ig_edges, directed=False)
    partition = la.find_partition(
        g,
        la.ModularityVertexPartition,
        seed=seed,
    )

    degrees = g.degree()
    results: list[dict] = []
    for community_id, member_indices in enumerate(partition):
        if len(member_indices) < min_size:
            continue
        members = [node_ids[i] for i in member_indices]
        hub_idx = max(member_indices, key=lambda i: degrees[i])
        hub_node = node_ids[hub_idx]
        member_labels = [graph["nodes"][nid].get("label", nid) for nid in members]
        record = {
            "community_id": community_id,
            "members": members,
            "hub_node": hub_node,
            "size": len(members),
            "label": label_for(member_labels),
        }
        results.append(record)

    results.sort(key=lambda c: (-c["size"], c["hub_node"]))
    for idx, c in enumerate(results):
        c["community_id"] = idx
    return results


def label_for(node_labels: list[str]) -> str:
    """Build a deterministic community label from member labels.

    Picks the 3 shortest non-empty labels (proxy for "most central
    concept names") joined with " / ". If the community has fewer than
    3 labels, joins what's available. Never returns empty — falls back
    to ``"unlabeled"`` if all labels are empty.
    """
    cleaned = [lbl for lbl in node_labels if lbl]
    if not cleaned:
        return "unlabeled"
    cleaned.sort(key=len)
    return " / ".join(cleaned[:3])
