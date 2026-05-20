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
    for member_indices in partition:
        if len(member_indices) < min_size:
            continue
        members = [node_ids[i] for i in member_indices]
        hub_idx = max(member_indices, key=lambda i: degrees[i])
        hub_node = node_ids[hub_idx]
        member_labels = [graph["nodes"][nid].get("label", nid) for nid in members]
        results.append({
            "members": members,
            "hub_node": hub_node,
            "size": len(members),
            "label": label_for(member_labels),
        })

    # community_id is assigned post-sort so it reflects rank (largest = 0).
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


import hashlib
import json
from pathlib import Path


def _signature(graph: dict, seed: int) -> str:
    """Stable hash over the graph's node IDs and edge endpoints.

    Edge kinds/relations/confidences don't affect Leiden output so we
    exclude them from the signature — the cache survives metadata-only
    edits, e.g. a wikilink relation tweak.
    """
    h = hashlib.sha1()
    h.update(f"seed={seed}\n".encode())
    for nid in sorted(graph["nodes"]):
        h.update(f"n:{nid}\n".encode())
    edge_keys = sorted(
        (e["from"], e["to"]) for e in graph["edges"]
    )
    for a, b in edge_keys:
        h.update(f"e:{a}->{b}\n".encode())
    return h.hexdigest()


def load_or_compute(graph: dict, *, cache_path: Path, seed: int = 42, min_size: int = 2) -> list[dict]:
    """Return communities for ``graph``, loading from ``cache_path`` when the
    graph signature matches.

    Cache file shape:
        ``{"signature": "<sha1>", "seed": 42, "min_size": 2,
           "communities": [<community records>]}``
    """
    cache_path = Path(cache_path)
    sig = _signature(graph, seed)

    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if data.get("signature") == sig and data.get("min_size") == min_size:
                return data["communities"]
        except (json.JSONDecodeError, KeyError):
            pass  # corrupt cache — recompute

    result = detect(graph, seed=seed, min_size=min_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {"signature": sig, "seed": seed, "min_size": min_size, "communities": result},
            indent=2,
        ),
        encoding="utf-8",
    )
    return result
