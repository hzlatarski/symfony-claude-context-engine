"""Tests for Leiden community detection over the unified graph.

We construct deterministic toy graphs (two triangles bridged by one
edge; star + chain) and assert the structural properties Leiden must
preserve. The randomized seed is fixed via the ``seed`` kwarg.
"""
from scripts import communities


def _make_graph(node_ids, edge_pairs):
    """Helper: build a unified-graph-shaped dict from terse inputs."""
    return {
        "nodes": {nid: {"kind": "article", "label": nid} for nid in node_ids},
        "edges": [{"from": a, "to": b, "kind": "wikilink"} for a, b in edge_pairs],
    }


def test_detect_returns_two_communities_for_bridged_triangles():
    graph = _make_graph(
        ["A", "B", "C", "D", "E", "F"],
        [("A", "B"), ("B", "C"), ("C", "A"), ("D", "E"), ("E", "F"), ("F", "D"), ("C", "D")],
    )
    result = communities.detect(graph, seed=42)
    assert isinstance(result, list)
    assert len(result) == 2
    member_sets = [frozenset(c["members"]) for c in result]
    assert frozenset({"A", "B", "C"}) in member_sets
    assert frozenset({"D", "E", "F"}) in member_sets


def test_detect_returns_hub_node_per_community():
    graph = _make_graph(
        ["H", "L1", "L2", "L3", "L4", "L5"],
        [("H", "L1"), ("H", "L2"), ("H", "L3"), ("H", "L4"), ("H", "L5")],
    )
    result = communities.detect(graph, seed=42)
    star = next(c for c in result if "H" in c["members"])
    assert star["hub_node"] == "H"


def test_detect_skips_singleton_communities_below_min_size():
    graph = _make_graph(
        ["A", "B", "C", "X"],
        [("A", "B"), ("B", "C"), ("C", "A")],
    )
    result = communities.detect(graph, seed=42, min_size=2)
    assert all(c["size"] >= 2 for c in result)
    assert all("X" not in c["members"] for c in result)


def test_detect_each_community_has_community_id_size_and_label():
    graph = _make_graph(["A", "B", "C"], [("A", "B"), ("B", "C"), ("C", "A")])
    result = communities.detect(graph, seed=42)
    assert len(result) == 1
    c = result[0]
    assert c["community_id"] == 0
    assert c["size"] == 3
    assert isinstance(c["label"], str) and len(c["label"]) > 0
    assert c["hub_node"] in {"A", "B", "C"}
