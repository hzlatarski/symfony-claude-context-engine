"""Tests for external graph exports (GraphML / Cypher / HTML / JSON).

The emitters are pure: input is a ``{nodes, edges}`` graph dict (the shape
produced by ``unified_graph.build``), output is a ``str``. We test against
small synthetic graphs — no live project / network / Chroma dependence.
"""
import json
import xml.etree.ElementTree as ET

from scripts import export_graph


# ── Fixtures ───────────────────────────────────────────────────────────


def _small_graph() -> dict:
    return {
        "nodes": {
            "article:concepts/foo": {
                "kind": "article",
                "label": "Foo Article",
                "type": "fact",
                "confidence": 0.8,
            },
            "file:src/Foo.php": {"kind": "file", "label": "src/Foo.php"},
            "class:App\\Foo": {"kind": "class", "label": "Foo"},
        },
        "edges": [
            {"from": "article:concepts/foo", "to": "file:src/Foo.php", "kind": "cites"},
            {
                "from": "file:src/Foo.php",
                "to": "class:App\\Foo",
                "kind": "contains",
            },
            {
                "from": "article:concepts/foo",
                "to": "class:App\\Foo",
                "kind": "wikilink",
                "confidence": 0.9,
                "relation": "depends_on",
            },
        ],
    }


def _special_chars_graph() -> dict:
    return {
        "nodes": {
            "article:concepts/spec": {
                "kind": "article",
                "label": 'Title with <, & and "quotes"',
            },
            "file:src/A & B.php": {"kind": "file", "label": "src/A & B.php"},
        },
        "edges": [
            {"from": "article:concepts/spec", "to": "file:src/A & B.php", "kind": "cites"},
        ],
    }


def _empty_graph() -> dict:
    return {"nodes": {}, "edges": []}


# ── GraphML ────────────────────────────────────────────────────────────


def test_graphml_parses_as_valid_xml():
    out = export_graph.to_graphml(_small_graph())
    root = ET.fromstring(out)  # raises if not well-formed
    assert root.tag.endswith("graphml")


def test_graphml_contains_node_ids():
    out = export_graph.to_graphml(_small_graph())
    root = ET.fromstring(out)
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    ids = {n.get("id") for n in root.iter("{http://graphml.graphdrawing.org/xmlns}node")}
    assert "article:concepts/foo" in ids
    assert "file:src/Foo.php" in ids
    assert "class:App\\Foo" in ids


def test_graphml_declares_key_elements():
    out = export_graph.to_graphml(_small_graph())
    root = ET.fromstring(out)
    keys = list(root.iter("{http://graphml.graphdrawing.org/xmlns}key"))
    assert keys, "expected at least one <key> declaration"
    attr_names = {k.get("attr.name") for k in keys}
    assert "label" in attr_names
    assert "kind" in attr_names


def test_graphml_has_one_edge_per_edge():
    out = export_graph.to_graphml(_small_graph())
    root = ET.fromstring(out)
    edges = list(root.iter("{http://graphml.graphdrawing.org/xmlns}edge"))
    assert len(edges) == 3
    sources = {e.get("source") for e in edges}
    targets = {e.get("target") for e in edges}
    assert "article:concepts/foo" in sources
    assert "class:App\\Foo" in targets


def test_graphml_escapes_special_characters():
    out = export_graph.to_graphml(_special_chars_graph())
    # Raw special chars must not leak into the serialized data values.
    assert "Title with <, &" not in out
    assert "&lt;" in out
    assert "&amp;" in out
    # Still valid XML and the escaped label round-trips.
    root = ET.fromstring(out)
    found = False
    for data in root.iter("{http://graphml.graphdrawing.org/xmlns}data"):
        if data.text and 'Title with <, & and "quotes"' in data.text:
            found = True
    assert found, "escaped label should decode back to original text"


def test_graphml_empty_graph_is_valid():
    out = export_graph.to_graphml(_empty_graph())
    root = ET.fromstring(out)
    assert list(root.iter("{http://graphml.graphdrawing.org/xmlns}node")) == []
    assert list(root.iter("{http://graphml.graphdrawing.org/xmlns}edge")) == []


# ── Cypher ─────────────────────────────────────────────────────────────


def test_cypher_has_merge_per_node():
    out = export_graph.to_cypher(_small_graph())
    merge_lines = [ln for ln in out.splitlines() if ln.strip().startswith("MERGE (n:Node")]
    assert len(merge_lines) == 3
    assert any('id: "article:concepts/foo"' in ln for ln in merge_lines)


def test_cypher_has_relationship_per_edge():
    out = export_graph.to_cypher(_small_graph())
    rel_lines = [ln for ln in out.splitlines() if "MERGE (a)-[:REL" in ln]
    assert len(rel_lines) == 3
    assert any('kind:"cites"' in ln or 'kind: "cites"' in ln for ln in rel_lines)


def test_cypher_escapes_quotes_and_backslashes():
    out = export_graph.to_cypher(_special_chars_graph())
    # Double-quotes inside a label must be backslash-escaped.
    assert '\\"quotes\\"' in out
    # A node id with a backslash (class:App\Foo style) must double the backslash.
    bs_graph = {
        "nodes": {"class:App\\Foo": {"kind": "class", "label": "Foo"}},
        "edges": [],
    }
    bs_out = export_graph.to_cypher(bs_graph)
    assert "class:App\\\\Foo" in bs_out


def test_cypher_is_deterministic():
    g = _small_graph()
    assert export_graph.to_cypher(g) == export_graph.to_cypher(g)


def test_cypher_empty_graph():
    out = export_graph.to_cypher(_empty_graph())
    assert "MERGE (n:Node" not in out
    assert "MERGE (a)-[:REL" not in out


# ── HTML ───────────────────────────────────────────────────────────────


def test_html_contains_vis_network_script_tag():
    out = export_graph.to_html(_small_graph())
    assert "vis-network/standalone/umd/vis-network.min.js" in out
    assert "<script" in out


def test_html_contains_every_node_id():
    out = export_graph.to_html(_small_graph())
    for nid in _small_graph()["nodes"]:
        # ids are embedded in the JSON blob (JSON-escaped backslashes).
        assert json.dumps(nid)[1:-1] in out


def test_html_has_title():
    out = export_graph.to_html(_small_graph())
    assert "Knowledge Graph" in out


def test_html_empty_graph_is_valid_string():
    out = export_graph.to_html(_empty_graph())
    assert "vis-network" in out
    assert "Knowledge Graph" in out


# ── JSON ───────────────────────────────────────────────────────────────


def test_json_round_trips():
    g = _small_graph()
    out = export_graph.to_json(g)
    assert json.loads(out) == g


def test_json_is_sorted_and_indented():
    out = export_graph.to_json(_small_graph())
    assert "\n" in out
    assert "  " in out  # indent=2


def test_json_empty_graph():
    out = export_graph.to_json(_empty_graph())
    assert json.loads(out) == {"nodes": {}, "edges": []}
