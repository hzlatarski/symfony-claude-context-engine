"""Tests for the unified knowledge graph fuser.

The fuser combines the call graph (from parsers.call_graph) with articles
in knowledge/concepts/ etc. We test against synthetic tmp_path knowledge
dirs and minimal call-graph dicts — no dependence on live project state.
"""
from pathlib import Path

from scripts import unified_graph


def test_build_with_empty_inputs_returns_empty_graph(tmp_path):
    call_graph = {"symbols": {}, "edges": [], "classes": {}}
    result = unified_graph.build(call_graph=call_graph, knowledge_root=tmp_path)
    assert result == {"nodes": {}, "edges": []}


def test_articles_in_concepts_become_article_nodes(tmp_path):
    concepts = tmp_path / "concepts"
    concepts.mkdir()
    (concepts / "foo.md").write_text("---\ntitle: Foo Article\ntype: fact\nconfidence: 0.8\n---\nbody", encoding="utf-8")
    (concepts / "bar.md").write_text("# Bar\n\nNo frontmatter.", encoding="utf-8")

    call_graph = {"symbols": {}, "edges": [], "classes": {}}
    result = unified_graph.build(call_graph=call_graph, knowledge_root=tmp_path)

    assert "article:concepts/foo" in result["nodes"]
    assert "article:concepts/bar" in result["nodes"]
    foo = result["nodes"]["article:concepts/foo"]
    assert foo["kind"] == "article"
    assert foo["label"] == "Foo Article"
    assert foo["type"] == "fact"
    assert foo["confidence"] == 0.8


def test_articles_in_other_subdirs_are_indexed(tmp_path):
    (tmp_path / "connections").mkdir()
    (tmp_path / "connections" / "link.md").write_text("# Link", encoding="utf-8")
    (tmp_path / "qa").mkdir()
    (tmp_path / "qa" / "q1.md").write_text("# Q1", encoding="utf-8")

    call_graph = {"symbols": {}, "edges": [], "classes": {}}
    result = unified_graph.build(call_graph=call_graph, knowledge_root=tmp_path)

    assert "article:connections/link" in result["nodes"]
    assert "article:qa/q1" in result["nodes"]


def test_article_without_title_falls_back_to_slug(tmp_path):
    (tmp_path / "concepts").mkdir()
    (tmp_path / "concepts" / "no-title.md").write_text("just text", encoding="utf-8")
    result = unified_graph.build(call_graph={"symbols": {}, "edges": [], "classes": {}}, knowledge_root=tmp_path)
    assert result["nodes"]["article:concepts/no-title"]["label"] == "no-title"


def test_wikilinks_become_article_to_article_edges(tmp_path):
    (tmp_path / "concepts").mkdir()
    (tmp_path / "concepts" / "src.md").write_text(
        "See [[concepts/dst]] and [[concepts/other]]{depends_on}.",
        encoding="utf-8",
    )
    (tmp_path / "concepts" / "dst.md").write_text("target", encoding="utf-8")
    (tmp_path / "concepts" / "other.md").write_text("target2", encoding="utf-8")

    result = unified_graph.build(call_graph={"symbols": {}, "edges": [], "classes": {}}, knowledge_root=tmp_path)

    pairs = [(e["from"], e["to"], e.get("relation")) for e in result["edges"] if e["kind"] == "wikilink"]
    assert ("article:concepts/src", "article:concepts/dst", None) in pairs
    assert ("article:concepts/src", "article:concepts/other", "depends_on") in pairs


def test_wikilink_to_missing_article_is_skipped(tmp_path):
    (tmp_path / "concepts").mkdir()
    (tmp_path / "concepts" / "src.md").write_text("See [[concepts/ghost]].", encoding="utf-8")

    result = unified_graph.build(call_graph={"symbols": {}, "edges": [], "classes": {}}, knowledge_root=tmp_path)
    wikilink_edges = [e for e in result["edges"] if e["kind"] == "wikilink"]
    assert wikilink_edges == []


def test_src_anchors_become_article_to_file_edges(tmp_path):
    (tmp_path / "concepts").mkdir()
    (tmp_path / "concepts" / "a.md").write_text(
        "Refs [src:src/Foo/Bar.php] and [src:src/Other.php] in body.",
        encoding="utf-8",
    )

    result = unified_graph.build(call_graph={"symbols": {}, "edges": [], "classes": {}}, knowledge_root=tmp_path)

    assert "file:src/Foo/Bar.php" in result["nodes"]
    assert result["nodes"]["file:src/Foo/Bar.php"]["kind"] == "file"
    pairs = [(e["from"], e["to"]) for e in result["edges"] if e["kind"] == "cites"]
    assert ("article:concepts/a", "file:src/Foo/Bar.php") in pairs
    assert ("article:concepts/a", "file:src/Other.php") in pairs


def test_duplicate_src_anchors_dedupe_edges(tmp_path):
    (tmp_path / "concepts").mkdir()
    (tmp_path / "concepts" / "a.md").write_text(
        "[src:src/Foo.php] mentioned twice [src:src/Foo.php].",
        encoding="utf-8",
    )
    result = unified_graph.build(call_graph={"symbols": {}, "edges": [], "classes": {}}, knowledge_root=tmp_path)
    cites = [e for e in result["edges"] if e["kind"] == "cites"]
    assert len(cites) == 1


def test_call_graph_symbols_and_classes_become_nodes(tmp_path):
    call_graph = {
        "symbols": {
            "App\\Service\\Foo::bar": {"file": "src/Service/Foo.php", "line": 10, "class": "App\\Service\\Foo"},
            "App\\Service\\Baz::qux": {"file": "src/Service/Baz.php", "line": 5, "class": "App\\Service\\Baz"},
        },
        "edges": [
            {"from": "App\\Service\\Foo::bar", "to": "App\\Service\\Baz::qux", "kind": "call", "confidence": 1.0, "evidence": "$this->baz->qux()"},
            {"from": "App\\Service\\Foo::bar", "to": "template:foo/index.html.twig", "kind": "render", "confidence": 1.0, "evidence": "$this->render('foo/index.html.twig')"},
        ],
        "classes": {
            "App\\Service\\Foo": {"file": "src/Service/Foo.php", "extends": ""},
            "App\\Service\\Baz": {"file": "src/Service/Baz.php", "extends": ""},
        },
    }
    result = unified_graph.build(call_graph=call_graph, knowledge_root=tmp_path)

    assert result["nodes"]["class:App\\Service\\Foo"]["kind"] == "class"
    assert result["nodes"]["symbol:App\\Service\\Foo::bar"]["kind"] == "symbol"
    assert result["nodes"]["file:src/Service/Foo.php"]["kind"] == "file"
    assert result["nodes"]["template:foo/index.html.twig"]["kind"] == "template"

    kinds = {(e["from"], e["to"]): e["kind"] for e in result["edges"]}
    assert kinds[("file:src/Service/Foo.php", "class:App\\Service\\Foo")] == "contains"
    assert kinds[("class:App\\Service\\Foo", "symbol:App\\Service\\Foo::bar")] == "defines"
    assert kinds[("symbol:App\\Service\\Foo::bar", "symbol:App\\Service\\Baz::qux")] == "call"
    assert kinds[("symbol:App\\Service\\Foo::bar", "template:foo/index.html.twig")] == "render"


def test_unresolved_fetch_placeholder_edges_are_skipped(tmp_path):
    """An unresolved JS fetch() edge must not become a malformed symbol: node."""
    call_graph = {
        "symbols": {
            "js:arena::connect": {"file": "assets/controllers/arena_controller.js", "class": ""},
        },
        "edges": [
            {"from": "js:arena::connect", "to": "fetch:POST /api/session", "kind": "call", "confidence": 0.7},
        ],
        "classes": {},
    }
    result = unified_graph.build(call_graph=call_graph, knowledge_root=tmp_path)

    # No node ID may carry the fetch: placeholder (no "symbol:fetch:..." garbage).
    assert not any("fetch:" in nid for nid in result["nodes"])
    # The dead edge is dropped entirely.
    assert result["edges"] == []
