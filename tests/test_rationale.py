"""Tests for inline rationale extraction (Feature #2).

Covers the call-graph parser lifting WHY / HACK / TODO / @deprecated
comments onto the method or class they annotate, and unified_graph
materializing them as ``note:`` leaf nodes with ``annotates`` edges.
"""
from __future__ import annotations

from pathlib import Path

from scripts.parsers import call_graph
from scripts import unified_graph

FIXTURE = Path(__file__).parent / "fixtures" / "php_call_graph" / "rationale"


def _parse() -> dict:
    return call_graph.parse(FIXTURE)


def test_inline_hack_and_todo_attach_to_method():
    result = _parse()
    sym = result["symbols"]["App\\Rationale::withInlineHack"]
    tags = {r["tag"] for r in sym.get("rationale", [])}
    assert "HACK" in tags
    assert "TODO" in tags


def test_deprecated_docblock_attaches_to_method():
    result = _parse()
    sym = result["symbols"]["App\\Rationale::oldMethod"]
    tags = {r["tag"] for r in sym.get("rationale", [])}
    assert "DEPRECATED" in tags


def test_class_level_why_attaches_to_class():
    result = _parse()
    cls = result["classes"]["App\\Rationale"]
    tags = {r["tag"] for r in cls.get("rationale", [])}
    assert "WHY" in tags


def test_summary_first_deprecated_docblock_is_captured():
    """H1 regression: @deprecated on line 3+ of a docblock, not line 1."""
    result = _parse()
    sym = result["symbols"]["App\\Rationale::summaryFirstDeprecated"]
    tags = {r["tag"] for r in sym.get("rationale", [])}
    assert "DEPRECATED" in tags
    dep = next(r for r in sym["rationale"] if r["tag"] == "DEPRECATED")
    assert "PriceService" in dep["text"]


def test_long_docblock_deprecated_attaches_to_its_method_not_class():
    """H2 regression: a multi-line docblock must still bind to the method
    below its closing */ (attribution measured from the comment's end)."""
    result = _parse()
    # DEPRECATED must land on the method, NOT leak up to the class.
    cls_tags = [r for r in result["classes"]["App\\Rationale"].get("rationale", [])
                if r["tag"] == "DEPRECATED"]
    assert not cls_tags


def test_prose_note_that_is_not_captured_as_rationale():
    """M2 regression: uppercase-only bare tags — 'Note that…' is not a NOTE."""
    result = _parse()
    sym = result["symbols"]["App\\Rationale::clean"]
    assert not sym.get("rationale")
    # The 'Note that this once handled proration' prose inside the
    # summary-first docblock must not mint a NOTE either.
    summ = result["symbols"]["App\\Rationale::summaryFirstDeprecated"]
    assert all(r["tag"] != "NOTE" for r in summ.get("rationale", []))


def test_ordinary_comment_is_not_captured():
    result = _parse()
    sym = result["symbols"]["App\\Rationale::clean"]
    assert not sym.get("rationale")


def test_rationale_text_is_captured():
    result = _parse()
    sym = result["symbols"]["App\\Rationale::withInlineHack"]
    hack = next(r for r in sym["rationale"] if r["tag"] == "HACK")
    assert "repaint" in hack["text"]


def test_unified_graph_emits_note_nodes_and_annotates_edges():
    result = _parse()
    graph = unified_graph.build(call_graph=result, knowledge_root=Path("/nonexistent"))

    note_nodes = {nid: n for nid, n in graph["nodes"].items() if nid.startswith("note:")}
    assert note_nodes, "expected at least one note: node"
    assert all(n["kind"] == "rationale" for n in note_nodes.values())

    annotates = [e for e in graph["edges"] if e["kind"] == "annotates"]
    assert annotates
    # Every annotates edge targets a note node and originates from a symbol/class.
    for e in annotates:
        assert e["to"].startswith("note:")
        assert e["from"].startswith(("symbol:", "class:"))


def test_note_nodes_are_leaf_degree_one():
    """Notes must not become hubs — each has exactly one inbound annotates edge."""
    result = _parse()
    graph = unified_graph.build(call_graph=result, knowledge_root=Path("/nonexistent"))
    for nid in (n for n in graph["nodes"] if n.startswith("note:")):
        touching = [e for e in graph["edges"] if e["from"] == nid or e["to"] == nid]
        assert len(touching) == 1
