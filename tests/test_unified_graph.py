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
