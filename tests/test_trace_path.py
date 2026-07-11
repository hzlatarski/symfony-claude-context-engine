"""Tests for trace_path — shortest connection between two unified-graph nodes."""
from __future__ import annotations

import pytest

from scripts import mcp_server


@pytest.fixture()
def synthetic_graph(monkeypatch):
    """A small unified graph: article A cites a file whose class defines a
    symbol; article B also cites the same file. A and B connect through it.
    An isolated node D sits in its own component.
    """
    graph = {
        "nodes": {
            "article:concepts/a": {"kind": "article", "label": "Concept A"},
            "article:concepts/b": {"kind": "article", "label": "Concept B"},
            "file:src/Foo.php": {"kind": "file", "label": "src/Foo.php"},
            "class:App\\Foo": {"kind": "class", "label": "Foo"},
            "symbol:App\\Foo::bar": {"kind": "symbol", "label": "bar"},
            "article:concepts/d": {"kind": "article", "label": "Concept D"},
        },
        "edges": [
            {"from": "article:concepts/a", "to": "file:src/Foo.php", "kind": "cites"},
            {"from": "article:concepts/b", "to": "file:src/Foo.php", "kind": "cites"},
            {"from": "file:src/Foo.php", "to": "class:App\\Foo", "kind": "contains"},
            {"from": "class:App\\Foo", "to": "symbol:App\\Foo::bar", "kind": "defines"},
        ],
    }
    monkeypatch.setattr(mcp_server._cache, "get_unified_graph", lambda: graph)
    return graph


def test_direct_path_between_article_and_symbol(synthetic_graph):
    out = mcp_server._build_trace_path("article:concepts/a", "symbol:App\\Foo::bar")
    assert "Hops: **3**" in out
    assert "cites" in out
    assert "contains" in out
    assert "defines" in out


def test_two_articles_connect_through_shared_file(synthetic_graph):
    out = mcp_server._build_trace_path("article:concepts/a", "article:concepts/b")
    assert "Hops: **2**" in out
    assert "file:src/Foo.php" in out


def test_missing_node_reports_friendly(synthetic_graph):
    out = mcp_server._build_trace_path("article:concepts/a", "article:concepts/nope")
    assert "not found" in out.lower()


def test_same_node_is_zero_length(synthetic_graph):
    out = mcp_server._build_trace_path("article:concepts/a", "article:concepts/a")
    assert "same node" in out.lower()


def test_disconnected_component_reports_no_path(synthetic_graph):
    out = mcp_server._build_trace_path("article:concepts/a", "article:concepts/d")
    assert "No path" in out


def test_depth_limit_blocks_long_path(synthetic_graph):
    # a→file→class→symbol is 3 hops; a max_depth of 2 can't reach the symbol.
    out = mcp_server._build_trace_path(
        "article:concepts/a", "symbol:App\\Foo::bar", max_depth=2
    )
    assert "No path" in out
