"""Tests for ``get_neighborhood`` — codegraph_explore-inspired MCP tool.

Covers the three things that can silently break:
- grouping multiple symbols from one file under a single heading
- per-symbol + total budget truncation
- bare-file header path (file: nodes with no `contains` edges)
- node-not-found friendly error
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts import mcp_server


@pytest.fixture
def fake_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Synthesize a tiny project + matching unified/call graphs.

    Layout:
        src/Foo/Bar.php    — class Bar with methods a() and b()
        src/Foo/Baz.php    — class Baz with method c()
        config/services.yaml — bare file (no class), only reachable via cites edge

    Graph: Bar::a calls Bar::b and Baz::c; file Bar.php contains class Bar.
    """
    (tmp_path / "src" / "Foo").mkdir(parents=True)
    (tmp_path / "src" / "Foo" / "Bar.php").write_text(
        "<?php\n"
        "class Bar {\n"
        "  public function a() { return 1; }\n"
        "  public function b() { return 2; }\n"
        "}\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "Foo" / "Baz.php").write_text(
        "<?php\n"
        "class Baz {\n"
        "  public function c() { return 3; }\n"
        "}\n",
        encoding="utf-8",
    )
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "services.yaml").write_text(
        "services:\n  foo: bar\n", encoding="utf-8"
    )

    fake_unified = {
        "nodes": {
            "symbol:Foo\\Bar::a": {"kind": "symbol", "label": "a"},
            "symbol:Foo\\Bar::b": {"kind": "symbol", "label": "b"},
            "symbol:Foo\\Baz::c": {"kind": "symbol", "label": "c"},
            "file:src/Foo/Bar.php": {"kind": "file", "label": "src/Foo/Bar.php"},
            "file:src/Foo/Baz.php": {"kind": "file", "label": "src/Foo/Baz.php"},
            "file:config/services.yaml": {
                "kind": "file",
                "label": "config/services.yaml",
            },
            "class:Foo\\Bar": {"kind": "class", "label": "Bar"},
        },
        "edges": [
            {"from": "symbol:Foo\\Bar::a", "to": "symbol:Foo\\Bar::b", "kind": "call"},
            {"from": "symbol:Foo\\Bar::a", "to": "symbol:Foo\\Baz::c", "kind": "call"},
            {"from": "file:src/Foo/Bar.php", "to": "class:Foo\\Bar", "kind": "contains"},
        ],
    }
    fake_call_graph = {
        "symbols": {
            "Foo\\Bar::a": {"file": "src/Foo/Bar.php", "line": 3, "end_line": 3},
            "Foo\\Bar::b": {"file": "src/Foo/Bar.php", "line": 4, "end_line": 4},
            "Foo\\Baz::c": {"file": "src/Foo/Baz.php", "line": 3, "end_line": 3},
        },
    }

    monkeypatch.setattr(mcp_server._cache, "get_unified_graph", lambda: fake_unified)
    monkeypatch.setattr(mcp_server._cache, "get_call_graph", lambda: fake_call_graph)
    monkeypatch.setattr(mcp_server, "PROJECT_ROOT", tmp_path)
    return tmp_path


def test_node_not_found_returns_friendly_string():
    """Unknown node IDs must not raise — MCP callers prefer a string."""
    # No fixture needed; the empty default unified graph still has no `symbol:Nope`.
    result = mcp_server._build_neighborhood("symbol:Does\\Not::exist")
    assert "Node not found" in result
    assert "Does\\Not::exist" in result


def test_groups_symbols_from_same_file_under_one_heading(fake_project):
    """Bar::a and Bar::b both live in Bar.php — must render under one heading."""
    out = mcp_server._build_neighborhood("symbol:Foo\\Bar::a", depth=2)
    assert out.count("### `src/Foo/Bar.php`") == 1, out
    assert "Foo\\Bar::a" in out
    assert "Foo\\Bar::b" in out
    # Different file gets its own heading.
    assert "### `src/Foo/Baz.php`" in out
    assert "Foo\\Baz::c" in out


def test_source_bundle_includes_actual_source_lines(fake_project):
    out = mcp_server._build_neighborhood("symbol:Foo\\Bar::a", depth=2)
    assert "public function a() { return 1; }" in out
    assert "public function b() { return 2; }" in out
    assert "public function c() { return 3; }" in out


def test_php_files_get_php_language_fence(fake_project):
    out = mcp_server._build_neighborhood("symbol:Foo\\Bar::a", depth=2)
    assert "```php" in out


def test_include_source_false_suppresses_source_bundle(fake_project):
    out = mcp_server._build_neighborhood(
        "symbol:Foo\\Bar::a", depth=2, include_source=False
    )
    assert "## Source bundle" not in out
    assert "public function a()" not in out
    # Relationship map must still render.
    assert "Outgoing" in out or "Incoming" in out


def test_bare_file_gets_header_excerpt_with_correct_fence(fake_project):
    """services.yaml has no `contains` edge → must render a header excerpt."""
    g = mcp_server._cache.get_unified_graph()
    # Connect bare YAML file to the root node so BFS reaches it at depth=1.
    g["edges"].append(
        {
            "from": "symbol:Foo\\Bar::a",
            "to": "file:config/services.yaml",
            "kind": "call",
        }
    )
    out = mcp_server._build_neighborhood("symbol:Foo\\Bar::a", depth=1)
    assert "## File headers" in out
    assert "```yaml" in out
    assert "services:" in out
    assert "foo: bar" in out


def test_file_with_classes_does_not_double_render_as_bare(fake_project):
    """Bar.php has a contains→class edge, so it must NOT appear in headers section."""
    out = mcp_server._build_neighborhood("symbol:Foo\\Bar::a", depth=2)
    # Source bundle has Bar.php; header section, if present, must not duplicate it.
    if "## File headers" in out:
        headers_section = out.split("## File headers", 1)[1]
        assert "src/Foo/Bar.php" not in headers_section


def test_per_symbol_budget_truncates_long_methods(fake_project):
    """A symbol spanning more lines than `max_source_lines` must get truncated."""
    cg = mcp_server._cache.get_call_graph()
    # Make Bar::a "look" 999 lines long even though the underlying file is short.
    cg["symbols"]["Foo\\Bar::a"]["end_line"] = 1001
    out = mcp_server._build_neighborhood(
        "symbol:Foo\\Bar::a", depth=2, max_source_lines=10
    )
    assert "more lines truncated" in out


def test_total_budget_halts_emission(fake_project):
    """5x max_source_lines is the hard cap across all symbols."""
    cg = mcp_server._cache.get_call_graph()
    # Force every symbol to be 1000 lines so budget bites fast.
    for s in cg["symbols"].values():
        s["end_line"] = s["line"] + 999
    out = mcp_server._build_neighborhood(
        "symbol:Foo\\Bar::a", depth=2, max_source_lines=10
    )
    assert "Source budget exhausted" in out


def test_depth_is_clamped_to_one_through_three(fake_project):
    """Same contract as get_unified_neighbors — should not raise on out-of-range."""
    mcp_server._build_neighborhood("symbol:Foo\\Bar::a", depth=0)
    mcp_server._build_neighborhood("symbol:Foo\\Bar::a", depth=999)
