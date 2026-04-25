"""Tests for the PHP call-graph parser.

Uses small, dedicated PHP fixture files so each resolution rule
(constructor injection, typed local var, static call, Doctrine repository,
inheritance, render) can be asserted in isolation.

A separate suite of smoke tests at the bottom runs against the real
Symfony codebase to catch regressions in aggregate stats.
"""
from __future__ import annotations

from pathlib import Path

from scripts.parsers import PROJECT_ROOT, call_graph

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "php_call_graph"


def _parse(project_root: Path) -> dict:
    """Convenience wrapper: parse a fixture project rooted at FIXTURE_ROOT/<name>."""
    return call_graph.parse(project_root)


class TestParseShape:
    def test_returns_symbols_edges_stats(self):
        result = _parse(FIXTURE_ROOT / "shape")
        assert "symbols" in result
        assert "edges" in result
        assert "stats" in result
        assert isinstance(result["symbols"], dict)
        assert isinstance(result["edges"], list)
        assert isinstance(result["stats"], dict)

    def test_stats_counts_files(self):
        result = _parse(FIXTURE_ROOT / "shape")
        assert result["stats"]["total_files"] >= 1


class TestSymbolExtraction:
    def test_method_symbol_uses_fqcn_double_colon_method(self):
        result = _parse(FIXTURE_ROOT / "shape")
        assert "App\\Foo::bar" in result["symbols"]

    def test_symbol_records_file_and_line(self):
        result = _parse(FIXTURE_ROOT / "shape")
        sym = result["symbols"]["App\\Foo::bar"]
        assert sym["file"] == "src/Foo.php"
        assert sym["line"] >= 1
        assert sym["kind"] == "method"
        assert sym["visibility"] == "public"

    def test_multiple_methods_in_one_class(self):
        result = _parse(FIXTURE_ROOT / "multi_method")
        assert "App\\Multi::one" in result["symbols"]
        assert "App\\Multi::two" in result["symbols"]
        assert "App\\Multi::three" in result["symbols"]

    def test_records_visibility(self):
        result = _parse(FIXTURE_ROOT / "multi_method")
        assert result["symbols"]["App\\Multi::one"]["visibility"] == "public"
        assert result["symbols"]["App\\Multi::two"]["visibility"] == "protected"
        assert result["symbols"]["App\\Multi::three"]["visibility"] == "private"

    def test_total_symbols_in_stats(self):
        result = _parse(FIXTURE_ROOT / "multi_method")
        assert result["stats"]["total_symbols"] == 3

    def test_symbol_records_end_line(self):
        """end_line is the closing brace's line — needed for diff-to-symbol mapping."""
        result = _parse(FIXTURE_ROOT / "multi_method")
        sym = result["symbols"]["App\\Multi::one"]
        assert "end_line" in sym
        assert sym["end_line"] >= sym["line"]
        # one() is `public function one(): void {}` — start and end on same line
        assert sym["end_line"] == sym["line"]


class TestConstructorInjectionEdges:
    def test_promoted_ctor_param_resolves_call(self):
        result = _parse(FIXTURE_ROOT / "ctor_injection")
        edge = _find_edge(
            result,
            "App\\Controller\\PromotedController::run",
            "App\\Service\\SessionService::start",
        )
        assert edge is not None, f"edge not found in {[(e['from'], e['to']) for e in result['edges']]}"
        assert edge["kind"] == "call"
        assert edge["confidence"] == 1.0

    def test_classic_ctor_assignment_resolves_call(self):
        result = _parse(FIXTURE_ROOT / "ctor_injection")
        edge = _find_edge(
            result,
            "App\\Controller\\ClassicController::run",
            "App\\Service\\SessionService::start",
        )
        assert edge is not None
        assert edge["confidence"] == 1.0

    def test_evidence_records_source_location(self):
        result = _parse(FIXTURE_ROOT / "ctor_injection")
        edge = _find_edge(
            result,
            "App\\Controller\\PromotedController::run",
            "App\\Service\\SessionService::start",
        )
        # evidence is "<rel_path>:<line>"
        assert edge["evidence"].startswith("src/Controller/PromotedController.php:")

    def test_unresolved_dynamic_call_skipped(self):
        """Untyped $x->method() must not produce an edge."""
        result = _parse(FIXTURE_ROOT / "ctor_injection")
        for edge in result["edges"]:
            assert "::dynamic" not in edge["to"], (
                f"dynamic call should not be resolved, got {edge}"
            )


class TestStaticCallEdges:
    def test_imported_static_call_resolves(self):
        result = _parse(FIXTURE_ROOT / "static_calls")
        edge = _find_edge(
            result,
            "App\\Caller::run",
            "App\\Service\\Helper::staticCall",
        )
        assert edge is not None
        assert edge["confidence"] == 1.0

    def test_self_static_call_resolves_to_current_class(self):
        result = _parse(FIXTURE_ROOT / "static_calls")
        edge = _find_edge(
            result,
            "App\\Caller::run",
            "App\\Caller::helper",
        )
        assert edge is not None
        assert edge["confidence"] == 1.0

    def test_fully_qualified_static_call_resolves(self):
        result = _parse(FIXTURE_ROOT / "static_calls")
        edge = _find_edge(
            result,
            "App\\Caller::run",
            "App\\Service\\Helper::other",
        )
        assert edge is not None


class TestTypedLocalEdges:
    def test_method_param_type_resolves_local_call(self):
        result = _parse(FIXTURE_ROOT / "typed_locals")
        edge = _find_edge(
            result,
            "App\\Caller::takesParam",
            "App\\Service\\Helper::doIt",
        )
        assert edge is not None
        assert edge["confidence"] == 0.7

    def test_new_expression_resolves_local_call(self):
        result = _parse(FIXTURE_ROOT / "typed_locals")
        edge = _find_edge(
            result,
            "App\\Caller::newExpr",
            "App\\Service\\Helper::doIt",
        )
        assert edge is not None
        assert edge["confidence"] == 0.7

    def test_untyped_local_skipped(self):
        result = _parse(FIXTURE_ROOT / "typed_locals")
        # untyped() body has $x = something(); $x->doIt(); — must NOT emit an edge
        for edge in result["edges"]:
            assert edge["from"] != "App\\Caller::untyped", (
                f"untyped local should not produce edge: {edge}"
            )


class TestDoctrineRepositoryEdges:
    def test_chained_get_repository_resolves_to_repo_class(self):
        result = _parse(FIXTURE_ROOT / "doctrine_repo")
        edge = _find_edge(
            result,
            "App\\Service\\ExerciseService::chained",
            "App\\Repository\\ExerciseRepository::findActive",
        )
        assert edge is not None
        assert edge["confidence"] == 1.0

    def test_local_var_get_repository_resolves(self):
        result = _parse(FIXTURE_ROOT / "doctrine_repo")
        edge = _find_edge(
            result,
            "App\\Service\\ExerciseService::viaLocal",
            "App\\Repository\\ExerciseRepository::findOneByName",
        )
        assert edge is not None
        assert edge["confidence"] == 1.0


class TestInheritance:
    def test_classes_map_records_extends(self):
        result = _parse(FIXTURE_ROOT / "inheritance")
        classes = result.get("classes", {})
        assert "App\\Child" in classes
        assert classes["App\\Child"]["extends"] == "App\\Base"

    def test_parent_static_call_resolves_to_base_class(self):
        result = _parse(FIXTURE_ROOT / "inheritance")
        edge = _find_edge(
            result,
            "App\\Child::callsParent",
            "App\\Base::inherited",
        )
        assert edge is not None
        assert edge["confidence"] == 1.0

    def test_this_method_call_resolves_to_inherited_method(self):
        result = _parse(FIXTURE_ROOT / "inheritance")
        edge = _find_edge(
            result,
            "App\\Child::callsThis",
            "App\\Base::inherited",
        )
        assert edge is not None


class TestRenderEdges:
    def test_render_emits_template_edge(self):
        result = _parse(FIXTURE_ROOT / "render")
        edge = _find_edge(
            result,
            "App\\Controller\\PageController::index",
            "template:page/index.html.twig",
        )
        assert edge is not None
        assert edge["kind"] == "render"
        assert edge["confidence"] == 1.0

    def test_render_view_emits_template_edge(self):
        result = _parse(FIXTURE_ROOT / "render")
        edge = _find_edge(
            result,
            "App\\Controller\\PageController::api",
            "template:page/api.html.twig",
        )
        assert edge is not None
        assert edge["kind"] == "render"


class TestCacheLayer:
    def test_load_or_parse_writes_cache(self, tmp_path):
        # Copy fixture to tmp so we don't pollute the real fixture dir.
        proj = tmp_path / "proj"
        (proj / "src").mkdir(parents=True)
        (proj / "src" / "Foo.php").write_text(
            "<?php\nnamespace App;\nclass Foo { public function bar(): void {} }\n"
        )
        cache = proj / "cache" / "call-graph.json"
        result = call_graph.load_or_parse(proj, cache_file=cache)
        assert "App\\Foo::bar" in result["symbols"]
        assert cache.exists()

    def test_returns_cached_on_second_call_when_unchanged(self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"
        (proj / "src").mkdir(parents=True)
        (proj / "src" / "Foo.php").write_text(
            "<?php\nnamespace App;\nclass Foo { public function bar(): void {} }\n"
        )
        cache = proj / "cache" / "call-graph.json"
        call_graph.load_or_parse(proj, cache_file=cache)

        # Second call must NOT re-parse — sentinel by monkey-patching parse.
        called = []
        original_parse = call_graph.parse
        monkeypatch.setattr(call_graph, "parse", lambda *a, **kw: (called.append(1) or original_parse(*a, **kw)))
        result = call_graph.load_or_parse(proj, cache_file=cache)
        assert called == [], "parse() should not be invoked when cache is valid"
        assert "App\\Foo::bar" in result["symbols"]

    def test_invalidates_when_file_changes(self, tmp_path):
        proj = tmp_path / "proj"
        (proj / "src").mkdir(parents=True)
        foo = proj / "src" / "Foo.php"
        foo.write_text("<?php\nnamespace App;\nclass Foo { public function old(): void {} }\n")
        cache = proj / "cache" / "call-graph.json"
        call_graph.load_or_parse(proj, cache_file=cache)

        import os
        import time
        time.sleep(0.05)  # ensure mtime changes on filesystems with low resolution
        foo.write_text("<?php\nnamespace App;\nclass Foo { public function fresh(): void {} }\n")
        os.utime(foo, None)

        result = call_graph.load_or_parse(proj, cache_file=cache)
        assert "App\\Foo::fresh" in result["symbols"]
        assert "App\\Foo::old" not in result["symbols"]


class TestTrace:
    @staticmethod
    def _build_graph():
        """Tiny synthetic graph for testing the traversal directly."""
        return {
            "symbols": {
                "A::root":  {"file": "a.php", "line": 1},
                "A::leaf":  {"file": "a.php", "line": 5},
                "B::mid":   {"file": "b.php", "line": 1},
                "C::deep":  {"file": "c.php", "line": 1},
                "C::cycle": {"file": "c.php", "line": 5},
            },
            "edges": [
                {"from": "A::root", "to": "B::mid",   "kind": "call",   "confidence": 1.0, "evidence": "a.php:2"},
                {"from": "A::root", "to": "A::leaf",  "kind": "call",   "confidence": 1.0, "evidence": "a.php:3"},
                {"from": "B::mid",  "to": "C::deep",  "kind": "call",   "confidence": 0.7, "evidence": "b.php:2"},
                {"from": "B::mid",  "to": "tpl:x",    "kind": "render", "confidence": 1.0, "evidence": "b.php:3"},
                {"from": "C::deep", "to": "C::cycle", "kind": "call",   "confidence": 1.0, "evidence": "c.php:2"},
                {"from": "C::cycle","to": "C::deep",  "kind": "call",   "confidence": 1.0, "evidence": "c.php:6"},
            ],
            "classes": {},
            "stats": {},
        }

    def test_trace_returns_tree_for_root(self):
        graph = self._build_graph()
        tree = call_graph.trace(graph, "A::root", max_depth=6)
        assert tree["symbol"] == "A::root"
        children = tree["children"]
        # children preserve emission order from the edges list
        assert [c["symbol"] for c in children] == ["B::mid", "A::leaf"]

    def test_trace_includes_render_edges(self):
        graph = self._build_graph()
        tree = call_graph.trace(graph, "A::root", max_depth=6)
        b_mid = next(c for c in tree["children"] if c["symbol"] == "B::mid")
        kinds = [c["kind"] for c in b_mid["children"]]
        assert "render" in kinds

    def test_trace_carries_edge_metadata_to_child(self):
        graph = self._build_graph()
        tree = call_graph.trace(graph, "A::root", max_depth=6)
        b_mid = next(c for c in tree["children"] if c["symbol"] == "B::mid")
        assert b_mid["confidence"] == 1.0
        assert b_mid["evidence"] == "a.php:2"
        c_deep = b_mid["children"][0]  # the call edge, not the render
        assert c_deep["confidence"] == 0.7

    def test_trace_respects_max_depth(self):
        graph = self._build_graph()
        tree = call_graph.trace(graph, "A::root", max_depth=1)
        # depth 1 = root + immediate children, no grandchildren
        for child in tree["children"]:
            assert child["children"] == []

    def test_trace_breaks_cycles(self):
        graph = self._build_graph()
        tree = call_graph.trace(graph, "C::deep", max_depth=10)
        # C::deep -> C::cycle -> (C::deep loop) — the back-edge node is
        # emitted but not recursed into.
        c_cycle = tree["children"][0]
        assert c_cycle["symbol"] == "C::cycle"
        loopback = c_cycle["children"][0]
        assert loopback["symbol"] == "C::deep"
        assert loopback.get("truncated") == "cycle"
        assert loopback["children"] == []

    def test_trace_unknown_symbol_returns_empty_tree(self):
        graph = self._build_graph()
        tree = call_graph.trace(graph, "Nonexistent::method", max_depth=6)
        assert tree["symbol"] == "Nonexistent::method"
        assert tree["children"] == []
        assert tree.get("missing") is True


class TestFindChangedSymbols:
    @staticmethod
    def _graph():
        return {
            "symbols": {
                "App\\Foo::one":   {"file": "src/Foo.php", "line": 5,  "end_line": 10},
                "App\\Foo::two":   {"file": "src/Foo.php", "line": 12, "end_line": 20},
                "App\\Foo::three": {"file": "src/Foo.php", "line": 22, "end_line": 30},
                "App\\Bar::x":     {"file": "src/Bar.php", "line": 5,  "end_line": 8},
            },
            "edges": [],
            "classes": {},
            "stats": {},
        }

    def test_finds_single_method_in_range(self):
        graph = self._graph()
        changed = call_graph.find_changed_symbols(graph, {"src/Foo.php": [(7, 9)]})
        assert changed == {"App\\Foo::one"}

    def test_overlap_at_boundary_counts(self):
        graph = self._graph()
        # range starts exactly at method's last line — should still count
        changed = call_graph.find_changed_symbols(graph, {"src/Foo.php": [(10, 11)]})
        assert "App\\Foo::one" in changed

    def test_multiple_methods_per_file(self):
        graph = self._graph()
        changed = call_graph.find_changed_symbols(
            graph,
            {"src/Foo.php": [(7, 9), (15, 16), (25, 26)]},
        )
        assert changed == {"App\\Foo::one", "App\\Foo::two", "App\\Foo::three"}

    def test_range_outside_any_method_returns_empty(self):
        graph = self._graph()
        # gap between two() and three() is lines 21
        changed = call_graph.find_changed_symbols(graph, {"src/Foo.php": [(21, 21)]})
        assert changed == set()

    def test_unknown_file_skipped(self):
        graph = self._graph()
        changed = call_graph.find_changed_symbols(graph, {"src/Unknown.php": [(1, 100)]})
        assert changed == set()


class TestReverseCallers:
    @staticmethod
    def _graph():
        # A -> B, A -> C, B -> D, C -> D, X -> A
        return {
            "symbols": {
                "A::a": {"file": "a.php", "line": 1, "end_line": 5},
                "B::b": {"file": "b.php", "line": 1, "end_line": 5},
                "C::c": {"file": "c.php", "line": 1, "end_line": 5},
                "D::d": {"file": "d.php", "line": 1, "end_line": 5},
                "X::x": {"file": "x.php", "line": 1, "end_line": 5},
            },
            "edges": [
                {"from": "A::a", "to": "B::b", "kind": "call", "confidence": 1.0, "evidence": "a.php:2"},
                {"from": "A::a", "to": "C::c", "kind": "call", "confidence": 1.0, "evidence": "a.php:3"},
                {"from": "B::b", "to": "D::d", "kind": "call", "confidence": 1.0, "evidence": "b.php:2"},
                {"from": "C::c", "to": "D::d", "kind": "call", "confidence": 1.0, "evidence": "c.php:2"},
                {"from": "X::x", "to": "A::a", "kind": "call", "confidence": 1.0, "evidence": "x.php:2"},
            ],
            "classes": {},
            "stats": {},
        }

    def test_direct_callers_returned(self):
        graph = self._graph()
        callers = call_graph.reverse_callers(graph, "D::d", max_depth=1)
        # Direct callers of D are B and C
        assert {c["symbol"] for c in callers} == {"B::b", "C::c"}

    def test_transitive_callers_within_depth(self):
        graph = self._graph()
        callers = call_graph.reverse_callers(graph, "D::d", max_depth=3)
        # Transitively: B, C (depth 1), A (depth 2 via B and C), X (depth 3)
        assert {c["symbol"] for c in callers} == {"B::b", "C::c", "A::a", "X::x"}

    def test_each_caller_includes_min_depth(self):
        graph = self._graph()
        callers = call_graph.reverse_callers(graph, "D::d", max_depth=10)
        depths = {c["symbol"]: c["depth"] for c in callers}
        assert depths["B::b"] == 1
        assert depths["C::c"] == 1
        assert depths["A::a"] == 2
        assert depths["X::x"] == 3

    def test_unknown_target_returns_empty(self):
        graph = self._graph()
        assert call_graph.reverse_callers(graph, "Nope::n", max_depth=5) == []


class TestDiffHunkParser:
    def test_parses_single_file_single_hunk(self):
        diff = (
            "diff --git a/src/Foo.php b/src/Foo.php\n"
            "--- a/src/Foo.php\n"
            "+++ b/src/Foo.php\n"
            "@@ -10,0 +11,3 @@\n"
            "+a\n+b\n+c\n"
        )
        ranges = call_graph.parse_diff_hunks(diff)
        assert ranges == {"src/Foo.php": [(11, 13)]}

    def test_parses_multiple_files(self):
        diff = (
            "diff --git a/src/A.php b/src/A.php\n"
            "--- a/src/A.php\n"
            "+++ b/src/A.php\n"
            "@@ -5,1 +5,2 @@\n"
            "+x\n"
            "diff --git a/src/B.php b/src/B.php\n"
            "--- a/src/B.php\n"
            "+++ b/src/B.php\n"
            "@@ -20,0 +21,1 @@\n"
            "+y\n"
        )
        ranges = call_graph.parse_diff_hunks(diff)
        assert ranges == {"src/A.php": [(5, 6)], "src/B.php": [(21, 21)]}

    def test_parses_multiple_hunks_per_file(self):
        diff = (
            "diff --git a/src/A.php b/src/A.php\n"
            "--- a/src/A.php\n"
            "+++ b/src/A.php\n"
            "@@ -10,0 +11,2 @@\n"
            "+x\n+y\n"
            "@@ -50,0 +52,1 @@\n"
            "+z\n"
        )
        ranges = call_graph.parse_diff_hunks(diff)
        assert ranges == {"src/A.php": [(11, 12), (52, 52)]}

    def test_handles_pure_deletion_hunk(self):
        """When a hunk only deletes lines (new_count=0), record the surviving line."""
        diff = (
            "diff --git a/src/A.php b/src/A.php\n"
            "--- a/src/A.php\n"
            "+++ b/src/A.php\n"
            "@@ -10,3 +9,0 @@\n"
            "-x\n-y\n-z\n"
        )
        ranges = call_graph.parse_diff_hunks(diff)
        # We mark lines 9 (the surviving anchor) so a method straddling
        # the deletion still gets flagged.
        assert ranges == {"src/A.php": [(9, 9)]}

    def test_only_php_files_returned(self):
        diff = (
            "diff --git a/src/A.php b/src/A.php\n"
            "--- a/src/A.php\n"
            "+++ b/src/A.php\n"
            "@@ -1,0 +2,1 @@\n"
            "+x\n"
            "diff --git a/README.md b/README.md\n"
            "--- a/README.md\n"
            "+++ b/README.md\n"
            "@@ -1,0 +2,1 @@\n"
            "+y\n"
        )
        ranges = call_graph.parse_diff_hunks(diff)
        assert "src/A.php" in ranges
        assert "README.md" not in ranges


class TestStimulusJsParsing:
    def test_method_symbols_use_js_prefix(self):
        result = _parse(FIXTURE_ROOT / "stimulus_basic")
        assert "js:basic::startSession" in result["symbols"]
        assert "js:basic::endSession" in result["symbols"]

    def test_symbol_records_file_and_line(self):
        result = _parse(FIXTURE_ROOT / "stimulus_basic")
        sym = result["symbols"]["js:basic::startSession"]
        assert sym["kind"] == "stimulus_method"
        assert sym["file"].endswith("basic_controller.js")
        assert sym["line"] >= 1
        assert sym["end_line"] >= sym["line"]

    def test_literal_fetch_emits_fetch_edge(self):
        result = _parse(FIXTURE_ROOT / "stimulus_basic")
        edge = _find_edge(
            result,
            "js:basic::startSession",
            "fetch:POST /api/session/start",
        )
        assert edge is not None
        assert edge["kind"] == "fetch"
        assert edge["confidence"] == 1.0

    def test_template_literal_fetch_emits_wildcard_edge(self):
        result = _parse(FIXTURE_ROOT / "stimulus_basic")
        edge = _find_edge(
            result,
            "js:basic::endSession",
            "fetch:GET /api/session/*/status",
        )
        assert edge is not None
        assert edge["confidence"] == 0.7

    def test_unresolvable_fetch_skipped(self):
        result = _parse(FIXTURE_ROOT / "stimulus_basic")
        # `fetch(this.urlValue)` cannot be statically resolved — must NOT emit.
        for edge in result["edges"]:
            if edge["from"] == "js:basic::dynamic":
                assert "fetch" not in edge["kind"], f"unexpected dynamic fetch edge: {edge}"

    def test_method_defaults_to_get_when_no_options(self):
        result = _parse(FIXTURE_ROOT / "stimulus_basic")
        edge = _find_edge(
            result,
            "js:basic::endSession",
            "fetch:GET /api/session/*/status",
        )
        # GET is the default fetch method; the edge target must include GET
        assert edge is not None


class TestResolveFetchEdges:
    def test_resolves_literal_fetch_to_php_symbol(self):
        graph = {
            "symbols": {"js:foo::run": {"file": "assets/controllers/foo_controller.js", "line": 1, "end_line": 5, "kind": "stimulus_method"}},
            "edges": [
                {"from": "js:foo::run", "to": "fetch:POST /api/session/start",
                 "kind": "fetch", "confidence": 1.0, "evidence": "assets/foo:3"},
            ],
            "classes": {},
            "stats": {},
        }
        routes = {"routes": {
            "/api/session/start": {
                "methods": ["POST"],
                "controller": "App\\Controller\\Api\\SessionApiController",
                "action": "start",
                "file": "src/Controller/Api/SessionApiController.php",
                "template": None,
                "services": [],
                "name": "api_session_start",
            },
        }}
        call_graph.resolve_fetch_edges(graph, routes)
        edge = graph["edges"][0]
        assert edge["to"] == "App\\Controller\\Api\\SessionApiController::start"

    def test_resolves_wildcard_fetch_to_php_symbol(self):
        """JS template literal `/api/x/${id}/y` -> route `/api/x/{id}/y`."""
        graph = {
            "symbols": {"js:foo::run": {"file": "assets/controllers/foo_controller.js", "line": 1, "end_line": 5, "kind": "stimulus_method"}},
            "edges": [
                {"from": "js:foo::run", "to": "fetch:GET /api/session/*/status",
                 "kind": "fetch", "confidence": 0.7, "evidence": "x"},
            ],
            "classes": {},
            "stats": {},
        }
        routes = {"routes": {
            "/api/session/{id}/status": {
                "methods": ["GET"],
                "controller": "App\\Controller\\Api\\SessionApiController",
                "action": "status",
                "file": "src/Controller/Api/SessionApiController.php",
                "template": None,
                "services": [],
                "name": "api_session_status",
            },
        }}
        call_graph.resolve_fetch_edges(graph, routes)
        edge = graph["edges"][0]
        assert edge["to"] == "App\\Controller\\Api\\SessionApiController::status"

    def test_unresolved_fetch_keeps_placeholder(self):
        """If no route matches, the edge stays as-is so users see what was attempted."""
        graph = {
            "symbols": {},
            "edges": [
                {"from": "js:x::y", "to": "fetch:GET /api/nope",
                 "kind": "fetch", "confidence": 1.0, "evidence": "x"},
            ],
            "classes": {},
            "stats": {},
        }
        call_graph.resolve_fetch_edges(graph, {"routes": {}})
        assert graph["edges"][0]["to"] == "fetch:GET /api/nope"


class TestRealCodebase:
    """Smoke checks against the live Symfony app — assert invariants, not exact values."""

    def test_parses_without_error(self):
        result = call_graph.parse(PROJECT_ROOT)
        assert result["stats"]["total_files"] > 100, (
            f"expected real Symfony src/ to have >100 PHP files, got {result['stats']['total_files']}"
        )

    def test_finds_controllers_and_services(self):
        result = call_graph.parse(PROJECT_ROOT)
        controller_classes = [c for c in result["classes"] if "\\Controller\\" in c]
        service_classes = [c for c in result["classes"] if "\\Service\\" in c]
        assert len(controller_classes) >= 20
        assert len(service_classes) >= 20

    def test_emits_call_edges(self):
        result = call_graph.parse(PROJECT_ROOT)
        # Real codebase should have hundreds of resolved edges across all files
        assert len(result["edges"]) >= 100
        # Confidence breakdown should include both 1.0 and 0.7 hits
        by_conf = result["stats"]["by_confidence"]
        assert by_conf.get("1.0", 0) > 0

    def test_render_edges_present(self):
        result = call_graph.parse(PROJECT_ROOT)
        render_edges = [e for e in result["edges"] if e["kind"] == "render"]
        assert len(render_edges) >= 5, f"expected >=5 render edges, got {len(render_edges)}"


def _find_edge(result: dict, from_id: str, to_id: str) -> dict | None:
    for edge in result["edges"]:
        if edge["from"] == from_id and edge["to"] == to_id:
            return edge
    return None
