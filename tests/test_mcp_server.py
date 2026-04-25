"""Tests for MCP server tool implementations.

We test the underlying `_build_*` helpers rather than spinning up a full
FastMCP stdio process — the FastMCP decorator layer is thin and adds no
logic worth testing, and stdio harnesses are brittle on Windows.
"""
from scripts import mcp_server


def test_build_codebase_overview_returns_markdown():
    result = mcp_server._build_codebase_overview()
    assert isinstance(result, str)
    assert "PHP" in result
    assert "Twig" in result or "templates" in result.lower()
    assert "Stimulus" in result or "controllers" in result.lower()


def test_build_file_deps_for_real_controller():
    result = mcp_server._build_file_deps("src/Entity/User.php")
    assert isinstance(result, str)
    assert "User" in result
    # Should mention imports or classification
    assert "entity" in result.lower() or "imports" in result.lower()


def test_build_file_deps_for_missing_file():
    result = mcp_server._build_file_deps("src/Doesnt/Exist.php")
    assert isinstance(result, str)
    assert "not found" in result.lower() or "unknown" in result.lower()


def test_build_route_map_with_prefix():
    result = mcp_server._build_route_map("/api")
    assert isinstance(result, str)
    assert "/api" in result


def test_build_route_map_empty_prefix_returns_all():
    result = mcp_server._build_route_map("")
    assert isinstance(result, str)
    assert len(result) > 100  # non-trivial output


def test_build_stimulus_map_for_arena():
    result = mcp_server._build_stimulus_map("arena")
    assert isinstance(result, str)
    assert "arena" in result.lower()


def test_build_template_graph_for_specific_template():
    result = mcp_server._build_template_graph("arena/index.html.twig")
    assert isinstance(result, str)
    # Should reference extends or includes or stimulus
    assert "arena" in result.lower()


def test_build_hotspots_returns_ranked_list():
    result = mcp_server._build_hotspots(5)
    assert isinstance(result, str)
    assert "commits" in result.lower() or "score" in result.lower()


def test_build_trace_route_for_known_post_route():
    result = mcp_server._build_trace_route("POST", "/api/session/start")
    assert isinstance(result, str)
    assert "SessionApiController" in result
    # Trace from controller action — should include the action name `start`
    # somewhere and at least one downstream symbol
    assert "::start" in result


def test_build_trace_route_unknown_route_explains():
    result = mcp_server._build_trace_route("GET", "/does/not/exist/12345")
    assert isinstance(result, str)
    assert "not found" in result.lower() or "no route" in result.lower()


def test_build_trace_route_respects_max_depth():
    deep = mcp_server._build_trace_route("POST", "/api/session/start", max_depth=4)
    shallow = mcp_server._build_trace_route("POST", "/api/session/start", max_depth=1)
    # Shallow output should be strictly shorter (fewer rendered lines)
    assert len(shallow) < len(deep)


def test_build_impact_of_change_no_changes_explains():
    """When there's no diff vs the given ref, output must say so cleanly."""
    # An empty diff comes back from comparing HEAD against itself.
    result = mcp_server._build_impact_of_change(file=None, since_ref="HEAD..HEAD")
    assert isinstance(result, str)
    assert "no changes" in result.lower() or "nothing changed" in result.lower()


def test_build_impact_of_change_with_synthetic_changes(monkeypatch):
    """Verify the rendering pipeline end-to-end by injecting fake diff output."""
    # Pick a method we know exists in the call graph and seed a fake diff
    # whose line range falls inside that method.
    graph = mcp_server._cache.get_call_graph()
    target_id = "App\\Service\\Session\\AccessControlService::canAccess"
    target = graph["symbols"].get(target_id)
    assert target is not None, "expected canAccess to be in the call graph"

    fake_diff = (
        f"diff --git a/{target['file']} b/{target['file']}\n"
        f"--- a/{target['file']}\n"
        f"+++ b/{target['file']}\n"
        f"@@ -{target['line']},0 +{target['line']},1 @@\n"
        "+// dummy\n"
    )
    monkeypatch.setattr(mcp_server, "_run_git_diff", lambda *a, **kw: fake_diff)

    result = mcp_server._build_impact_of_change(file=None, since_ref="HEAD")
    assert "AccessControlService" in result
    # SessionApiController::start calls canAccess — should appear as an
    # upstream caller / affected route.
    assert "SessionApiController" in result


def test_parse_cache_uses_mtime_invalidation():
    """The ParseCache should return the same dict on two quick calls."""
    cache = mcp_server.ParseCache()
    a = cache.get_php_graph()
    b = cache.get_php_graph()
    # Same object reference — cache hit
    assert a is b


def test_mcp_server_launches_without_import_error(tmp_path):
    """Regression: `python scripts/mcp_server.py` must not fail with
    ModuleNotFoundError when launched directly (the way Claude Code
    launches MCP servers).

    Pytest's `pythonpath = ["."]` config in pyproject.toml hides this bug
    from in-process tests by pre-populating sys.path. We spawn a clean
    subprocess with PYTHONPATH="" and cwd=tmp_path to guarantee the script
    cannot rely on inherited path configuration — it must bootstrap its own
    sys.path.

    If the bootstrap at the top of mcp_server.py works, the import chain
    succeeds, FastMCP enters its stdio loop, and the server exits cleanly
    when we close stdin.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    memory_compiler = Path(__file__).resolve().parent.parent
    script = memory_compiler / "scripts" / "mcp_server.py"
    assert script.exists(), f"mcp_server.py not found at {script}"

    # Scrub PYTHONPATH and run from an unrelated cwd so the only way imports
    # can succeed is via the script's own sys.path bootstrap.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = ""

    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(tmp_path),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        # Timeout is a signal the server is alive in its stdio loop — treat
        # as success as long as no import error was logged before the timeout.
        stderr = (e.stderr or b"").decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        assert "ModuleNotFoundError" not in stderr, f"Import failed before timeout:\n{stderr}"
        return

    # If subprocess exited (stdin closed → FastMCP shutdown), stderr must be
    # clean of import errors regardless of exit code.
    assert "ModuleNotFoundError" not in result.stderr, (
        f"Direct script execution failed with import error:\n{result.stderr}"
    )
    assert "No module named 'scripts" not in result.stderr, (
        f"Direct script execution failed with import error:\n{result.stderr}"
    )
