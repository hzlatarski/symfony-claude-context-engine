"""End-to-end integration test.

Runs every parser plus the MCP server tool functions against the real
Symfony codebase and asserts on the success criteria from the design doc.
"""
import time

from scripts import mcp_server
from scripts.parsers import PROJECT_ROOT, git_intel


def test_success_criterion_1_file_deps_under_1s():
    """get_file_deps must return in under 1 second (with all caches warm).

    _build_file_deps for a PHP entity reads PHP graph + route map + git intel,
    so all three caches must be warm before timing to avoid measuring one-off
    subprocess/disk costs.
    """
    mcp_server._cache.get_php_graph()
    mcp_server._cache.get_route_map()
    mcp_server._cache.get_git_intel()
    start = time.time()
    result = mcp_server._build_file_deps("src/Entity/User.php")
    elapsed = time.time() - start
    assert elapsed < 1.0, f"Expected <1s, got {elapsed:.2f}s"
    assert "User" in result


def test_success_criterion_2_api_route_map():
    """get_route_map('/api') returns API routes with controllers and services."""
    result = mcp_server._build_route_map("/api")
    assert "/api" in result
    assert "Controller" in result or "controller" in result


def test_success_criterion_3_arena_stimulus_map():
    """get_stimulus_map('arena') returns templates using the arena controller."""
    result = mcp_server._build_stimulus_map("arena")
    assert "arena" in result.lower()
    assert ".twig" in result


def test_success_criterion_5_hotspots_ranked():
    """get_hotspots(5) returns 5 most-churned files with ownership."""
    git_intel.load_or_parse(PROJECT_ROOT)  # ensure cache
    result = mcp_server._build_hotspots(5)
    assert "Hotspots" in result
    assert "owner" in result.lower() or "commits" in result.lower()


def test_success_criterion_6_codebase_overview_compact():
    """get_codebase_overview returns all stats in one response."""
    result = mcp_server._build_codebase_overview()
    assert "PHP" in result
    assert "Routes" in result
    assert "Templates" in result
    assert "Stimulus" in result
    assert "Hotspots" in result


def test_parser_cache_invalidates_on_mtime_change():
    """Touching a .php file should force a rebuild on the next call.

    ParseCache uses max(mtime) across all .php files, so we need to bump
    User.php ABOVE the current max — not just above its own current mtime.
    """
    import os

    cache = mcp_server.ParseCache()
    first = cache.get_php_graph()
    first_id = id(first)

    target = PROJECT_ROOT / "src" / "Entity" / "User.php"
    original_mtime = target.stat().st_mtime

    # Find the current max mtime across all .php files in src/ so we can
    # bump User.php above it and guarantee cache invalidation.
    current_max = max(
        p.stat().st_mtime for p in (PROJECT_ROOT / "src").rglob("*.php")
    )
    try:
        new_mtime = current_max + 10.0
        os.utime(target, (new_mtime, new_mtime))
        second = cache.get_php_graph()
        assert id(second) != first_id, "Cache should have been rebuilt after mtime change"
    finally:
        os.utime(target, (original_mtime, original_mtime))
