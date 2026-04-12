"""Tests for Stimulus controller ↔ template map parser."""
from scripts.parsers import PROJECT_ROOT, stimulus_map


def test_parse_returns_expected_shape():
    result = stimulus_map.parse(PROJECT_ROOT)
    assert "controllers" in result
    assert "orphan_controllers" in result
    assert "missing_controllers" in result
    assert "stats" in result


def test_finds_arena_controller():
    """The arena_controller.js file must be discovered and identified as 'arena'."""
    result = stimulus_map.parse(PROJECT_ROOT)
    assert "arena" in result["controllers"], f"Expected 'arena' in controllers, got {list(result['controllers'].keys())[:10]}"
    arena = result["controllers"]["arena"]
    assert arena["file"].endswith("arena_controller.js")


def test_arena_controller_used_in_templates():
    """The arena Stimulus controller should be referenced by at least one template."""
    result = stimulus_map.parse(PROJECT_ROOT)
    arena = result["controllers"].get("arena", {})
    used_in = arena.get("used_in", [])
    assert len(used_in) >= 1, f"Expected arena controller used in >=1 template, got {used_in}"


def test_loading_btn_controller_is_widely_used():
    """loading-btn is a shared controller used in many admin templates."""
    result = stimulus_map.parse(PROJECT_ROOT)
    loading_btn = result["controllers"].get("loading-btn", {})
    used_in = loading_btn.get("used_in", [])
    assert len(used_in) >= 3, f"Expected loading-btn used in >=3 templates, got {len(used_in)}"


def test_summary_returns_short_string():
    s = stimulus_map.summary(PROJECT_ROOT)
    assert isinstance(s, str)
    assert len(s) < 500
