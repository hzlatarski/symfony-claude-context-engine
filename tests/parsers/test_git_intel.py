"""Tests for git intelligence parser."""
import json
from pathlib import Path

from scripts.parsers import PROJECT_ROOT, git_intel


def test_parse_returns_expected_shape():
    result = git_intel.parse(PROJECT_ROOT)
    assert "head" in result
    assert "generated_at" in result
    assert "hotspots" in result
    assert "decisions" in result
    assert "stats" in result


def test_head_matches_git_rev_parse(tmp_path):
    """The cached HEAD should match git rev-parse HEAD at the time of parse."""
    import subprocess
    result = git_intel.parse(PROJECT_ROOT)
    current_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
    ).strip()
    assert result["head"] == current_head[:12], f"Expected HEAD {current_head[:12]}, got {result['head']}"


def test_hotspots_are_ranked_by_score():
    result = git_intel.parse(PROJECT_ROOT)
    scores = [h["score"] for h in result["hotspots"]]
    assert scores == sorted(scores, reverse=True), "Hotspots must be ranked by score descending"


def test_hotspot_has_required_fields():
    result = git_intel.parse(PROJECT_ROOT)
    if not result["hotspots"]:
        return  # Fresh repo — acceptable
    h = result["hotspots"][0]
    assert "file" in h
    assert "score" in h
    assert "commits_total" in h
    assert "primary_owner" in h
    assert "co_change_partners" in h


def test_decisions_have_commit_classification():
    result = git_intel.parse(PROJECT_ROOT)
    for d in result["decisions"]:
        assert d["type"] in {"refactor", "migration", "extraction", "replacement", "introduction", "removal", "other"}
        assert "commit" in d
        assert "message" in d


def test_cache_roundtrip(tmp_path):
    """load_or_parse should use cache when HEAD is unchanged."""
    cache_file = tmp_path / "git-intel.json"
    first = git_intel.load_or_parse(PROJECT_ROOT, cache_file=cache_file)
    assert cache_file.exists(), "Cache file should be written"
    second = git_intel.load_or_parse(PROJECT_ROOT, cache_file=cache_file)
    # Same HEAD → same generated_at (cache hit)
    assert first["generated_at"] == second["generated_at"]


def test_summary_returns_short_string():
    s = git_intel.summary(PROJECT_ROOT)
    assert isinstance(s, str)
    assert len(s) < 500
