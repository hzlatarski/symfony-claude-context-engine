"""Tests for merge-order conflict risk (Feature #3), pure core only."""
from __future__ import annotations

from scripts import merge_risk


def _clusters():
    """Two communities: billing (Foo/Bar) and email (Mail)."""
    return [
        {"community_id": 0, "label": "billing", "members": [
            "file:src/Billing/Foo.php", "file:src/Billing/Bar.php",
        ]},
        {"community_id": 1, "label": "email", "members": [
            "file:src/Email/Mail.php",
        ]},
    ]


def test_direct_conflict_when_same_file_on_two_branches():
    node_comm = merge_risk.build_node_community_map(_clusters())
    result = merge_risk.compute(
        {
            "feat/a": ["src/Billing/Foo.php"],
            "feat/b": ["src/Billing/Foo.php"],
        },
        node_comm,
    )
    assert len(result["direct_conflicts"]) == 1
    conflict = result["direct_conflicts"][0]
    assert conflict["file"] == "src/Billing/Foo.php"
    assert conflict["branches"] == ["feat/a", "feat/b"]


def test_community_overlap_when_different_files_same_cluster():
    node_comm = merge_risk.build_node_community_map(_clusters())
    result = merge_risk.compute(
        {
            "feat/a": ["src/Billing/Foo.php"],
            "feat/b": ["src/Billing/Bar.php"],
        },
        node_comm,
    )
    assert not result["direct_conflicts"]
    assert len(result["community_overlaps"]) == 1
    ov = result["community_overlaps"][0]
    assert {ov["branch_a"], ov["branch_b"]} == {"feat/a", "feat/b"}
    assert ov["shared"][0]["label"] == "billing"


def test_no_risk_when_branches_touch_different_communities():
    node_comm = merge_risk.build_node_community_map(_clusters())
    result = merge_risk.compute(
        {
            "feat/a": ["src/Billing/Foo.php"],
            "feat/b": ["src/Email/Mail.php"],
        },
        node_comm,
    )
    assert not result["direct_conflicts"]
    assert not result["community_overlaps"]


def test_file_outside_any_community_still_counts_for_direct_conflict():
    node_comm = merge_risk.build_node_community_map(_clusters())
    result = merge_risk.compute(
        {
            "feat/a": ["src/Untracked/New.php"],
            "feat/b": ["src/Untracked/New.php"],
        },
        node_comm,
    )
    assert len(result["direct_conflicts"]) == 1
    assert not result["community_overlaps"]


def test_branch_summary_counts_files_and_communities():
    node_comm = merge_risk.build_node_community_map(_clusters())
    result = merge_risk.compute(
        {"feat/a": ["src/Billing/Foo.php", "src/Billing/Bar.php", "src/Untracked/x.php"]},
        node_comm,
    )
    summ = next(b for b in result["branches"] if b["branch"] == "feat/a")
    assert summ["file_count"] == 3
    assert summ["community_count"] == 1  # both billing files → one community


def test_stacked_branch_does_not_report_false_direct_conflict():
    """M3: B stacked on A restates A's files — not a conflict when merged in order."""
    node_comm = merge_risk.build_node_community_map(_clusters())
    result = merge_risk.compute(
        {
            "feat/a": ["src/Billing/Foo.php"],
            "feat/b": ["src/Billing/Foo.php", "src/Billing/Bar.php"],
        },
        node_comm,
        ancestry={("feat/a", "feat/b")},  # a is an ancestor of b
    )
    assert not result["direct_conflicts"]
    assert not result["community_overlaps"]


def test_third_independent_branch_still_conflicts_despite_a_stack():
    node_comm = merge_risk.build_node_community_map(_clusters())
    result = merge_risk.compute(
        {
            "feat/a": ["src/Billing/Foo.php"],
            "feat/b": ["src/Billing/Foo.php"],   # stacked on a
            "feat/c": ["src/Billing/Foo.php"],   # independent
        },
        node_comm,
        ancestry={("feat/a", "feat/b")},
    )
    # Foo.php is touched by tips {feat/b, feat/c} (feat/a is dominated by b).
    assert len(result["direct_conflicts"]) == 1
    assert result["direct_conflicts"][0]["branches"] == ["feat/b", "feat/c"]


def test_render_contains_sections():
    node_comm = merge_risk.build_node_community_map(_clusters())
    result = merge_risk.compute(
        {"feat/a": ["src/Billing/Foo.php"], "feat/b": ["src/Billing/Bar.php"]},
        node_comm,
    )
    out = merge_risk.render(result, base="main")
    assert "Merge-order conflict risk" in out
    assert "Community-overlap risk" in out
    assert "billing" in out


def test_render_empty_branches():
    out = merge_risk.render(
        {"branches": [], "direct_conflicts": [], "community_overlaps": []}, base="main"
    )
    assert "No branches found" in out
