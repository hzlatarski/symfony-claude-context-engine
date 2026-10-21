"""Tests for the maintain pass — focused on the one real judgement call.

``pick_weaker`` decides which twin of a near-duplicate pair is retired to cold;
``plan_retirements`` turns pairs into actions and skips the ones a human must
settle. Both are pure/injectable, so no Chroma or embeddings are needed here.
"""
from types import SimpleNamespace

import maintain


def _mk(knowledge_dir, slug, *, pinned=False, updated="", body="body text"):
    """Write a minimal article file and return its slug."""
    path = knowledge_dir / f"{slug}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = ["---", "title: T"]
    if pinned:
        fm.append("pinned: true")
    if updated:
        fm.append(f"updated: {updated}")
    fm.append("---")
    path.write_text("\n".join(fm) + f"\n\n## Truth\n\n{body}\n", encoding="utf-8")
    return slug


# ── pick_weaker ──────────────────────────────────────────────────────

def test_pinned_twin_is_always_stronger(tmp_path):
    _mk(tmp_path, "concepts/pinned", pinned=True)
    _mk(tmp_path, "concepts/plain", pinned=False)
    weaker, stronger = maintain.pick_weaker(
        "concepts/pinned", "concepts/plain",
        knowledge_dir=tmp_path, feedback_scores={},
    )
    assert weaker == "concepts/plain"
    assert stronger == "concepts/pinned"


def test_both_pinned_leaves_choice_to_human(tmp_path):
    _mk(tmp_path, "concepts/a", pinned=True)
    _mk(tmp_path, "concepts/b", pinned=True)
    assert maintain.pick_weaker(
        "concepts/a", "concepts/b", knowledge_dir=tmp_path, feedback_scores={},
    ) == (None, None)


def test_lower_feedback_score_is_weaker(tmp_path):
    _mk(tmp_path, "concepts/good")
    _mk(tmp_path, "concepts/bad")
    weaker, stronger = maintain.pick_weaker(
        "concepts/good", "concepts/bad",
        knowledge_dir=tmp_path,
        feedback_scores={"concepts/good": 0.8, "concepts/bad": 0.3},
    )
    assert weaker == "concepts/bad"
    assert stronger == "concepts/good"


def test_older_updated_is_weaker_when_scores_tie(tmp_path):
    _mk(tmp_path, "concepts/old", updated="2026-01-01")
    _mk(tmp_path, "concepts/new", updated="2026-09-01")
    weaker, stronger = maintain.pick_weaker(
        "concepts/old", "concepts/new",
        knowledge_dir=tmp_path, feedback_scores={},  # both neutral 0.5
    )
    assert weaker == "concepts/old"


def test_shorter_body_is_weaker_when_dates_tie(tmp_path):
    _mk(tmp_path, "concepts/short", updated="2026-01-01", body="tiny")
    _mk(tmp_path, "concepts/long", updated="2026-01-01", body="much longer body " * 20)
    weaker, stronger = maintain.pick_weaker(
        "concepts/short", "concepts/long",
        knowledge_dir=tmp_path, feedback_scores={},
    )
    assert weaker == "concepts/short"


def test_fully_tied_uses_deterministic_slug_tiebreak(tmp_path):
    _mk(tmp_path, "concepts/aaa", updated="2026-01-01", body="same")
    _mk(tmp_path, "concepts/bbb", updated="2026-01-01", body="same")
    weaker, stronger = maintain.pick_weaker(
        "concepts/aaa", "concepts/bbb",
        knowledge_dir=tmp_path, feedback_scores={},
    )
    # Lexicographically later slug is retired; result is order-independent.
    assert weaker == "concepts/bbb"
    assert stronger == "concepts/aaa"
    assert maintain.pick_weaker(
        "concepts/bbb", "concepts/aaa",
        knowledge_dir=tmp_path, feedback_scores={},
    ) == ("concepts/bbb", "concepts/aaa")


# ── plan_retirements ─────────────────────────────────────────────────

def _pair(a, b, sim):
    return SimpleNamespace(slug_a=a, slug_b=b, similarity=sim)


def test_plan_builds_action_for_a_pair(tmp_path):
    _mk(tmp_path, "concepts/keep")
    _mk(tmp_path, "concepts/drop")
    actions, skipped = maintain.plan_retirements(
        [_pair("concepts/keep", "concepts/drop", 0.96)],
        knowledge_dir=tmp_path,
        feedback_scores={"concepts/keep": 0.9, "concepts/drop": 0.2},
        already_cold=set(),
    )
    assert skipped == []
    assert len(actions) == 1
    assert actions[0].weaker == "concepts/drop"
    assert actions[0].stronger == "concepts/keep"
    assert "near-duplicate of concepts/keep" in actions[0].reason


def test_cluster_never_retires_last_hot_when_strongest_is_cold(tmp_path):
    """If the strongest member is already cold, the last hot copy must survive."""
    _mk(tmp_path, "concepts/strong")
    _mk(tmp_path, "concepts/weak")
    actions, skipped = maintain.plan_retirements(
        [_pair("concepts/strong", "concepts/weak", 0.97)],
        knowledge_dir=tmp_path,
        feedback_scores={"concepts/strong": 0.9, "concepts/weak": 0.2},
        already_cold={"concepts/strong"},   # the strongest is already retired
    )
    # 'weak' is the only hot article → it becomes the survivor, nothing retired.
    assert actions == []
    assert {s[0] for s in skipped} <= {"concepts/strong"}


def test_cluster_all_cold_is_a_noop(tmp_path):
    _mk(tmp_path, "concepts/a")
    _mk(tmp_path, "concepts/b")
    actions, skipped = maintain.plan_retirements(
        [_pair("concepts/a", "concepts/b", 0.97)],
        knowledge_dir=tmp_path,
        feedback_scores={},
        already_cold={"concepts/a", "concepts/b"},
    )
    assert actions == []  # nothing left to do


def test_cmp_strength_is_reflexive(tmp_path):
    _mk(tmp_path, "concepts/a")
    assert maintain._cmp_strength(
        "concepts/a", "concepts/a", tmp_path, {}
    ) == 0


def test_plan_skips_both_pinned_pair(tmp_path):
    _mk(tmp_path, "concepts/a", pinned=True)
    _mk(tmp_path, "concepts/b", pinned=True)
    actions, skipped = maintain.plan_retirements(
        [_pair("concepts/a", "concepts/b", 0.99)],
        knowledge_dir=tmp_path, feedback_scores={}, already_cold=set(),
    )
    assert actions == []
    assert skipped == [("concepts/a", "concepts/b", 0.99)]


def test_plan_keeps_one_survivor_per_cluster(tmp_path):
    """A 3-way near-dup cluster must keep exactly one survivor — never list an
    article as both kept and retired (the live-dry-run bug)."""
    _mk(tmp_path, "concepts/a")
    _mk(tmp_path, "concepts/b")
    _mk(tmp_path, "concepts/c")
    pairs = [
        _pair("concepts/a", "concepts/b", 0.97),
        _pair("concepts/b", "concepts/c", 0.96),
        _pair("concepts/a", "concepts/c", 0.95),
    ]
    fs = {"concepts/a": 0.2, "concepts/b": 0.5, "concepts/c": 0.9}
    actions, skipped = maintain.plan_retirements(
        pairs, knowledge_dir=tmp_path, feedback_scores=fs, already_cold=set(),
    )
    retired = {x.weaker for x in actions}
    survivors = {x.stronger for x in actions}
    assert retired == {"concepts/a", "concepts/b"}
    assert survivors == {"concepts/c"}        # single survivor
    assert retired.isdisjoint(survivors)       # never both kept and retired


def test_cluster_forms_from_chained_pairs_without_closing_edge(tmp_path):
    """A~B and B~C (no A~C edge) is still ONE cluster → 2 retired, 1 survivor."""
    _mk(tmp_path, "concepts/a")
    _mk(tmp_path, "concepts/b")
    _mk(tmp_path, "concepts/c")
    pairs = [
        _pair("concepts/a", "concepts/b", 0.96),
        _pair("concepts/b", "concepts/c", 0.96),
    ]
    fs = {"concepts/a": 0.9, "concepts/b": 0.5, "concepts/c": 0.2}
    actions, _ = maintain.plan_retirements(
        pairs, knowledge_dir=tmp_path, feedback_scores=fs, already_cold=set(),
    )
    assert {x.weaker for x in actions} == {"concepts/b", "concepts/c"}
    assert {x.stronger for x in actions} == {"concepts/a"}


def test_plan_skips_when_weaker_already_cold(tmp_path):
    _mk(tmp_path, "concepts/keep")
    _mk(tmp_path, "concepts/drop")
    actions, skipped = maintain.plan_retirements(
        [_pair("concepts/keep", "concepts/drop", 0.96)],
        knowledge_dir=tmp_path,
        feedback_scores={"concepts/keep": 0.9, "concepts/drop": 0.2},
        already_cold={"concepts/drop"},
    )
    assert actions == []
    assert len(skipped) == 1
