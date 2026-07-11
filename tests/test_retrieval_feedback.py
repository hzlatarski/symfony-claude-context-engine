"""Tests for the retrieval-outcome feedback loop.

Covers the store (record/load/aggregate), the recency-weighted scoring
(neutral default, useful raises, corrected sinks harder than dead_end,
old events decay), and the reflect.py digest bucketing.
"""
from datetime import date

import pytest

import retrieval_feedback as rf
import reflect


# ── Store round-trip ─────────────────────────────────────────────────

def test_record_creates_store_and_increments_counts(tmp_path):
    store = tmp_path / "feedback.json"
    rf.record("concepts/foo", "useful", now="2026-07-11T10:00:00+00:00", path=store)
    rf.record("concepts/foo", "useful", now="2026-07-11T11:00:00+00:00", path=store)
    rf.record("concepts/foo", "corrected", now="2026-07-11T12:00:00+00:00", path=store)

    data = rf.load(store)
    rec = data["outcomes"]["concepts/foo"]
    assert rec["useful"] == 2
    assert rec["corrected"] == 1
    assert rec["dead_end"] == 0
    assert len(rec["events"]) == 3


def test_record_strips_md_suffix(tmp_path):
    store = tmp_path / "feedback.json"
    rf.record("concepts/foo.md", "useful", now="2026-07-11T10:00:00+00:00", path=store)
    assert "concepts/foo" in rf.load(store)["outcomes"]
    assert "concepts/foo.md" not in rf.load(store)["outcomes"]


def test_record_rejects_unknown_outcome(tmp_path):
    with pytest.raises(ValueError):
        rf.record("concepts/foo", "great", path=tmp_path / "f.json")


def test_load_missing_file_returns_empty_store(tmp_path):
    data = rf.load(tmp_path / "nope.json")
    assert data == {"version": 1, "outcomes": {}}


def test_load_corrupt_file_degrades_gracefully(tmp_path):
    store = tmp_path / "feedback.json"
    store.write_text("{not json", encoding="utf-8")
    assert rf.load(store) == {"version": 1, "outcomes": {}}


def test_record_over_corrupt_file_backs_it_up_not_clobbers(tmp_path):
    """M4: a corrupt store is moved aside, not silently overwritten."""
    store = tmp_path / "feedback.json"
    store.write_text("{garbage not json", encoding="utf-8")
    rf.record("concepts/foo", "useful", now="2026-07-11T10:00:00+00:00", path=store)
    # A .corrupt-* sibling preserves the damaged bytes.
    backups = list(tmp_path.glob("feedback.json.corrupt-*"))
    assert backups, "expected the corrupt file to be preserved as a backup"
    assert "garbage" in backups[0].read_text(encoding="utf-8")
    # And the new store is valid with the recorded event.
    assert "concepts/foo" in rf.load(store)["outcomes"]


def test_bad_timestamp_event_is_skipped_not_anchored(tmp_path):
    """L3: an unparseable `at` must not weight the score at full strength forever."""
    store = tmp_path / "feedback.json"
    # One good recent 'corrected' + one malformed-timestamp 'useful'.
    rf.record("concepts/foo", "corrected", now="2026-07-11T10:00:00+00:00", path=store)
    data = rf.load(store)
    data["outcomes"]["concepts/foo"]["events"].append(
        {"outcome": "useful", "at": "not-a-date", "note": ""}
    )
    store.write_text(__import__("json").dumps(data), encoding="utf-8")
    # Score should reflect only the 'corrected' event (below neutral), not be
    # dragged up by the malformed 'useful'.
    assert rf.score_for("concepts/foo", today=date(2026, 7, 11), path=store) < rf.NEUTRAL_SCORE


def test_events_are_capped(tmp_path):
    store = tmp_path / "feedback.json"
    for i in range(rf.MAX_EVENTS_PER_SLUG + 10):
        rf.record("concepts/foo", "useful", now=f"2026-07-11T10:{i % 60:02d}:00+00:00", path=store)
    rec = rf.load(store)["outcomes"]["concepts/foo"]
    assert len(rec["events"]) == rf.MAX_EVENTS_PER_SLUG
    # Lifetime count is preserved despite event trimming.
    assert rec["useful"] == rf.MAX_EVENTS_PER_SLUG + 10


# ── Scoring ──────────────────────────────────────────────────────────

def test_score_neutral_when_no_feedback(tmp_path):
    assert rf.score_for("concepts/missing", path=tmp_path / "f.json") == rf.NEUTRAL_SCORE


def test_useful_raises_score_above_neutral(tmp_path):
    store = tmp_path / "feedback.json"
    today = date(2026, 7, 11)
    rf.record("concepts/foo", "useful", now="2026-07-11T10:00:00+00:00", path=store)
    assert rf.score_for("concepts/foo", today=today, path=store) > rf.NEUTRAL_SCORE


def test_corrected_sinks_harder_than_dead_end(tmp_path):
    today = date(2026, 7, 11)
    s1 = tmp_path / "a.json"
    s2 = tmp_path / "b.json"
    rf.record("concepts/foo", "dead_end", now="2026-07-11T10:00:00+00:00", path=s1)
    rf.record("concepts/foo", "corrected", now="2026-07-11T10:00:00+00:00", path=s2)
    dead = rf.score_for("concepts/foo", today=today, path=s1)
    corrected = rf.score_for("concepts/foo", today=today, path=s2)
    assert corrected < dead < rf.NEUTRAL_SCORE


def test_old_negative_decays_toward_neutral(tmp_path):
    store = tmp_path / "feedback.json"
    rf.record("concepts/foo", "corrected", now="2026-01-01T10:00:00+00:00", path=store)
    recent = rf.score_for("concepts/foo", today=date(2026, 1, 2), path=store)
    aged = rf.score_for("concepts/foo", today=date(2026, 7, 11), path=store)
    # A months-old correction should have decayed closer to neutral.
    assert aged > recent
    assert aged < rf.NEUTRAL_SCORE


def test_load_scores_only_includes_rated_slugs(tmp_path):
    store = tmp_path / "feedback.json"
    rf.record("concepts/foo", "useful", now="2026-07-11T10:00:00+00:00", path=store)
    scores = rf.load_scores(today=date(2026, 7, 11), path=store)
    assert set(scores) == {"concepts/foo"}


# ── aggregate + reflect ──────────────────────────────────────────────

def test_aggregate_sorts_contested_first(tmp_path):
    store = tmp_path / "feedback.json"
    rf.record("concepts/good", "useful", now="2026-07-11T10:00:00+00:00", path=store)
    rf.record("concepts/bad", "corrected", now="2026-07-11T10:00:00+00:00", path=store)
    rf.record("concepts/bad", "corrected", now="2026-07-11T11:00:00+00:00", path=store)
    agg = rf.aggregate(today=date(2026, 7, 11), path=store)
    assert agg[0]["slug"] == "concepts/bad"
    assert agg[-1]["slug"] == "concepts/good"


def test_reflect_build_lessons_buckets_trusted_and_contested(tmp_path, monkeypatch):
    store = tmp_path / "feedback.json"
    rf.record("concepts/good", "useful", now="2026-07-11T10:00:00+00:00", path=store)
    rf.record("concepts/good", "useful", now="2026-07-11T11:00:00+00:00", path=store)
    rf.record("concepts/bad", "corrected", now="2026-07-11T10:00:00+00:00", path=store)

    # build_lessons reads via the default store path — point it at our tmp.
    content = reflect.build_lessons(today=date(2026, 7, 11), path=store)
    assert "Contested" in content
    assert "Trusted" in content
    assert "[[concepts/bad]]" in content
    assert "[[concepts/good]]" in content
