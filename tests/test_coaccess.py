"""Tests for the co-access edge store.

Covers the store (record/load round-trip, canonical pair keys, self/dup
handling), the loss-proof write path (corrupt backup, graceful load), the
recency-weighted edge weight (fresh vs aged, decay below floor), edge emission
with a weight threshold, bounding (per-pair event cap, pair-table eviction),
and the prune maintenance helper.
"""
from datetime import date

import pytest

import coaccess as ca


# ── Store round-trip ─────────────────────────────────────────────────

def test_record_creates_store_and_counts_pairs(tmp_path):
    store = tmp_path / "coaccess.json"
    n = ca.record_coaccess(
        ["concepts/a", "concepts/b", "concepts/c"],
        now="2026-10-21T10:00:00+00:00",
        path=store,
    )
    # 3 articles → C(3,2) = 3 pairs.
    assert n == 3
    pairs = ca.load(store)["pairs"]
    assert set(pairs) == {
        "concepts/a|concepts/b",
        "concepts/a|concepts/c",
        "concepts/b|concepts/c",
    }
    assert pairs["concepts/a|concepts/b"]["count"] == 1


def test_pair_key_is_order_independent(tmp_path):
    store = tmp_path / "coaccess.json"
    ca.record_coaccess(["concepts/b", "concepts/a"], now="2026-10-21T10:00:00+00:00", path=store)
    ca.record_coaccess(["concepts/a", "concepts/b"], now="2026-10-21T11:00:00+00:00", path=store)
    pairs = ca.load(store)["pairs"]
    # Both orderings collapse to one canonical key with two events.
    assert list(pairs) == ["concepts/a|concepts/b"]
    assert pairs["concepts/a|concepts/b"]["count"] == 2


def test_record_strips_md_suffix(tmp_path):
    store = tmp_path / "coaccess.json"
    ca.record_coaccess(["concepts/a.md", "concepts/b.md"], now="2026-10-21T10:00:00+00:00", path=store)
    assert "concepts/a|concepts/b" in ca.load(store)["pairs"]


def test_single_slug_is_noop(tmp_path):
    store = tmp_path / "coaccess.json"
    assert ca.record_coaccess(["concepts/a"], now="2026-10-21T10:00:00+00:00", path=store) == 0
    # Store file is never created for a no-op.
    assert not store.exists()


def test_duplicate_slugs_make_no_self_pair(tmp_path):
    store = tmp_path / "coaccess.json"
    n = ca.record_coaccess(
        ["concepts/a", "concepts/a", "concepts/b"],
        now="2026-10-21T10:00:00+00:00",
        path=store,
    )
    assert n == 1
    assert list(ca.load(store)["pairs"]) == ["concepts/a|concepts/b"]


def test_piped_slug_rejected(tmp_path):
    with pytest.raises(ValueError):
        ca.record_coaccess(["concepts/a|b", "concepts/c"], path=tmp_path / "f.json")


def test_empty_slug_rejected(tmp_path):
    with pytest.raises(ValueError):
        ca.record_coaccess(["", "concepts/b"], path=tmp_path / "f.json")


# ── Loss-proof I/O ───────────────────────────────────────────────────

def test_load_missing_file_returns_empty_store(tmp_path):
    assert ca.load(tmp_path / "nope.json") == {"version": 1, "pairs": {}}


def test_load_corrupt_file_degrades_gracefully(tmp_path):
    store = tmp_path / "coaccess.json"
    store.write_text("{not json", encoding="utf-8")
    assert ca.load(store) == {"version": 1, "pairs": {}}


def test_record_over_corrupt_file_backs_it_up_not_clobbers(tmp_path):
    store = tmp_path / "coaccess.json"
    store.write_text("{garbage not json", encoding="utf-8")
    ca.record_coaccess(["concepts/a", "concepts/b"], now="2026-10-21T10:00:00+00:00", path=store)
    backups = list(tmp_path.glob("coaccess.json.corrupt-*"))
    assert backups, "expected the corrupt file to be preserved as a backup"
    assert "garbage" in backups[0].read_text(encoding="utf-8")
    assert "concepts/a|concepts/b" in ca.load(store)["pairs"]


# ── Bounding ─────────────────────────────────────────────────────────

def test_events_are_capped_but_count_preserved(tmp_path):
    store = tmp_path / "coaccess.json"
    for i in range(ca.MAX_EVENTS_PER_PAIR + 10):
        ca.record_coaccess(
            ["concepts/a", "concepts/b"],
            now=f"2026-10-21T10:{i % 60:02d}:00+00:00",
            path=store,
        )
    rec = ca.load(store)["pairs"]["concepts/a|concepts/b"]
    assert len(rec["events"]) == ca.MAX_EVENTS_PER_PAIR
    assert rec["count"] == ca.MAX_EVENTS_PER_PAIR + 10


def test_pair_table_is_evicted_at_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "MAX_PAIRS", 3)
    store = tmp_path / "coaccess.json"
    # Give concepts/hot repeated events so it is NOT the one evicted.
    for _ in range(3):
        ca.record_coaccess(["concepts/hot", "concepts/warm"], now="2026-10-21T10:00:00+00:00", path=store)
    # Now add more distinct, single-event pairs to blow past MAX_PAIRS.
    for i in range(5):
        ca.record_coaccess([f"concepts/x{i}", f"concepts/y{i}"], now="2026-10-21T10:00:00+00:00", path=store)
    pairs = ca.load(store)["pairs"]
    assert len(pairs) == 3
    # The repeatedly-active pair survives eviction.
    assert "concepts/hot|concepts/warm" in pairs


# ── Weight + decay ───────────────────────────────────────────────────

def test_weight_zero_when_never_coaccessed(tmp_path):
    assert ca.weight_for("concepts/a", "concepts/b", path=tmp_path / "f.json") == 0.0


def test_fresh_single_coaccess_weighs_about_one(tmp_path):
    store = tmp_path / "coaccess.json"
    ca.record_coaccess(["concepts/a", "concepts/b"], now="2026-10-21T10:00:00+00:00", path=store)
    w = ca.weight_for("concepts/a", "concepts/b", today=date(2026, 10, 21), path=store)
    assert w == pytest.approx(1.0, abs=1e-9)


def test_repeated_coaccess_outweighs_single(tmp_path):
    today = date(2026, 10, 21)
    s1 = tmp_path / "a.json"
    s2 = tmp_path / "b.json"
    ca.record_coaccess(["concepts/a", "concepts/b"], now="2026-10-21T10:00:00+00:00", path=s1)
    for h in range(3):
        ca.record_coaccess(["concepts/a", "concepts/b"], now=f"2026-10-21T1{h}:00:00+00:00", path=s2)
    assert ca.weight_for("concepts/a", "concepts/b", today=today, path=s2) > \
        ca.weight_for("concepts/a", "concepts/b", today=today, path=s1)


def test_old_coaccess_decays_toward_zero(tmp_path):
    store = tmp_path / "coaccess.json"
    ca.record_coaccess(["concepts/a", "concepts/b"], now="2026-01-01T10:00:00+00:00", path=store)
    fresh = ca.weight_for("concepts/a", "concepts/b", today=date(2026, 1, 2), path=store)
    aged = ca.weight_for("concepts/a", "concepts/b", today=date(2026, 10, 21), path=store)
    assert aged < fresh
    assert 0.0 < aged < 0.5  # well past several half-lives


# ── Edge emission ────────────────────────────────────────────────────

def test_load_edges_filters_by_min_weight(tmp_path):
    store = tmp_path / "coaccess.json"
    # Fresh pair → weight ~1.0 (kept); a very old pair → weight < floor (dropped).
    ca.record_coaccess(["concepts/a", "concepts/b"], now="2026-10-21T10:00:00+00:00", path=store)
    ca.record_coaccess(["concepts/c", "concepts/d"], now="2025-01-01T10:00:00+00:00", path=store)
    edges = ca.load_edges(today=date(2026, 10, 21), path=store, min_weight=0.5)
    keys = {(e["from"], e["to"]) for e in edges}
    assert ("concepts/a", "concepts/b") in keys
    assert ("concepts/c", "concepts/d") not in keys
    for e in edges:
        assert e["kind"] == "co_access"
        assert e["from"] < e["to"]  # sorted endpoints


def test_load_edges_sorted_heaviest_first(tmp_path):
    store = tmp_path / "coaccess.json"
    ca.record_coaccess(["concepts/a", "concepts/b"], now="2026-10-21T10:00:00+00:00", path=store)
    for h in range(4):
        ca.record_coaccess(["concepts/c", "concepts/d"], now=f"2026-10-21T1{h}:00:00+00:00", path=store)
    edges = ca.load_edges(today=date(2026, 10, 21), path=store, min_weight=0.1)
    assert (edges[0]["from"], edges[0]["to"]) == ("concepts/c", "concepts/d")


# ── Prune ────────────────────────────────────────────────────────────

def test_prune_drops_faded_pairs_only(tmp_path):
    store = tmp_path / "coaccess.json"
    ca.record_coaccess(["concepts/a", "concepts/b"], now="2026-10-21T10:00:00+00:00", path=store)
    ca.record_coaccess(["concepts/c", "concepts/d"], now="2025-01-01T10:00:00+00:00", path=store)
    removed = ca.prune(today=date(2026, 10, 21), path=store, min_weight=0.5)
    assert removed == 1
    assert set(ca.load(store)["pairs"]) == {"concepts/a|concepts/b"}


# ── CoAccessTracker (search→useful bridge) ───────────────────────────

NOW = "2026-10-21T10:00:00+00:00"


def test_tracker_two_useful_records_one_pair(tmp_path):
    store = tmp_path / "coaccess.json"
    t = ca.CoAccessTracker(path=store)
    t.remember(["concepts/a", "concepts/b", "concepts/c"])
    assert t.mark_useful("concepts/a", now=NOW) == 0  # first useful, no prior
    assert t.mark_useful("concepts/b", now=NOW) == 1  # links to a
    assert list(ca.load(store)["pairs"]) == ["concepts/a|concepts/b"]


def test_tracker_one_useful_records_nothing(tmp_path):
    store = tmp_path / "coaccess.json"
    t = ca.CoAccessTracker(path=store)
    t.remember(["concepts/a", "concepts/b"])
    assert t.mark_useful("concepts/a", now=NOW) == 0
    assert not store.exists()  # a lone useful mark writes nothing


def test_tracker_three_useful_record_each_pair_once(tmp_path):
    store = tmp_path / "coaccess.json"
    t = ca.CoAccessTracker(path=store)
    t.remember(["concepts/a", "concepts/b", "concepts/c"])
    t.mark_useful("concepts/a", now=NOW)  # 0
    t.mark_useful("concepts/b", now=NOW)  # +ab
    t.mark_useful("concepts/c", now=NOW)  # +ca, +cb
    t.mark_useful("concepts/a", now=NOW)  # already useful → no-op
    pairs = ca.load(store)["pairs"]
    assert set(pairs) == {
        "concepts/a|concepts/b",
        "concepts/a|concepts/c",
        "concepts/b|concepts/c",
    }
    # Each pair recorded exactly once despite the repeat mark.
    assert all(rec["count"] == 1 for rec in pairs.values())


def test_tracker_useful_with_no_remembered_set_is_noop(tmp_path):
    store = tmp_path / "coaccess.json"
    t = ca.CoAccessTracker(path=store)
    assert t.mark_useful("concepts/orphan", now=NOW) == 0
    assert not store.exists()


def test_tracker_ignores_result_sets_smaller_than_two(tmp_path):
    store = tmp_path / "coaccess.json"
    t = ca.CoAccessTracker(path=store)
    t.remember(["concepts/solo"])
    assert t.mark_useful("concepts/solo", now=NOW) == 0
    assert not store.exists()


def test_tracker_tolerates_malformed_slugs(tmp_path):
    store = tmp_path / "coaccess.json"
    t = ca.CoAccessTracker(path=store)
    # A piped (foreign) slug is dropped from the remembered set, not raised.
    t.remember(["concepts/a", "linked|foreign", "concepts/b"])
    t.mark_useful("concepts/a", now=NOW)
    assert t.mark_useful("concepts/b", now=NOW) == 1
    assert t.mark_useful("linked|foreign", now=NOW) == 0  # malformed → no-op
    assert list(ca.load(store)["pairs"]) == ["concepts/a|concepts/b"]


def test_tracker_overlapping_searches_still_record_pair(tmp_path):
    """Interleaved searches must not lose a legitimate co-useful pair."""
    store = tmp_path / "coaccess.json"
    t = ca.CoAccessTracker(path=store)
    t.remember(["concepts/a", "concepts/b"])
    t.remember(["concepts/a", "concepts/c"])  # a appears in a newer set too
    t.mark_useful("concepts/a", now=NOW)
    t.mark_useful("concepts/b", now=NOW)
    # a–b (from the first search) is still recorded despite the newer a–c set.
    assert "concepts/a|concepts/b" in ca.load(store)["pairs"]


def test_tracker_write_failure_leaves_slug_retryable(tmp_path, monkeypatch):
    """A failed persist must not mark the slug useful — a retry must record it."""
    store = tmp_path / "coaccess.json"
    t = ca.CoAccessTracker(path=store)
    t.remember(["concepts/a", "concepts/b"])
    t.mark_useful("concepts/a", now=NOW)  # no prior → no write

    real_save = ca._save
    calls = {"n": 0}

    def flaky_save(data, path=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("disk full")
        return real_save(data, path)

    monkeypatch.setattr(ca, "_save", flaky_save)
    with pytest.raises(RuntimeError):
        t.mark_useful("concepts/b", now=NOW)  # write raises → b stays unmarked
    monkeypatch.undo()  # restore the real writer

    assert t.mark_useful("concepts/b", now=NOW) == 1  # retry succeeds
    assert "concepts/a|concepts/b" in ca.load(store)["pairs"]


def test_tracker_batches_one_write_per_set(tmp_path, monkeypatch):
    """A mark linking to k prior-useful slugs persists in ONE write, not k."""
    store = tmp_path / "coaccess.json"
    t = ca.CoAccessTracker(path=store)
    t.remember(["concepts/a", "concepts/b", "concepts/c", "concepts/d"])
    t.mark_useful("concepts/a", now=NOW)
    t.mark_useful("concepts/b", now=NOW)
    t.mark_useful("concepts/c", now=NOW)

    saves = {"n": 0}
    real_save = ca._save
    monkeypatch.setattr(ca, "_save", lambda d, p=None: (saves.__setitem__("n", saves["n"] + 1), real_save(d, p))[1])
    # Marking d links it to a, b, c (3 pairs) — but only one store write.
    assert t.mark_useful("concepts/d", now=NOW) == 3
    assert saves["n"] == 1


def test_tracker_uses_most_recent_set_and_is_bounded(tmp_path):
    store = tmp_path / "coaccess.json"
    t = ca.CoAccessTracker(max_sets=1, path=store)
    t.remember(["concepts/a", "concepts/b"])
    t.remember(["concepts/c", "concepts/d"])  # evicts the first set (max_sets=1)
    # 'a' is no longer remembered → its useful mark is a no-op.
    assert t.mark_useful("concepts/a", now=NOW) == 0
    t.mark_useful("concepts/c", now=NOW)
    assert t.mark_useful("concepts/d", now=NOW) == 1
    assert list(ca.load(store)["pairs"]) == ["concepts/c|concepts/d"]
