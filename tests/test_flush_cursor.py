"""Tests for the flush turn cursor and the lint issue-dict contract.

Every test here pins a defect found in adversarial review of the cursor feature.
The cursor decides which transcript turns get summarized into the permanent
daily log, so a bug here is silent, unrecoverable knowledge loss.
"""
from __future__ import annotations

import json
import threading

import pytest
from filelock import FileLock

from scripts import flush_cursor
from scripts.transcript import extract_conversation_context


def _write_transcript(path, n_turns: int):
    with path.open("w", encoding="utf-8") as f:
        for i in range(n_turns):
            role = "user" if i % 2 == 0 else "assistant"
            f.write(json.dumps({"message": {"role": role, "content": f"turn {i}"}}) + "\n")
    return path


class TestCursorStore:
    def test_missing_file_reads_as_zero(self, tmp_path):
        assert flush_cursor.load_cursor("s1", tmp_path / "nope.json") == 0

    def test_round_trip(self, tmp_path):
        sf = tmp_path / "cursors.json"
        flush_cursor.save_cursor("s1", 42, sf)
        assert flush_cursor.load_cursor("s1", sf) == 42

    def test_cursor_never_moves_backwards(self, tmp_path):
        """A late-finishing older flush must not rewind a newer one.

        PreCompact and SessionEnd flushes overlap; whichever writes last may
        carry the smaller cursor. Rewinding would re-summarize turns already in
        the daily log — the duplicate-entry bug this feature exists to fix.
        """
        sf = tmp_path / "cursors.json"
        flush_cursor.save_cursor("s1", 100, sf)
        flush_cursor.save_cursor("s1", 40, sf)
        assert flush_cursor.load_cursor("s1", sf) == 100

    def test_cursor_is_fsynced_before_atomic_publish(self, tmp_path, monkeypatch):
        events = []
        real_replace = flush_cursor.os.replace
        monkeypatch.setattr(
            flush_cursor.os,
            "fsync",
            lambda _fd: events.append("fsync"),
        )

        def recording_replace(source, target):
            events.append("replace")
            real_replace(source, target)

        monkeypatch.setattr(flush_cursor.os, "replace", recording_replace)

        flush_cursor.save_cursor("s1", 10, tmp_path / "cursors.json")

        assert events == ["fsync", "replace"]

    def test_sessions_are_independent(self, tmp_path):
        sf = tmp_path / "cursors.json"
        flush_cursor.save_cursor("a", 10, sf)
        flush_cursor.save_cursor("b", 20, sf)
        assert flush_cursor.load_cursor("a", sf) == 10
        assert flush_cursor.load_cursor("b", sf) == 20

    def test_save_waits_for_cross_process_lock(self, tmp_path):
        sf = tmp_path / "cursors.json"
        lock = FileLock(str(sf.with_suffix(".json.lock")))
        finished = threading.Event()

        with lock:
            worker = threading.Thread(
                target=lambda: (
                    flush_cursor.save_cursor("s1", 42, sf),
                    finished.set(),
                ),
            )
            worker.start()
            assert not finished.wait(0.1), "cursor write ignored the process lock"

        worker.join(timeout=2)
        assert finished.is_set()
        assert flush_cursor.load_cursor("s1", sf) == 42

    def test_whole_flush_lock_serializes_same_session(self, tmp_path):
        assert hasattr(flush_cursor, "session_flush_lock")
        sf = tmp_path / "cursors.json"
        entered = threading.Event()

        with flush_cursor.session_flush_lock("same-session", sf):
            worker = threading.Thread(
                target=lambda: (
                    flush_cursor.session_flush_lock("same-session", sf).__enter__(),
                    entered.set(),
                ),
            )
            worker.start()
            assert not entered.wait(0.1)

        worker.join(timeout=2)
        assert entered.is_set()

    def test_corrupt_file_reads_as_zero_instead_of_crashing(self, tmp_path):
        sf = tmp_path / "cursors.json"
        sf.write_text("{not json", encoding="utf-8")
        assert flush_cursor.load_cursor("s1", sf) == 0

    def test_negative_or_nonint_cursor_is_ignored(self, tmp_path):
        sf = tmp_path / "cursors.json"
        sf.write_text(json.dumps({"turn_cursors": {"s1": -5, "s2": "x"}}), encoding="utf-8")
        assert flush_cursor.load_cursor("s1", sf) == 0
        assert flush_cursor.load_cursor("s2", sf) == 0

    def test_updating_a_session_keeps_it_from_being_evicted(self, tmp_path):
        """Re-inserting on update makes eviction least-recently-written.

        Assigning in place would leave a long-lived session pinned at its
        original position, so it gets evicted while dormant newer sessions
        survive — resetting its cursor to 0 and re-summarizing its history.
        """
        sf = tmp_path / "cursors.json"
        flush_cursor.save_cursor("old", 5, sf)
        for i in range(flush_cursor.MAX_TRACKED_SESSIONS - 1):
            flush_cursor.save_cursor(f"s{i}", 1, sf)

        # "old" is about to fall off the front — touch it, then push one more in.
        flush_cursor.save_cursor("old", 6, sf)
        flush_cursor.save_cursor("newest", 1, sf)

        assert flush_cursor.load_cursor("old", sf) == 6, "recently-written session was evicted"

    def test_store_is_capped(self, tmp_path):
        sf = tmp_path / "cursors.json"
        for i in range(flush_cursor.MAX_TRACKED_SESSIONS + 25):
            flush_cursor.save_cursor(f"s{i}", 1, sf)
        data = json.loads(sf.read_text(encoding="utf-8"))
        assert len(data["turn_cursors"]) <= flush_cursor.MAX_TRACKED_SESSIONS

    def test_cursors_live_outside_last_flush_json(self):
        """Correctness-critical cursors stay separate from cost/dedup history."""
        assert flush_cursor.STATE_FILE.name != "last-flush.json"


class TestTranscriptWindow:
    def test_cold_start_reads_everything(self, tmp_path):
        t = _write_transcript(tmp_path / "t.jsonl", 10)
        ctx, total, window = extract_conversation_context(t, start_turn=0)
        assert (total, window) == (10, 10)
        assert "turn 0" in ctx and "turn 9" in ctx

    def test_cursor_skips_already_flushed_turns(self, tmp_path):
        t = _write_transcript(tmp_path / "t.jsonl", 10)
        ctx, total, window = extract_conversation_context(t, start_turn=7)
        assert (total, window) == (10, 3)
        assert "turn 6" not in ctx
        assert "turn 7" in ctx

    def test_cursor_at_end_yields_nothing_new(self, tmp_path):
        t = _write_transcript(tmp_path / "t.jsonl", 10)
        ctx, total, window = extract_conversation_context(t, start_turn=10)
        assert (total, window) == (10, 0)
        assert ctx == ""

    def test_stale_oversized_cursor_does_not_reflush_everything(self, tmp_path):
        """A cursor past the end must clamp, not wrap into a negative slice."""
        t = _write_transcript(tmp_path / "t.jsonl", 10)
        ctx, total, window = extract_conversation_context(t, start_turn=999)
        assert (total, window) == (10, 0)
        assert ctx == ""

    def test_negative_cursor_is_clamped(self, tmp_path):
        t = _write_transcript(tmp_path / "t.jsonl", 5)
        _, total, window = extract_conversation_context(t, start_turn=-3)
        assert (total, window) == (5, 5)

    def test_default_window_preserves_every_fresh_turn(self, tmp_path):
        t = _write_transcript(tmp_path / "t.jsonl", 50)
        ctx, cursor, window = extract_conversation_context(t, start_turn=0)
        assert (cursor, window) == (50, 50)
        assert "turn 0" in ctx and "turn 49" in ctx

    def test_bounded_window_consumes_oldest_turns_and_advances_only_past_them(self, tmp_path):
        t = _write_transcript(tmp_path / "t.jsonl", 50)
        ctx, cursor, window = extract_conversation_context(t, start_turn=0, max_turns=30)
        assert (cursor, window) == (30, 30)
        assert "turn 0" in ctx and "turn 29" in ctx
        assert "turn 30" not in ctx and "turn 49" not in ctx

    def test_character_budget_never_partially_consumes_a_turn(self, tmp_path):
        t = _write_transcript(tmp_path / "t.jsonl", 3)
        first, _, _ = extract_conversation_context(t, max_turns=1)
        ctx, cursor, window = extract_conversation_context(
            t,
            max_context_chars=len(first) + 1,
        )
        assert (cursor, window) == (1, 1)
        assert "turn 0" in ctx
        assert "turn 1" not in ctx

    def test_total_turns_is_the_cursor_to_record(self, tmp_path):
        t = _write_transcript(tmp_path / "t.jsonl", 12)
        _, total, _ = extract_conversation_context(t, start_turn=4)
        assert total == 12, "cursor must be an absolute turn count, not a window size"


class TestLintIssueContract:
    """Every lint issue dict must carry 'severity' — the report indexes it."""

    def test_new_checks_emit_severity(self, monkeypatch):
        import lint

        class FakePair:
            slug_a, slug_b, similarity, identical = "concepts/a", "concepts/b", 0.99, True

        import dedup
        monkeypatch.setattr(dedup, "find_near_duplicates", lambda **k: [FakePair()])
        monkeypatch.setattr(dedup, "find_stale_vectors", lambda: ["concepts/ghost"])

        issues = lint.check_near_duplicates() + lint.check_stale_vectors()
        assert issues, "fixture should produce issues"
        for issue in issues:
            assert "severity" in issue, f"missing severity: {issue}"
            assert issue["severity"] in {"error", "warning", "suggestion"}
            assert "check" in issue and "file" in issue and "detail" in issue

    def test_scan_failure_still_emits_a_valid_issue(self, monkeypatch):
        """Even the error path must satisfy the contract, or lint crashes on it."""
        import dedup
        import lint

        def boom(**kwargs):
            raise RuntimeError("chroma is down")

        monkeypatch.setattr(dedup, "find_near_duplicates", boom)
        monkeypatch.setattr(dedup, "find_stale_vectors", boom)

        for issue in lint.check_near_duplicates() + lint.check_stale_vectors():
            assert issue["severity"] in {"error", "warning", "suggestion"}

    def test_generate_report_accepts_the_new_issues(self, monkeypatch):
        """The actual crash: report indexes i['severity'] unconditionally."""
        import dedup
        import lint

        class FakePair:
            slug_a, slug_b, similarity, identical = "concepts/a", "concepts/b", 0.99, True

        monkeypatch.setattr(dedup, "find_near_duplicates", lambda **k: [FakePair()])
        monkeypatch.setattr(dedup, "find_stale_vectors", lambda: ["concepts/ghost"])

        issues = lint.check_near_duplicates() + lint.check_stale_vectors()
        report = lint.generate_report(issues)  # must not raise KeyError
        assert isinstance(report, str)
