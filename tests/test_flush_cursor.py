"""Tests for the flush turn cursor and the lint issue-dict contract.

Every test here pins a defect found in adversarial review of the cursor feature.
The cursor decides which transcript turns get summarized into the permanent
daily log, so a bug here is silent, unrecoverable knowledge loss.
"""
from __future__ import annotations

import json

import pytest

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

    def test_sessions_are_independent(self, tmp_path):
        sf = tmp_path / "cursors.json"
        flush_cursor.save_cursor("a", 10, sf)
        flush_cursor.save_cursor("b", 20, sf)
        assert flush_cursor.load_cursor("a", sf) == 10
        assert flush_cursor.load_cursor("b", sf) == 20

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
        """Cursors must NOT share last-flush.json.

        flush.py's save_flush_state() does a non-atomic read-modify-write of
        last-flush.json. Sharing the file lets a concurrent flush drop a cursor
        another process just committed, or catch a torn write and wipe them all.
        """
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

    def test_window_overflow_is_logged_not_silent(self, tmp_path, caplog):
        """Turns beyond the window get marked flushed but never summarized.

        That gap is real; it must at least be visible in flush.log.
        """
        t = _write_transcript(tmp_path / "t.jsonl", 50)
        with caplog.at_level("WARNING"):
            _, total, window = extract_conversation_context(t, start_turn=0, max_turns=30)
        assert (total, window) == (50, 30)
        assert any("overflow" in r.message.lower() for r in caplog.records)

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
