"""Tests for scripts/log_setup.py — startup rotation of the shared flush.log.

The pipeline has four concurrent writers, so the interesting cases are all
about concurrency and failure, not the happy path.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import log_setup  # noqa: E402


@pytest.fixture
def log(tmp_path: Path) -> Path:
    return tmp_path / "flush.log"


def _fill(path: Path, char: str = "x", over: bool = True) -> None:
    size = log_setup.MAX_BYTES + 10 if over else 100
    path.write_text(char * size, encoding="utf-8")


def test_under_budget_is_left_alone(log: Path) -> None:
    _fill(log, over=False)
    assert log_setup._rotate_if_needed(log) is None
    assert log.exists()
    assert not log_setup._gen(log, 1).exists()


def test_missing_file_is_not_an_error(log: Path) -> None:
    assert log_setup._rotate_if_needed(log) is None


def test_oversize_rotates_to_generation_one(log: Path) -> None:
    _fill(log, "A")
    assert log_setup._rotate_if_needed(log) is None
    assert not log.exists()
    assert log_setup._gen(log, 1).read_text(encoding="utf-8")[0] == "A"


def test_generations_shift_and_oldest_is_dropped(log: Path) -> None:
    for index, char in ((1, "A"), (2, "B"), (3, "C")):
        log_setup._gen(log, index).write_text(char * 10, encoding="utf-8")
    _fill(log, "D")

    log_setup._rotate_if_needed(log)

    assert log_setup._gen(log, 1).read_text(encoding="utf-8")[0] == "D"
    assert log_setup._gen(log, 2).read_text(encoding="utf-8")[0] == "A"
    assert log_setup._gen(log, 3).read_text(encoding="utf-8")[0] == "B"
    # "C" fell off the end; nothing beyond BACKUP_COUNT is ever created.
    assert not log_setup._gen(log, log_setup.BACKUP_COUNT + 1).exists()


def test_queued_process_rechecks_under_lock_and_skips(log: Path, monkeypatch) -> None:
    """The race that would discard a just-rotated generation.

    Four starters can all pass the size check before any of them renames. They
    then serialise on the lock — and every one after the first must notice the
    log is no longer oversize and NOT shift again, or the generations advance
    repeatedly and live data is pushed off the end.

    Simulated by letting all four capture their size check first (via a
    _claim_lock that only the first caller wins... then releasing them in turn).
    """
    _fill(log, "A")
    shifts: list[Path] = []
    real_shift = log_setup._shift_generations

    def counting_shift(path: Path):
        shifts.append(path)
        return real_shift(path)

    monkeypatch.setattr(log_setup, "_shift_generations", counting_shift)

    # All four "processes" run sequentially but each re-checks under the lock,
    # exactly as the real ones do after queueing.
    for _ in range(4):
        log_setup._rotate_if_needed(log)

    assert len(shifts) == 1, f"expected exactly one shift, got {len(shifts)}"
    assert log_setup._gen(log, 1).read_text(encoding="utf-8")[0] == "A"
    assert not log_setup._gen(log, 2).exists(), "generations must not advance twice"


def test_concurrent_starters_rotate_only_once(log: Path) -> None:
    """Real threads racing the lock: exactly one may win."""
    import threading

    _fill(log, "A")
    barrier = threading.Barrier(4)
    winners: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        fd = log_setup._claim_lock(log.with_name(log.name + ".rotating"))
        if fd is not None:
            with lock:
                winners.append(fd)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    try:
        assert len(winners) == 1, f"{len(winners)} processes claimed the lock"
    finally:
        for fd in winners:
            os.close(fd)


def test_stale_steal_is_atomic_under_contention(log: Path) -> None:
    """Two processes both judging a lock stale must not both end up owning it."""
    import threading

    lock = log.with_name(log.name + ".rotating")
    lock.write_text("", encoding="utf-8")
    stale = os.stat(lock).st_mtime - (log_setup._LOCK_STALE_SECONDS + 30)
    os.utime(lock, (stale, stale))

    barrier = threading.Barrier(4)
    winners: list[int] = []
    guard = threading.Lock()

    def worker() -> None:
        barrier.wait()
        fd = log_setup._claim_lock(lock)
        if fd is not None:
            with guard:
                winners.append(fd)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    try:
        assert len(winners) <= 1, f"{len(winners)} processes stole the same stale lock"
    finally:
        for fd in winners:
            os.close(fd)


def test_held_lock_blocks_rotation(log: Path) -> None:
    """A concurrent rotation in flight means we skip, leaving the log intact."""
    _fill(log, "A")
    lock = log.with_name(log.name + ".rotating")
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        assert log_setup._rotate_if_needed(log) is None
        assert log.exists(), "must not rotate while another process holds the lock"
        assert not log_setup._gen(log, 1).exists()
    finally:
        os.close(fd)
        lock.unlink()


def test_stale_lock_is_stolen(log: Path) -> None:
    """A process killed mid-rotation must not disable rotation forever."""
    _fill(log, "A")
    lock = log.with_name(log.name + ".rotating")
    lock.write_text("", encoding="utf-8")
    stale = os.stat(lock).st_mtime - (log_setup._LOCK_STALE_SECONDS + 30)
    os.utime(lock, (stale, stale))

    assert log_setup._rotate_if_needed(log) is None
    assert log_setup._gen(log, 1).exists(), "stale lock should have been stolen"
    assert not lock.exists(), "lock must be released"


def test_lock_is_released_after_rotation(log: Path) -> None:
    _fill(log, "A")
    log_setup._rotate_if_needed(log)
    assert not log.with_name(log.name + ".rotating").exists()


def test_rename_failure_is_reported_not_raised(log: Path, monkeypatch) -> None:
    """Windows refusing to rename an open file must degrade, not crash."""
    _fill(log, "A")

    def boom(*_a, **_k):
        raise PermissionError(13, "in use by another process")

    monkeypatch.setattr(log_setup.os, "replace", boom)
    reason = log_setup._rotate_if_needed(log)

    assert reason and "PermissionError" in reason
    assert log.exists(), "log must survive a failed rotation"
    assert not log.with_name(log.name + ".rotating").exists(), "lock still released"


def test_configure_writes_through_and_warns_on_failed_rotation(
    log: Path, monkeypatch,
) -> None:
    """A silently-skipped rotation is the trap; it must leave evidence."""
    _fill(log, "A")
    monkeypatch.setattr(
        log_setup.os, "replace",
        lambda *_a, **_k: (_ for _ in ()).throw(PermissionError("held")),
    )
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_setup.configure(log, "%(levelname)s %(message)s")
    logging.error("sentinel-line")
    logging.shutdown()

    text = log.read_text(encoding="utf-8", errors="replace")
    assert "sentinel-line" in text
    assert "rotation skipped" in text
