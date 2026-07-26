"""Shared rotating-file logging for the flush pipeline.

Four processes append to the same ``scripts/flush.log``: ``flush.py`` plus the
``post-tool-use`` / ``pre-compact`` / ``session-end`` hooks. Left unbounded the
file grows forever — the AiTutor copy reached 4.9 MB over ~3.5 months of normal
use, which makes tailing it for a diagnosis slow and costs disk in every
installed project.

Why not ``logging.handlers.RotatingFileHandler``
------------------------------------------------
It rotates *mid-write*: when the size threshold trips it renames the file it is
actively logging to. On Windows that rename raises ``PermissionError`` while any
other process holds a handle — and here processes routinely overlap (a
SessionEnd hook spawns ``flush.py``, which logs for minutes while further hooks
fire). So mid-write rotation is the wrong strategy for this pipeline.

We rotate at *startup* instead, before this process opens the file.

Concurrency
-----------
Startup rotation is still a multi-step shift (``.2→.3``, ``.1→.2``, ``log→.1``),
and four processes can start close enough together to all pass the size check.
Interleaved, their shifts would advance the generations several times in a row
and push live data off the end — one process can delete the log another just
rotated in. Suppressing each ``OSError`` does not make the *sequence* safe.

So the whole shift is guarded by an exclusive lock file created with
``O_CREAT|O_EXCL``, which is atomic on both Windows and POSIX. Exactly one
process rotates; the rest skip and simply append. The winner re-checks the size
while holding the lock, so a process that queued behind a rotation does not
immediately rotate again.

Rotation remains best-effort: a failure must never take down the pipeline, since
that would suppress the very flush errors this log exists to record. But a
silent failure is its own trap — if a long-running ``flush.py`` holds the handle,
the rename fails and the log would grow unbounded again with nothing explaining
why. So failures are recorded (see ``configure``) rather than swallowed.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

# 2 MB per file with 3 generations kept ≈ 8 MB worst case per project. Large
# enough to hold weeks of flush history for diagnosis, small enough to tail.
MAX_BYTES = 2 * 1024 * 1024
BACKUP_COUNT = 3

# A crash between acquiring the lock and releasing it would otherwise disable
# rotation permanently. Any lock older than this is treated as abandoned.
_LOCK_STALE_SECONDS = 60.0


def _gen(log_file: Path, index: int) -> Path:
    """Path of backup generation ``index`` (``flush.log.2``)."""
    return log_file.with_name(f"{log_file.name}.{index}")


def _claim_lock(lock: Path) -> int | None:
    """Atomically create ``lock``; return its fd, or None if someone else holds it.

    A lock older than ``_LOCK_STALE_SECONDS`` is abandoned (a process killed
    mid-rotation) and stolen, so a crash cannot disable rotation forever.

    The steal must not be a plain ``unlink`` + recreate: two processes could
    both judge the lock stale, and the second's unlink would delete the fresh
    lock the first had just created, leaving two simultaneous "owners". Instead
    the stale lock is *renamed away* with ``os.replace``, which is atomic —
    exactly one process can win that move, and the loser's replace fails and
    backs off.
    """
    for attempt in (0, 1):
        try:
            return os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if attempt:
                return None
            try:
                if time.time() - lock.stat().st_mtime < _LOCK_STALE_SECONDS:
                    return None
                victim = lock.with_name(f"{lock.name}.stale.{os.getpid()}")
                os.replace(lock, victim)  # atomic claim of the steal
                os.unlink(victim)
            except OSError:
                return None  # lost the steal race, or it vanished — back off
        except OSError:
            return None
    return None


def _shift_generations(log_file: Path) -> list[str]:
    """``.2→.3``, ``.1→.2``, ``log→.1``, dropping the oldest. Lock must be held.

    Returns warnings for *unexpected* failures. A missing generation is normal
    (there is no ``.1`` before the first rotation) and is not reported; a
    permission error is not normal and must not be swallowed, or the log
    silently stops rotating with nothing to explain why.
    """
    problems: list[str] = []
    try:
        _gen(log_file, BACKUP_COUNT).unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        problems.append(f"drop .{BACKUP_COUNT}: {type(exc).__name__}")
    # Walk downwards so a generation is never overwritten before it has moved.
    for index in range(BACKUP_COUNT - 1, 0, -1):
        try:
            os.replace(_gen(log_file, index), _gen(log_file, index + 1))
        except FileNotFoundError:
            pass
        except OSError as exc:
            problems.append(f".{index}->.{index + 1}: {type(exc).__name__}")
    os.replace(log_file, _gen(log_file, 1))  # caller reports failure
    return problems


def _rotate_if_needed(log_file: Path) -> str | None:
    """Rotate ``log_file`` if oversize. Returns a reason string on failure.

    ``None`` means "nothing to do, or rotated cleanly" — both are success.
    """
    try:
        if not log_file.exists() or log_file.stat().st_size < MAX_BYTES:
            return None
    except OSError:
        return None

    lock = log_file.with_name(log_file.name + ".rotating")
    fd = _claim_lock(lock)
    if fd is None:
        return None  # another process is rotating; it will handle it

    try:
        # Re-check under the lock: we may have queued behind the process that
        # just rotated, and rotating again would discard a fresh generation.
        try:
            if not log_file.exists() or log_file.stat().st_size < MAX_BYTES:
                return None
        except OSError:
            return None
        try:
            problems = _shift_generations(log_file)
        except OSError as exc:
            # Almost always Windows refusing to rename a file another writer
            # still holds open. Appending to an oversize log beats crashing.
            return f"{type(exc).__name__}: {exc}"
        return "; ".join(problems) if problems else None
    finally:
        os.close(fd)
        try:
            lock.unlink()
        except OSError:
            pass


def configure(log_file: Path | str, fmt: str, level: int = logging.INFO) -> None:
    """Rotate if over budget, then point ``logging`` at ``log_file``.

    ``fmt`` carries each writer's own prefix (e.g. ``"[hook]"``) so the shared
    log stays attributable to the process that wrote each line.
    """
    log_file = Path(log_file)
    failure = _rotate_if_needed(log_file)
    logging.basicConfig(
        filename=str(log_file),
        level=level,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if failure:
        # Logged *after* basicConfig so it lands in the very file that failed to
        # rotate — the one place someone investigating unbounded growth looks.
        logging.warning(
            "flush.log rotation skipped (%s) — file is over %d bytes and still "
            "growing; a long-running writer probably holds the handle open.",
            failure, MAX_BYTES,
        )
