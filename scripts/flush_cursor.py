"""Per-session transcript cursor — how many turns a session has already flushed.

Without a cursor, every flush re-reads the *last 30 turns of the whole
transcript*. A long session flushes more than once (PreCompact fires, then
SessionEnd fires), so the same turns get summarized repeatedly and the daily
log fills with overlapping, near-duplicate ``### Session`` entries.

The cursor records how far into the transcript a session has been consumed.
Hooks read it to slice off only the turns they have not summarized yet;
``flush.py`` advances it *after* a flush succeeds — so a crashed or errored
flush re-processes its window on the next run instead of losing it.

Cursors live in their own atomically written file rather than cost/dedup
metadata in ``last-flush.json``. This keeps the correctness-critical monotonic
cursor lifecycle isolated from operational history and its retention policy.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock

STATE_FILE = Path(__file__).resolve().parent / "flush-cursors.json"

# Cap the number of sessions tracked so last-flush.json cannot grow without
# bound. Sessions are pruned oldest-first by their recorded cursor write order.
MAX_TRACKED_SESSIONS = 50


def _load_state(state_file: Path | None = None) -> dict:
    path = state_file or STATE_FILE
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def load_cursor(session_id: str, state_file: Path | None = None) -> int:
    """Return how many transcript turns ``session_id`` has already flushed."""
    cursors = _load_state(state_file).get("turn_cursors", {})
    if not isinstance(cursors, dict):
        return 0
    value = cursors.get(session_id, 0)
    if not isinstance(value, int) or value < 0:
        return 0
    return value


@contextmanager
def session_flush_lock(
    session_id: str,
    state_file: Path | None = None,
):
    """Serialize the complete flush transaction for one session."""
    path = state_file or STATE_FILE
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:20]
    lock_path = path.with_name(f"{path.name}.{digest}.flush.lock")
    with FileLock(str(lock_path)):
        yield


def save_cursor(session_id: str, cursor: int, state_file: Path | None = None) -> None:
    """Record that ``session_id`` has been flushed through turn ``cursor``.

    Never moves the cursor backwards: two flushes can race (a PreCompact flush
    still running when SessionEnd fires), and the later-finishing one may carry
    the smaller cursor. Taking the max keeps turns from being re-summarized.
    """
    path = state_file or STATE_FILE
    lock = FileLock(str(path.with_suffix(f"{path.suffix}.lock")))
    with lock:
        state = _load_state(path)

        cursors = state.get("turn_cursors")
        if not isinstance(cursors, dict):
            cursors = {}

        merged = max(cursor, cursors.get(session_id, 0))
        # Re-insert rather than assign in place: assigning to an existing key leaves
        # it at its original position, so a long-running session that first flushed
        # 51 sessions ago would be evicted below while dormant newer sessions
        # survive — resetting its cursor to 0 and re-summarizing its whole history.
        cursors.pop(session_id, None)
        cursors[session_id] = merged

        if len(cursors) > MAX_TRACKED_SESSIONS:
            # dicts preserve insertion order, so the least-recently-written are first
            for stale in list(cursors)[: len(cursors) - MAX_TRACKED_SESSIONS]:
                del cursors[stale]

        state["turn_cursors"] = cursors

        tmp = path.with_suffix(f".json.tmp{os.getpid()}")
        try:
            with tmp.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(state, handle)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
