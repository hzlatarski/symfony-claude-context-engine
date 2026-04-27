"""Cross-process file locks for Chroma write operations.

Two ingest paths can run concurrently — a `SessionEnd` hook flushing a
session, a manually-launched `ingest.py`, a `reindex.py --all`, or the
file watcher (`watch.py`) — and Chroma's local SQLite-backed store does
not coordinate writers across processes. Without locking, concurrent
upserts on the same collection can corrupt the WAL or, more commonly,
yield "database is locked" SQLite errors that surface as opaque MCP
failures.

Each Chroma collection gets its own lock file under
``knowledge/chroma/.locks/<collection>.lock``. Locks are advisory:
read-only operations (queries, counts) are NOT locked, since Chroma
handles concurrent reads safely. Only the upsert/delete paths take the
lock, and only for the duration of the single Chroma call — held just
long enough to serialize writers, not so long that readers stall.

Stale locks from crashed processes are automatically reclaimed by the
``filelock`` library on the next acquisition attempt (it checks PID
liveness on Unix; on Windows the OS releases the file handle when the
crashed process terminates). Acquisition waits up to
``CHROMA_LOCK_TIMEOUT_SECONDS`` (default 60s) before raising.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from config import CHROMA_LOCK_TIMEOUT_SECONDS, CHROMA_LOCKS_DIR


def _lock_path(collection: str) -> Path:
    CHROMA_LOCKS_DIR.mkdir(parents=True, exist_ok=True)
    return CHROMA_LOCKS_DIR / f"{collection}.lock"


@contextmanager
def chroma_write_lock(collection: str, timeout: float | None = None):
    """Acquire the cross-process write lock for a Chroma collection.

    Use as a context manager around upsert/delete calls. Raises
    ``filelock.Timeout`` if another process holds the lock past the
    timeout window — surface that to the caller so the operation can
    be retried at a higher level instead of silently corrupting state.
    """
    wait = CHROMA_LOCK_TIMEOUT_SECONDS if timeout is None else timeout
    lock = FileLock(str(_lock_path(collection)))
    try:
        with lock.acquire(timeout=wait):
            yield
    except Timeout as exc:
        raise TimeoutError(
            f"Timed out after {wait:.0f}s waiting for write lock on Chroma "
            f"collection {collection!r}. Another memory-compiler process "
            f"is likely holding it — check `ps`/Task Manager and retry."
        ) from exc
