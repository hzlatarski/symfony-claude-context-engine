"""One process-wide Chroma client, reloaded from disk before every write.

Why this exists — lost vectors (found 2026-10-08):
Chroma's local store keeps each collection's HNSW vector index in process
memory and periodically saves that in-memory copy over the files on disk.
Several memory-compiler processes share one store (compile.py and ingest.py
run for minutes between LLM calls; session-end hooks, the scheduler and
concurrent sessions start more). A process that loaded the index early and
writes late saves its OLD copy, erasing every vector another process added
in between. The write log is purged past the saved point, so those vectors
are gone for good: the live store had 392 of 2,503 article zones in its
metadata but not in its vector index — unreadable ("Error finding id") and
invisible to vector search. ``chroma_write_lock`` alone does not prevent this:
it serializes the write calls, not the in-memory copies.

The rule enforced here: a write runs on a client freshly loaded from disk
while the collection's cross-process write lock is held (``write_session``).
Loading replays the write log, so the writer starts from everything every
other process saved. Reproduced and verified by tests/test_chroma_multiprocess.py.

All stores in this process share this one client because resetting Chroma's
system cache invalidates every client object that came from it.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import chromadb
from chromadb.api.shared_system_client import SharedSystemClient

import config
from chroma_lock import chroma_write_lock
from chroma_path import ensure_configured_chroma_dir

# Shared by every store module: queries hold it too, so no thread can be
# mid-query on a client that another thread's write is replacing.
lock = threading.RLock()

_client: Any = None
_client_dir: Path | None = None
_resolved: dict[Path, Path] = {}


def _db_dir() -> Path:
    configured = Path(config.CHROMA_DB_DIR)
    resolved = _resolved.get(configured)
    if resolved is None:
        resolved = ensure_configured_chroma_dir(configured)
        _resolved[configured] = resolved
    return resolved


def _alive(client: Any) -> bool:
    # A cache reset elsewhere (tests, another store) orphans old clients.
    return getattr(client, "_identifier", None) in SharedSystemClient._identifier_to_system


def get_client() -> Any:
    global _client, _client_dir
    with lock:
        db_dir = _db_dir()
        if _client is None or _client_dir != db_dir or not _alive(_client):
            _client = chromadb.PersistentClient(path=str(db_dir))
            _client_dir = db_dir
        return _client


def reset() -> None:
    """Drop every in-memory index in this process; the next use reloads from disk."""
    global _client, _client_dir
    with lock:
        SharedSystemClient.clear_system_cache()
        _client = None
        _client_dir = None


@contextmanager
def write_session(collection: str) -> Iterator[Any]:
    """Cross-process write lock + a client reloaded from disk inside it."""
    with chroma_write_lock(collection):
        with lock:
            reset()
            yield get_client()
