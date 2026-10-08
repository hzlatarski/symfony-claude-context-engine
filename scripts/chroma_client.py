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
import time
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
_client_generation: int = -1
_resolved: dict[Path, Path] = {}

# Touched after every write. A long-lived reader (the MCP server) compares it
# before each use and reloads when another process has written, otherwise it
# keeps serving the index it loaded at start and never sees new vectors.
_GENERATION_FILE = ".write-generation"


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


def _generation(db_dir: Path) -> int:
    try:
        return (db_dir / _GENERATION_FILE).stat().st_mtime_ns
    except FileNotFoundError:
        return 0


def _bump_generation(db_dir: Path) -> int:
    marker = db_dir / _GENERATION_FILE
    marker.write_text(str(time.time_ns()), encoding="utf-8")
    return _generation(db_dir)


def get_client() -> Any:
    """The process client, reloaded first if any process wrote since it loaded."""
    global _client, _client_dir, _client_generation
    with lock:
        db_dir = _db_dir()
        generation = _generation(db_dir)
        stale = _client is not None and (
            _client_dir != db_dir or not _alive(_client) or generation != _client_generation
        )
        if stale:
            reset()
        if _client is None:
            # Read the generation BEFORE loading, so a write that lands during
            # the load is caught by the next call rather than missed.
            _client = chromadb.PersistentClient(path=str(db_dir))
            _client_dir = db_dir
            _client_generation = generation
        return _client


def reset() -> None:
    """Drop every in-memory index in this process; the next use reloads from disk.

    clear_system_cache() only forgets the systems; stopping them is what frees
    their sqlite connection, threads and HNSW copies. Without it each write
    leaked a whole system until the cyclic GC ran (~1.6 GB after 100 writes).
    """
    global _client, _client_dir, _client_generation
    with lock:
        systems = list(SharedSystemClient._identifier_to_system.values())
        SharedSystemClient.clear_system_cache()
        for system in systems:
            try:
                system.stop()
            except Exception:  # an already-stopped or half-built system
                pass
        _client = None
        _client_dir = None
        _client_generation = -1


@contextmanager
def write_session(collection: str) -> Iterator[Any]:
    """Cross-process write lock + a client reloaded from disk inside it."""
    global _client_generation
    with chroma_write_lock(collection):
        with lock:
            reset()
            client = get_client()
            try:
                yield client
            finally:
                # Even a failed write may have changed the store. Tell
                # long-lived readers in other processes to reload; this
                # process already holds the post-write state, so it does not.
                _client_generation = _bump_generation(_db_dir())
