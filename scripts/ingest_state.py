"""Live progress + cooperative cancellation for the ingest pipeline.

`ingest.py` now writes a one-line JSON status snapshot to
``knowledge/.ingest-status.json`` at each file boundary, and polls
``knowledge/.ingest-stop`` for a stop signal between files. Both the
viewer and the ``ingest_status`` / ``ingest_stop`` MCP tools read and
write through this module so the contract is in one place.

The status file is best-effort — losing it doesn't break ingestion,
and stale entries from a crashed run are harmless because
``processed`` won't advance. ``read_status()`` reports the file's
``mtime`` so callers can decide whether the run is still alive.

Per-file durability is already handled by ``ingest.ingest_source_file``,
which writes the file hash to ``state.json`` after every Sonnet call.
This module deliberately does NOT duplicate that — it only adds the
live-progress and stop-signal layer that ``state.json`` doesn't expose.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from config import INGEST_STATUS_FILE, INGEST_STOP_FILE


def write_status(
    *,
    phase: str,
    current_file: str | None = None,
    processed: int = 0,
    total: int = 0,
    total_cost: float = 0.0,
    started_at: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write or replace the status snapshot atomically.

    `phase` is one of ``starting``, ``running``, ``finished``, ``stopped``,
    ``error``. Atomic via ``os.replace`` so a partial JSON write never
    corrupts the file mid-poll.
    """
    payload: dict[str, Any] = {
        "phase": phase,
        "current_file": current_file,
        "processed": processed,
        "total": total,
        "total_cost": round(total_cost, 6),
        "started_at": started_at if started_at is not None else time.time(),
        "updated_at": time.time(),
    }
    if extra:
        payload.update(extra)
    INGEST_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = INGEST_STATUS_FILE.with_suffix(INGEST_STATUS_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Windows: os.replace can fail with ERROR_ACCESS_DENIED if a reader
    # (MCP server polling ingest_status, AV scanner, indexer) briefly holds
    # the destination open. Retry with backoff before giving up.
    last_err: OSError | None = None
    for attempt in range(8):
        try:
            os.replace(tmp, INGEST_STATUS_FILE)
            return
        except PermissionError as err:
            last_err = err
            if attempt < 7:
                time.sleep(0.05 * (attempt + 1))
    if last_err is not None:
        raise last_err


def read_status() -> dict[str, Any] | None:
    """Read the most recent status snapshot, or None if nothing has run."""
    if not INGEST_STATUS_FILE.exists():
        return None
    try:
        data = json.loads(INGEST_STATUS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data


def request_stop() -> bool:
    """Touch the stop file. Returns True if newly created, False if already there."""
    INGEST_STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
    if INGEST_STOP_FILE.exists():
        return False
    INGEST_STOP_FILE.write_text("requested", encoding="utf-8")
    return True


def should_stop() -> bool:
    """Check whether a stop has been requested. Cheap — pure file existence."""
    return INGEST_STOP_FILE.exists()


def clear_stop() -> None:
    """Remove the stop file. Called by ingest.py on entry and after honoring."""
    try:
        INGEST_STOP_FILE.unlink()
    except FileNotFoundError:
        pass
