"""Durable work records for detached session flushes."""

from __future__ import annotations

import json
import hashlib
import os
import threading
import uuid
from pathlib import Path


def pending_job_path(context_file: Path) -> Path:
    return context_file.with_name(f"{context_file.name}.pending.json")


def create_pending_flush(
    state_dir: Path,
    prefix: str,
    context: str,
    session_id: str,
    new_cursor: int,
    transcript_archive: Path,
) -> Path:
    """Publish a self-contained job, then materialize its context payload."""
    state_dir.mkdir(parents=True, exist_ok=True)
    session_digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:20]
    job_id = uuid.uuid4().hex
    context_file = state_dir / f"{prefix}-{session_digest}-{job_id}.md"
    queue_pending_flush(
        context_file,
        session_id,
        new_cursor,
        transcript_archive,
        context=context,
    )
    _write_context_atomic(context_file, context)
    return context_file


def _write_context_atomic(context_file: Path, context: str) -> None:
    tmp = context_file.with_name(
        f".{context_file.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(context)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, context_file)
    finally:
        tmp.unlink(missing_ok=True)


def queue_pending_flush(
    context_file: Path,
    session_id: str,
    new_cursor: int,
    transcript_archive: Path,
    *,
    context: str | None = None,
) -> Path:
    """Persist all arguments needed to retry a detached flush."""
    marker = pending_job_path(context_file)
    payload = {
        "context_file": str(context_file),
        "session_id": session_id,
        "new_cursor": new_cursor,
        "transcript_archive": str(transcript_archive),
    }
    if context is not None:
        payload["context"] = context
    tmp = marker.with_name(
        f".{marker.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, marker)
    finally:
        tmp.unlink(missing_ok=True)
    return marker


def load_pending_flushes(state_dir: Path) -> list[tuple[Path, dict]]:
    """Return valid pending jobs oldest-first, raising on corrupt records."""
    jobs = []
    for marker in sorted(state_dir.glob("*.md.pending.json")):
        payload = json.loads(marker.read_text(encoding="utf-8"))
        required = {
            "context_file",
            "session_id",
            "new_cursor",
            "transcript_archive",
        }
        if not required.issubset(payload):
            raise ValueError(f"incomplete pending flush record: {marker}")
        context_file = Path(payload["context_file"])
        if not context_file.exists():
            if "context" not in payload:
                raise ValueError(
                    f"pending flush context is missing and not recoverable: {marker}"
                )
            _write_context_atomic(context_file, payload["context"])
        jobs.append((marker, payload))
    return jobs


def remove_pending_flush(context_file: Path) -> None:
    pending_job_path(context_file).unlink(missing_ok=True)
