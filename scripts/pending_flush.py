"""Durable work records for detached session flushes."""

from __future__ import annotations

import json
import hashlib
import os
import sys
import threading
import time
import uuid
from pathlib import Path

from filelock import FileLock

# A job that keeps failing must stop being respawned. Without a ceiling one
# permanently-broken job is relaunched by every capture hook forever, and the
# per-session lock turns that into a pile of processes waiting on each other.
MAX_FLUSH_ATTEMPTS = 5

# How long a handed-out job is assumed to be running. SessionEnd and PreCompact
# can fire back to back for the same session; without this both would launch a
# worker for the same job. Deliberately a soft lease, not an ownership
# transfer: the marker is never hidden or renamed, so an expired lease always
# returns the job. A worker whose process was never created costs a delayed
# retry, never a lost one.
SPAWN_LEASE_SECONDS = 300


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


def _queue_lock(state_dir: Path) -> FileLock:
    """Serialize queue mutations so two hooks cannot race on the same marker."""
    return FileLock(str(state_dir / ".pending-queue.lock"))


def _write_payload_atomic(marker: Path, payload: dict) -> None:
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


def queue_pending_flush(
    context_file: Path,
    session_id: str,
    new_cursor: int,
    transcript_archive: Path,
    *,
    context: str | None = None,
    attempts: int = 0,
    created_at: float | None = None,
) -> Path:
    """Persist all arguments needed to retry a detached flush."""
    marker = pending_job_path(context_file)
    payload = {
        "context_file": str(context_file),
        "session_id": session_id,
        "new_cursor": new_cursor,
        "transcript_archive": str(transcript_archive),
        "attempts": attempts,
        # Filenames carry a session digest and a random UUID (so untrusted
        # session IDs cannot redirect the write), which makes them useless for
        # ordering. The creation time has to be recorded explicitly.
        "created_at": time.time() if created_at is None else created_at,
    }
    if context is not None:
        payload["context"] = context
    _write_payload_atomic(marker, payload)
    return marker


def record_failed_attempt(context_file: Path) -> int:
    """Count one *confirmed* worker failure against a still-queued job.

    Attempts are counted here, by the worker that actually ran and failed —
    never at hand-out time. A job whose spawn never happened (hook died, the
    process could not be created) must keep its full budget, otherwise the
    durable queue would discard real work that was never once attempted.
    """
    marker = pending_job_path(context_file)
    with _queue_lock(marker.parent):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0
        if not isinstance(payload, dict):
            return 0
        attempts = payload.get("attempts", 0)
        attempts = attempts + 1 if isinstance(attempts, int) and attempts >= 0 else 1
        payload["attempts"] = attempts
        # A worker that failed is definitively not running, so release the
        # spawn lease instead of making the next retry wait it out.
        payload.pop("spawned_at", None)
        try:
            _write_payload_atomic(marker, payload)
        except OSError:
            return 0
        return attempts


def _quarantine(marker: Path, suffix: str, reason: str) -> None:
    """Move an unusable marker aside so it cannot block the whole queue.

    Never deletes. A failed rename leaves the marker exactly where it was: the
    marker is the only record of the session, cursor, and archive path, so
    destroying it on a transient filesystem error would lose the job outright.
    """
    target = marker.with_name(f"{marker.name}.{suffix}")
    try:
        os.replace(marker, target)
    except OSError as exc:
        print(
            f"pending flush {marker.name} could not be quarantined "
            f"({reason}); left in place: {exc}",
            file=sys.stderr,
        )
        return
    print(
        f"pending flush {marker.name} quarantined ({reason}): {target.name}",
        file=sys.stderr,
    )


def load_pending_flushes(state_dir: Path) -> list[tuple[Path, dict]]:
    """Return runnable pending jobs, oldest-first by recorded creation time.

    A record that is *structurally* unusable is moved aside rather than raised.
    Raising would mean one corrupt file stops every future capture for every
    session, which is far worse than setting one job aside.

    An I/O error is deliberately NOT treated as corruption: a marker that is
    briefly unreadable (antivirus, sharing violation, permissions) is left
    untouched for the next run. Only malformed content is quarantined.
    """
    jobs: list[tuple[Path, dict]] = []
    required = {
        "context_file",
        "session_id",
        "new_cursor",
        "transcript_archive",
    }
    now = time.time()
    with _queue_lock(state_dir):
        for marker in sorted(state_dir.glob("*.md.pending.json")):
            try:
                raw = marker.read_text(encoding="utf-8")
            except OSError as exc:
                print(
                    f"pending flush {marker.name} unreadable, left for retry: {exc}",
                    file=sys.stderr,
                )
                continue
            try:
                payload = json.loads(raw)
            except ValueError as exc:
                _quarantine(marker, "corrupt", f"malformed record: {exc}")
                continue
            if not isinstance(payload, dict) or not required.issubset(payload):
                _quarantine(marker, "corrupt", "incomplete record")
                continue

            attempts = payload.get("attempts", 0)
            if not isinstance(attempts, int) or attempts < 0:
                attempts = 0
            payload["attempts"] = attempts
            if attempts >= MAX_FLUSH_ATTEMPTS:
                _quarantine(
                    marker,
                    "exhausted",
                    f"{attempts} failed attempts without success",
                )
                continue

            spawned_at = payload.get("spawned_at")
            if (
                isinstance(spawned_at, (int, float))
                and now - spawned_at < SPAWN_LEASE_SECONDS
            ):
                # Presumed still running from a recent hook. Skipped, never
                # hidden: the lease expires on its own if the worker died.
                continue

            context_file = Path(payload["context_file"])
            if not context_file.exists():
                if "context" not in payload:
                    _quarantine(
                        marker, "corrupt", "context missing and unrecoverable"
                    )
                    continue
                try:
                    _write_context_atomic(context_file, payload["context"])
                except OSError as exc:
                    print(
                        f"pending flush {marker.name} context could not be "
                        f"restored, left for retry: {exc}",
                        file=sys.stderr,
                    )
                    continue

            # Stamp the lease before handing the job out. Only `spawned_at`
            # changes, so a failed write can never consume an attempt or strand
            # the job — at worst a duplicate worker starts.
            payload["spawned_at"] = now
            try:
                _write_payload_atomic(marker, payload)
            except OSError as exc:
                print(
                    f"pending flush {marker.name} lease not recorded: {exc}",
                    file=sys.stderr,
                )

            jobs.append((marker, payload))

    jobs.sort(key=lambda item: (item[1].get("created_at") or 0.0, item[0].name))
    return jobs


def remove_pending_flush(context_file: Path) -> None:
    pending_job_path(context_file).unlink(missing_ok=True)
