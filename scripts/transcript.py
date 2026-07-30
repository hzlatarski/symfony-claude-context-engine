"""Shared transcript reader for the flush hooks.

``session-end.py`` and ``pre-compact.py`` both need to turn a Claude Code JSONL
transcript into a markdown context blob for flush.py. They used to carry
byte-identical copies of this function, which is how one of them could grow a
turn cursor while the other kept re-reading from turn zero. One copy, here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

from config import KNOWLEDGE_DIR
from filelock import FileLock

DEFAULT_MAX_TURNS: int | None = None
DEFAULT_MAX_CONTEXT_CHARS: int | None = None
TRANSCRIPT_CHUNK_CHARS = 6_000
TRANSCRIPT_CHUNK_OVERLAP = 256
_EXACT_REFERENCE_RE = re.compile(
    r"https?://[^\s<>\"']+|[A-Za-z0-9][A-Za-z0-9_.:/?&=%+-]{31,}"
)


def _conversation_turns(transcript_path: Path) -> list[str]:
    """Return user/assistant text turns used by the summary pipeline."""
    turns: list[str] = []

    with transcript_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = entry.get("message", {})
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
            else:
                role = entry.get("role", "")
                content = entry.get("content", "")

            if role not in ("user", "assistant"):
                continue

            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "\n".join(text_parts)

            if isinstance(content, str) and content.strip():
                label = "User" if role == "user" else "Assistant"
                turns.append(f"**{label}:** {content.strip()}\n")

    return turns


def extract_conversation_context(
    transcript_path: Path,
    start_turn: int = 0,
    max_turns: int | None = DEFAULT_MAX_TURNS,
    max_context_chars: int | None = DEFAULT_MAX_CONTEXT_CHARS,
) -> tuple[str, int, int]:
    """Extract unflushed turns without marking omitted content as consumed.

    The default has no turn or character limit: the complete fresh transcript
    reaches the summary model. Optional bounds are still available to callers
    that deliberately batch work, but batches are oldest-first and the returned
    cursor advances only through the complete turns present in ``context``.

    Returns ``(context, next_cursor, window_turns)``.
    """
    turns = _conversation_turns(transcript_path)
    total_turns = len(turns)

    start = max(0, min(start_turn, total_turns))
    fresh = turns[start:]
    selected = fresh if max_turns is None else fresh[:max(0, max_turns)]

    if max_context_chars is not None and selected:
        # The ceiling is checked before a turn is admitted, including the first
        # one. Admitting an oversized first turn would return more than the
        # caller asked for AND advance the cursor past it, which is the same
        # class of silent over-consumption this module exists to prevent.
        bounded: list[str] = []
        used = 0
        for turn in selected:
            separator_chars = 1 if bounded else 0
            if used + separator_chars + len(turn) > max_context_chars:
                break
            bounded.append(turn)
            used += separator_chars + len(turn)
        selected = bounded

    context = "\n".join(selected)
    next_cursor = start + len(selected)
    return context, next_cursor, len(selected)


def _fsync_dir(directory: Path) -> None:
    """Best-effort fsync of a directory so a rename is durably published.

    On POSIX the file's own fsync does not guarantee the new directory entry
    survives a power loss. Windows has no directory-fsync equivalent and
    raises, so this degrades to a no-op there rather than failing the archive.
    """
    try:
        fd = os.open(directory, getattr(os, "O_DIRECTORY", os.O_RDONLY))
    except (OSError, AttributeError, ValueError):
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _archive_stem(session_id: str) -> str:
    """Return a collision-free archive filename stem for a session ID.

    Claude's own session IDs are UUIDs, which pass through unchanged so the
    existing archives keep their filenames. Anything else is untrusted hook
    input: blind character replacement would map two distinct IDs onto one
    archive, where the second session is then rejected as a divergent snapshot
    and can never be archived. A digest suffix keeps such IDs distinct.
    """
    if _SAFE_SESSION_ID.fullmatch(session_id):
        return session_id
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:20]
    readable = re.sub(r"[^A-Za-z0-9_-]", "_", session_id)[:40]
    return f"{readable}-{digest}"


def archive_transcript(
    transcript_path: Path,
    session_id: str,
    archive_dir: Path | None = None,
) -> Path:
    """Copy the original JSONL byte-for-byte to the searchable archive.

    The same session path is replaced atomically as a live transcript grows, so
    PreCompact and SessionEnd can both call this safely without duplicate data.
    """
    target_dir = archive_dir or KNOWLEDGE_DIR / "daily" / "transcripts"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{_archive_stem(session_id)}.jsonl"
    tmp = target.with_suffix(f".jsonl.tmp{os.getpid()}")
    lock = FileLock(str(target.with_suffix(".jsonl.lock")))
    with lock:
        source_bytes = transcript_path.read_bytes()
        if target.exists():
            archived_bytes = target.read_bytes()
            if source_bytes == archived_bytes or archived_bytes.startswith(source_bytes):
                return target
            if not source_bytes.startswith(archived_bytes):
                raise ValueError(
                    f"Transcript snapshot for {session_id!r} diverges from its archive"
                )
        try:
            # The archive is the authoritative copy the cursor is allowed to
            # advance past, so it must reach stable storage before the rename
            # publishes it — the same guarantee the daily log and the cursor
            # already had. Without the fsync a power loss can leave a cursor
            # that consumed turns the archive no longer contains.
            with tmp.open("wb") as handle:
                handle.write(source_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            tmp.replace(target)
            _fsync_dir(target_dir)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
    return target


def _transcript_date(lines: list[str], transcript_path: Path) -> str:
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        timestamp = entry.get("timestamp") if isinstance(entry, dict) else None
        if isinstance(timestamp, str) and re.match(r"^\d{4}-\d{2}-\d{2}", timestamp):
            return timestamp[:10]
    return datetime.fromtimestamp(transcript_path.stat().st_mtime).astimezone().strftime(
        "%Y-%m-%d"
    )


def _record_date(line: str, fallback: str) -> str:
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        return fallback
    timestamp = entry.get("timestamp") if isinstance(entry, dict) else None
    if isinstance(timestamp, str) and re.match(r"^\d{4}-\d{2}-\d{2}", timestamp):
        return timestamp[:10]
    return fallback


def _overlapping_chunks(text: str) -> list[str]:
    if len(text) <= TRANSCRIPT_CHUNK_CHARS:
        return [text]
    step = TRANSCRIPT_CHUNK_CHARS - TRANSCRIPT_CHUNK_OVERLAP
    return [
        text[start : start + TRANSCRIPT_CHUNK_CHARS]
        for start in range(0, len(text), step)
    ]


def _exact_references(text: str) -> list[str]:
    return list(dict.fromkeys(match.group(0) for match in _EXACT_REFERENCE_RE.finditer(text)))


def embed_transcript_file(transcript_path: Path) -> int:
    """Index every raw JSONL record, including tool inputs and tool results."""
    from vector_store import replace_chunks_for_source

    index_lock = FileLock(str(transcript_path.with_suffix(".jsonl.index.lock")))
    with index_lock:
        rel = str(transcript_path.relative_to(KNOWLEDGE_DIR)).replace("\\", "/")
        lines = transcript_path.read_text(encoding="utf-8").splitlines()
        fallback_date = _transcript_date(lines, transcript_path)

        chunks: list[dict] = []
        seen_references: set[str] = set()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            date_str = _record_date(line, fallback_date)
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                entry = None
            message = entry.get("message") if isinstance(entry, dict) else None
            role = message.get("role", "event") if isinstance(message, dict) else "event"
            for part_number, text in enumerate(_overlapping_chunks(line), start=1):
                chunks.append({
                    "chunk_id": (
                        f"transcript-{transcript_path.stem}-"
                        f"{line_number:06d}-{part_number:03d}"
                    ),
                    "source_file": rel,
                    "text": text,
                    "metadata": {
                        "section": f"{role} record {line_number}",
                        "date": date_str,
                        "session_id": transcript_path.stem,
                    },
                })
            for reference_number, reference in enumerate(
                _exact_references(line),
                start=1,
            ):
                if reference in seen_references:
                    continue
                seen_references.add(reference)
                chunks.append({
                    "chunk_id": (
                        f"transcript-{transcript_path.stem}-"
                        f"{line_number:06d}-ref-{reference_number:03d}"
                    ),
                    "source_file": rel,
                    "text": reference,
                    "metadata": {
                        "section": f"exact reference record {line_number}",
                        "date": date_str,
                        "session_id": transcript_path.stem,
                    },
                })
        replace_chunks_for_source(rel, chunks)
        return len(chunks)
