"""Cold storage — a reversible, non-destructive article archive.

Mycelium-memory's maintenance marks a duplicate *cold* rather than deleting
it: the article stays on disk and in the index, but drops out of the places
that would let it pollute an answer. Nothing is destroyed, so a wrong call
costs a search miss, never lost knowledge — the one failure this system
cannot recover from (see ``dedup.py``, which for the same reason refuses to
auto-merge). Idea borrowed from mycelium (AGPL); this is our own MIT code.

How "cold" hides an article (both runtime, no reindex needed, instantly
reversible):

* ``compile_truth.py`` skips cold slugs when regenerating ``compiled-truth.md``
  (same line as the contradiction quarantine).
* the knowledge MCP server drops cold hits from ``search_knowledge`` results
  (unless ``include_cold=True``), so the agent stops seeing the retired twin.

Deliberately NOT hidden from internal maintenance: ``dedup`` and the like still
see cold articles, so a cold duplicate can still be compared and, later,
merged by a human. ``get_article`` on a cold slug still works too — cold means
"kept out of the way", not "gone".

Unlike the contradiction quarantine (a bare slug set), cold records carry
provenance — why, when, by what — so a human reviewing the archive can decide
whether to merge-and-delete or restore. Store shape
(``knowledge/cold.json``)::

    {
      "cold": {
        "concepts/foo": {"reason": "near-duplicate of concepts/bar",
                         "at": "2026-10-21T10:00:00+00:00", "by": "maintain"}
      },
      "updated": "2026-10-21T10:00:00+00:00"
    }

``load_cold`` returns just the slug set for the hot exclusion paths;
``load_cold_records`` returns the full records for review/restore tooling.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

try:
    from scripts import config
except ImportError:  # pragma: no cover - exercised only via CLI entrypoint
    import config


def _cold_file(path: Path | None = None) -> Path:
    return path or (config.KNOWLEDGE_DIR / "cold.json")


def load_cold_records(path: Path | None = None) -> dict[str, dict]:
    """Return the full ``{slug: {reason, at, by}}`` map of cold articles.

    Mirrors ``utils.load_contradictions``: a missing file is an empty archive,
    but corruption or a schema violation RAISES rather than silently returning
    empty — a cold archive that vanishes on an I/O hiccup would quietly let
    retired duplicates back into every answer, defeating the feature. Callers
    on the hot search path wrap this in try/except so a corrupt file can never
    break a search; batch callers (compile_truth, maintain) let it raise.
    """
    target = _cold_file(path)
    if not target.exists():
        return {}
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as e:
        raise RuntimeError(
            f"Failed to read cold-storage file at {target}: {e}. "
            f"Fix the I/O issue or delete the file to reset."
        ) from e
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Corrupted cold-storage file at {target}: {e}. "
            f"Fix the JSON manually or delete the file to reset."
        ) from e
    if not isinstance(data, dict):
        raise RuntimeError(
            f"Cold-storage file {target} has wrong schema: expected dict, "
            f"got {type(data).__name__}."
        )
    cold = data.get("cold", {})
    if not isinstance(cold, dict):
        raise RuntimeError(
            f"Cold-storage file {target} has wrong schema: 'cold' must be an "
            f"object, got {type(cold).__name__}."
        )
    if not all(isinstance(k, str) and isinstance(v, dict) for k, v in cold.items()):
        raise RuntimeError(
            f"Cold-storage file {target} has wrong schema: every 'cold' entry "
            f"must map a string slug to an object."
        )
    return cold


def load_cold(path: Path | None = None) -> set[str]:
    """The set of cold slugs — what the exclusion paths actually need."""
    return set(load_cold_records(path).keys())


def _save(records: dict[str, dict], path: Path | None = None) -> None:
    """Persist the cold archive atomically (temp-file + rename)."""
    target = _cold_file(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "cold": {slug: records[slug] for slug in sorted(records)},
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    # PID + per-write uuid so two writers never share a temp path and clobber
    # each other's half-written bytes before the atomic rename. (The unlocked
    # read-modify-write still assumes a single writer — one maintain run at a
    # time — exactly like utils.save_contradictions; concurrent markers can
    # lose a union, but the file can never be left corrupt.)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.{uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(target)


def _audit(action: str, slug: str, detail: str) -> None:
    """Append a one-line audit record to ``knowledge/log.md`` (best-effort).

    Every cold mark/restore is logged so a weekly review can catch a wrong
    call. A logging failure must never abort the operation itself, so this
    swallows I/O errors.
    """
    try:
        now = config.now_iso()
        line = f"\n## [{now}] cold | {action} {slug}"
        if detail:
            line += f" — {detail}"
        line += "\n"
        with config.LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except (OSError, AttributeError):  # pragma: no cover - logging is best-effort
        pass


def mark_cold(
    slug: str,
    reason: str = "",
    *,
    by: str = "maintain",
    now: str | None = None,
    path: Path | None = None,
    audit: bool = True,
) -> bool:
    """Move ``slug`` into cold storage. Idempotent.

    Returns True if the archive changed (a fresh mark), False if ``slug`` was
    already cold (its original record is kept — the first reason stands).
    """
    slug = slug.removesuffix(".md")
    if now is None:
        now = config.now_iso()
    records = load_cold_records(path)
    if slug in records:
        return False
    records[slug] = {"reason": reason, "at": now, "by": by}
    _save(records, path)
    if audit:
        _audit("mark", slug, reason)
    return True


def restore(slug: str, *, path: Path | None = None, audit: bool = True) -> bool:
    """Bring ``slug`` back from cold storage (reverse of ``mark_cold``).

    Returns True if ``slug`` was cold and is now restored, False if it was not
    cold to begin with.
    """
    slug = slug.removesuffix(".md")
    records = load_cold_records(path)
    if slug not in records:
        return False
    del records[slug]
    _save(records, path)
    if audit:
        _audit("restore", slug, "")
    return True


def is_cold(slug: str, *, path: Path | None = None) -> bool:
    """True when ``slug`` is currently in cold storage."""
    return slug.removesuffix(".md") in load_cold(path)
