"""Retrieval-outcome feedback loop.

A signal the time-based confidence decay can't capture: *was an article
actually useful when it was retrieved?* Graphify's ``save-result`` /
``reflect`` pattern — an agent marks a retrieval ``useful`` / ``dead_end``
/ ``corrected``, those outcomes are recency-weighted, and they feed back
into how the article ranks and whether it's flagged for review.

Design notes:

* **Additive, backward-compatible.** An article with no feedback scores a
  neutral ``0.5`` — so introducing this weight leaves the relative ranking
  of never-rated articles untouched. Only rated articles move.
* **Recency-weighted.** Each outcome event carries a timestamp; older
  events decay with a 45-day half-life so a stale "dead_end" from months
  ago doesn't permanently bury an article that has since been fixed.
* **Zero LLM cost.** Pure file I/O and arithmetic, like ``compile_truth``.
* **Atomic writes.** Temp-file + rename (mirrors ``utils.save_contradictions``)
  so a crash mid-write can't corrupt the store.

Store shape (``knowledge/retrieval-feedback.json``)::

    {
      "version": 1,
      "updated": "2026-07-11T10:00:00+02:00",
      "outcomes": {
        "concepts/foo": {
          "useful": 3, "dead_end": 0, "corrected": 1,
          "events": [{"outcome": "useful", "at": "...", "note": "..."}, ...]
        }
      }
    }

``events`` is capped at ``MAX_EVENTS_PER_SLUG`` (newest kept) so the file
can't grow without bound; the rolling ``useful`` / ``dead_end`` /
``corrected`` counts are lifetime totals and are never trimmed.
"""
from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path

# Valid outcomes and the raw value each contributes to the net score.
# ``corrected`` (the retrieved claim was wrong and had to be fixed) is a
# stronger negative than ``dead_end`` (merely unhelpful) — a wrong article
# is more harmful than a useless one.
OUTCOME_VALUES = {
    "useful": 1.0,
    "dead_end": -1.0,
    "corrected": -2.0,
}
OUTCOMES = frozenset(OUTCOME_VALUES)

MAX_EVENTS_PER_SLUG = 50
FEEDBACK_HALF_LIFE_DAYS = 45
# Logistic steepness: net of +/-``SQUASH_K`` maps to ~0.73 / ~0.27.
SQUASH_K = 2.0
NEUTRAL_SCORE = 0.5


def _feedback_file(path: Path | None = None) -> Path:
    if path is not None:
        return path
    from config import KNOWLEDGE_DIR

    return KNOWLEDGE_DIR / "retrieval-feedback.json"


def load(path: Path | None = None) -> dict:
    """Load the feedback store. Missing / corrupt file returns an empty store.

    Unlike the quarantine loader we deliberately degrade gracefully on
    corruption rather than raise: feedback is an enhancement signal, and a
    single bad byte must never take down ``compile_truth`` or the MCP
    server. A corrupt file is treated as "no feedback yet".
    """
    target = _feedback_file(path)
    if not target.exists():
        return {"version": 1, "outcomes": {}}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "outcomes": {}}
    if not isinstance(data, dict) or not isinstance(data.get("outcomes"), dict):
        return {"version": 1, "outcomes": {}}
    return data


def _save(data: dict, path: Path | None = None) -> None:
    target = _feedback_file(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data["updated"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    # PID-suffixed tmp name so two concurrent writers don't collide on the
    # same temp path (the RMW race is still possible — see _read_for_write —
    # but at least the atomic-rename step won't clobber a sibling's tmp).
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(target)


def _read_for_write(target: Path) -> dict:
    """Read the store for a read-modify-write cycle, protecting existing data.

    Unlike ``load`` (which is for read-only consumers and degrades any
    failure to an empty store), this distinguishes three cases so a write
    can never silently destroy accumulated feedback:

    * **Absent** → fresh empty store (normal first-write path).
    * **Present but transiently unreadable** (``OSError`` — e.g. a Windows
      sharing violation while another process holds the handle) → raise, so
      ``record`` aborts instead of overwriting real data with one event.
    * **Present but corrupt** (bad JSON / wrong schema) → move the bad file
      aside to ``<name>.corrupt-<pid>`` and start fresh, preserving the
      damaged data for forensics rather than deleting it.
    """
    if not target.exists():
        return {"version": 1, "outcomes": {}}
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as e:
        raise RuntimeError(
            f"Feedback store {target} exists but is currently unreadable ({e}); "
            f"refusing to overwrite it to avoid destroying accumulated feedback. "
            f"Retry once the file is free."
        ) from e
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get("outcomes"), dict):
            raise ValueError("wrong schema")
    except (json.JSONDecodeError, ValueError):
        backup = target.with_name(f"{target.name}.corrupt-{os.getpid()}")
        try:
            target.replace(backup)
        except OSError:
            pass
        return {"version": 1, "outcomes": {}}
    return data


def record(
    slug: str,
    outcome: str,
    note: str = "",
    *,
    now: str | None = None,
    path: Path | None = None,
) -> dict:
    """Record one retrieval outcome for ``slug``.

    Args:
        slug: article slug without ``.md`` (e.g. ``concepts/foo``).
        outcome: one of ``useful`` | ``dead_end`` | ``corrected``.
        note: optional free-text context (what was wrong / how it helped).
        now: ISO timestamp override (tests pass a fixed value).
        path: store-file override (tests pass a tmp path).

    Returns the updated per-slug record. Raises ``ValueError`` on an
    unknown outcome so a typo surfaces instead of silently recording noise.
    """
    if outcome not in OUTCOMES:
        raise ValueError(
            f"outcome must be one of {sorted(OUTCOMES)}, got {outcome!r}"
        )
    if not slug or not isinstance(slug, str):
        raise ValueError(f"slug must be a non-empty string, got {slug!r}")

    slug = slug.removesuffix(".md")
    if now is None:
        from config import now_iso

        now = now_iso()

    data = _read_for_write(_feedback_file(path))
    outcomes = data.setdefault("outcomes", {})
    rec = outcomes.setdefault(
        slug, {"useful": 0, "dead_end": 0, "corrected": 0, "events": []}
    )
    rec[outcome] = int(rec.get(outcome, 0)) + 1
    events = rec.setdefault("events", [])
    events.append({"outcome": outcome, "at": now, "note": note[:280]})
    # Keep only the newest MAX_EVENTS_PER_SLUG events (lifetime counts above
    # are preserved regardless of trimming).
    if len(events) > MAX_EVENTS_PER_SLUG:
        del events[: len(events) - MAX_EVENTS_PER_SLUG]

    _save(data, path)
    return rec


def _event_score(events: list[dict], today: date) -> float:
    """Recency-weighted net over a slug's events, squashed to ``[0, 1]``.

    Neutral ``0.5`` when there are no events. Newer events dominate via a
    45-day exponential half-life.
    """
    if not events:
        return NEUTRAL_SCORE

    net = 0.0
    total_weight = 0.0
    for ev in events:
        value = OUTCOME_VALUES.get(ev.get("outcome"))
        if value is None:
            continue
        at = ev.get("at")
        if not at:
            # No timestamp — count it undecayed (legacy / hand-authored event).
            weight = 1.0
        else:
            try:
                ev_date = datetime.fromisoformat(at).date()
            except (ValueError, TypeError):
                # A malformed timestamp must not anchor the score at full
                # weight forever — skip the event entirely.
                continue
            age = max(0, (today - ev_date).days)
            weight = 0.5 ** (age / FEEDBACK_HALF_LIFE_DAYS)
        net += weight * value
        total_weight += weight

    if total_weight == 0.0:
        return NEUTRAL_SCORE
    return 1.0 / (1.0 + math.exp(-net / SQUASH_K))


def score_for(slug: str, *, today: date | None = None, path: Path | None = None) -> float:
    """Feedback score for one slug in ``[0, 1]`` (0.5 = neutral / no data)."""
    data = load(path)
    rec = data.get("outcomes", {}).get(slug.removesuffix(".md"))
    if not rec:
        return NEUTRAL_SCORE
    return _event_score(rec.get("events", []), today or date.today())


def load_scores(*, today: date | None = None, path: Path | None = None) -> dict[str, float]:
    """Return ``{slug: score}`` for every slug with recorded feedback.

    Slugs absent from the map have no feedback and should be treated as
    ``NEUTRAL_SCORE`` by callers — computing them here would just return a
    constant for thousands of articles.
    """
    today = today or date.today()
    data = load(path)
    return {
        slug: _event_score(rec.get("events", []), today)
        for slug, rec in data.get("outcomes", {}).items()
    }


def aggregate(*, today: date | None = None, path: Path | None = None) -> list[dict]:
    """Per-slug summary sorted by score ascending (most-contested first).

    Each entry: ``{slug, score, useful, dead_end, corrected, total,
    last_at, last_note}``. Consumed by ``reflect.py`` and ``kb_health``.
    """
    today = today or date.today()
    data = load(path)
    out: list[dict] = []
    for slug, rec in data.get("outcomes", {}).items():
        events = rec.get("events", [])
        last = events[-1] if events else {}
        useful = int(rec.get("useful", 0))
        dead_end = int(rec.get("dead_end", 0))
        corrected = int(rec.get("corrected", 0))
        out.append({
            "slug": slug,
            "score": _event_score(events, today),
            "useful": useful,
            "dead_end": dead_end,
            "corrected": corrected,
            "total": useful + dead_end + corrected,
            "last_at": last.get("at"),
            "last_note": last.get("note", ""),
        })
    out.sort(key=lambda r: (r["score"], -r["total"], r["slug"]))
    return out
