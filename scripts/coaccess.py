"""Co-access edge store — links that learn from use.

A signal neither the static cross-linker nor the per-article feedback loop
captures: *which articles keep getting retrieved TOGETHER in answers that
worked?* Mycelium-memory's "connections form through co-access, unused paths
decay" idea — rebuilt here as our own pure-Python store (their code is AGPL,
this is MIT: the idea is borrowed, not a line of code).

Where the siblings sit:

* ``crosslink.py`` — edges from STATIC prose mentions (one article names
  another's title). Build-time text analysis.
* ``retrieval_feedback.py`` — a PER-ARTICLE usefulness score.
* ``coaccess.py`` (this) — a PAIRWISE edge whose weight rises each time two
  articles are retrieved together in a useful answer and decays with age.

Fed into ``unified_graph`` as ``kind="co_access"`` edges, so ``salience``,
``communities`` and ``linked_search`` all benefit without further wiring.

Design notes (mirrors ``retrieval_feedback`` deliberately — same guarantees):

* **Additive, backward-compatible.** Absent store ⇒ no co-access edges ⇒
  the graph is exactly what it is today. Only recorded pairs add edges.
* **Recency-weighted.** Each co-access event carries a timestamp; older
  events decay with a 45-day half-life, so a pair that stopped co-occurring
  months ago fades below the edge floor and drops out of the graph.
* **Zero LLM cost.** Pure file I/O and arithmetic.
* **Atomic, loss-proof writes.** Temp-file + rename; a read-modify-write that
  refuses to clobber an unreadable store and sidelines a corrupt one for
  forensics — identical to the feedback store, because the same data-loss
  bug (2026-07-31 state.json truncation) must not recur here.
* **Bounded.** Per-pair events capped; the pair table capped by evicting the
  lowest-activity pairs, so the file can't grow without limit.

Store shape (``knowledge/coaccess.json``)::

    {
      "version": 1,
      "updated": "2026-10-21T10:00:00+02:00",
      "pairs": {
        "concepts/a|concepts/b": {"count": 3, "events": [{"at": "..."}, ...]}
      }
    }

A pair key is the two slugs sorted and joined with ``|`` (``PAIR_SEP``), which
cannot occur inside a slug (slugs are ``subdir/lower-hyphen-stem``), so the key
round-trips unambiguously. ``count`` is the lifetime total and is never
trimmed; ``events`` is capped at ``MAX_EVENTS_PER_PAIR`` (newest kept).
"""
from __future__ import annotations

import json
import os
from collections import deque
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

PAIR_SEP = "|"

MAX_EVENTS_PER_PAIR = 50
MAX_PAIRS = 50_000
COACCESS_HALF_LIFE_DAYS = 45

# An edge is emitted into the graph only when its decayed weight is at least
# this. One fresh co-access weighs 1.0; after ~one half-life it is ~0.5, so a
# lone, aging pair fades out while a repeatedly co-accessed pair stays well
# above the floor. Callers may override.
DEFAULT_MIN_EDGE_WEIGHT = 0.5


def _store_file(path: Path | None = None) -> Path:
    if path is not None:
        return path
    from config import KNOWLEDGE_DIR

    return KNOWLEDGE_DIR / "coaccess.json"


def _empty() -> dict:
    return {"version": 1, "pairs": {}}


def load(path: Path | None = None) -> dict:
    """Load the co-access store. Missing / corrupt file returns an empty store.

    Degrades gracefully on corruption (like the feedback loader, unlike the
    quarantine loader): co-access is an enhancement signal, and a single bad
    byte must never take down ``unified_graph`` or the MCP server.
    """
    target = _store_file(path)
    if not target.exists():
        return _empty()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty()
    if not isinstance(data, dict) or not isinstance(data.get("pairs"), dict):
        return _empty()
    return data


def _save(data: dict, path: Path | None = None) -> None:
    target = _store_file(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data["updated"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    # PID + per-write uuid so two writers — even two threads in ONE process, as
    # the async MCP server can produce — never share a temp path and clobber
    # each other's half-written bytes before the atomic rename.
    tmp = target.with_name(f"{target.name}.{os.getpid()}.{uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(target)


def _read_for_write(target: Path) -> dict:
    """Read for a read-modify-write, protecting existing data.

    * **Absent** → fresh empty store.
    * **Present but transiently unreadable** (``OSError``) → raise, so the
      write aborts instead of overwriting real data with one event.
    * **Present but corrupt** → move aside to ``<name>.corrupt-<pid>`` and
      start fresh, preserving the damaged bytes for forensics.
    """
    if not target.exists():
        return _empty()
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as e:
        raise RuntimeError(
            f"Co-access store {target} exists but is currently unreadable ({e}); "
            f"refusing to overwrite it to avoid destroying accumulated edges. "
            f"Retry once the file is free."
        ) from e
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get("pairs"), dict):
            raise ValueError("wrong schema")
    except (json.JSONDecodeError, ValueError):
        backup = target.with_name(f"{target.name}.corrupt-{os.getpid()}")
        try:
            target.replace(backup)
        except OSError:
            pass
        return _empty()
    return data


def _norm_slug(slug: str) -> str:
    """Normalize one slug: strip ``.md``, trim. Reject an empty or piped slug."""
    if not slug or not isinstance(slug, str):
        raise ValueError(f"slug must be a non-empty string, got {slug!r}")
    slug = slug.strip().removesuffix(".md")
    if not slug:
        raise ValueError("slug must be non-empty after normalization")
    if PAIR_SEP in slug:
        raise ValueError(f"slug must not contain {PAIR_SEP!r}: {slug!r}")
    return slug


def _pair_key(a: str, b: str) -> str:
    """Canonical key for an unordered pair: the two slugs sorted, ``|``-joined."""
    lo, hi = sorted((a, b))
    return f"{lo}{PAIR_SEP}{hi}"


def _split_key(key: str) -> tuple[str, str]:
    a, _, b = key.partition(PAIR_SEP)
    return a, b


def _unordered_pairs(slugs: list[str]) -> list[tuple[str, str]]:
    """Every unordered, distinct pair among ``slugs`` (deduped, self-pairs out)."""
    uniq: list[str] = []
    seen: set[str] = set()
    for s in slugs:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    pairs: list[tuple[str, str]] = []
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            pairs.append((uniq[i], uniq[j]))
    return pairs


def _evict_if_needed(pairs: dict) -> None:
    """Keep the pair table under ``MAX_PAIRS`` by dropping least-active pairs.

    Eviction is by event count ascending (a cheap, today-independent proxy for
    weight); ties break on the key for determinism. Mutates ``pairs`` in place.
    """
    if len(pairs) <= MAX_PAIRS:
        return
    ranked = sorted(
        pairs.items(),
        key=lambda kv: (len(kv[1].get("events", [])), kv[0]),
    )
    for key, _ in ranked[: len(pairs) - MAX_PAIRS]:
        del pairs[key]


def record_coaccess(
    slugs: list[str],
    *,
    now: str | None = None,
    path: Path | None = None,
) -> int:
    """Record that every article in ``slugs`` was retrieved together, once.

    Call this only for a result set that proved *useful* — co-access is a
    positive signal, so wiring it to a ``useful`` outcome (not every search)
    keeps noise out of the graph. A set of fewer than two distinct slugs is a
    no-op (no pair to strengthen).

    Args:
        slugs: article slugs (``.md`` suffix tolerated) retrieved together.
        now: ISO timestamp override (tests pass a fixed value).
        path: store-file override (tests pass a tmp path).

    Returns the number of pairs strengthened. Raises ``ValueError`` if any
    slug is malformed, so a bad caller surfaces instead of recording noise.
    """
    normalized = [_norm_slug(s) for s in slugs]
    pairs_to_bump = _unordered_pairs(normalized)
    if not pairs_to_bump:
        return 0

    if now is None:
        from config import now_iso

        now = now_iso()

    data = _read_for_write(_store_file(path))
    pairs = data.setdefault("pairs", {})
    _bump_pairs(pairs, pairs_to_bump, now)
    _evict_if_needed(pairs)
    _save(data, path)
    return len(pairs_to_bump)


def _bump_pairs(pairs: dict, pair_list: list[tuple[str, str]], now: str) -> None:
    """Increment count + append a dated event for each ``(a, b)`` pair in place."""
    for a, b in pair_list:
        key = _pair_key(a, b)
        rec = pairs.setdefault(key, {"count": 0, "events": []})
        rec["count"] = int(rec.get("count", 0)) + 1
        events = rec.setdefault("events", [])
        events.append({"at": now})
        if len(events) > MAX_EVENTS_PER_PAIR:
            del events[: len(events) - MAX_EVENTS_PER_PAIR]


def _record_pairs(
    pair_list: list[tuple[str, str]],
    *,
    now: str | None = None,
    path: Path | None = None,
) -> int:
    """Record an explicit list of already-normalized ``(a, b)`` pairs in ONE write.

    Unlike ``record_coaccess`` (which forms the full clique of a slug list),
    this bumps exactly the pairs given — the tracker uses it to persist only
    the NEW edges from one useful mark in a single read-modify-write, so a
    result set where ``k`` articles prove useful costs ``k`` writes, not
    ``C(k, 2)``. Self-pairs are dropped. Returns the number recorded.
    """
    clean = [(a, b) for a, b in pair_list if a != b]
    if not clean:
        return 0
    if now is None:
        from config import now_iso

        now = now_iso()
    data = _read_for_write(_store_file(path))
    pairs = data.setdefault("pairs", {})
    _bump_pairs(pairs, clean, now)
    _evict_if_needed(pairs)
    _save(data, path)
    return len(clean)


def _pair_weight(events: list[dict], today: date) -> float:
    """Recency-weighted edge weight: a decayed sum over a pair's events.

    Each event contributes ``0.5 ** (age_days / half_life)``. Monotonic in
    corroboration (more / newer co-accesses ⇒ heavier edge), unlike the
    feedback score this is NOT squashed to ``[0, 1]`` — it is a raw edge weight
    the graph thresholds against ``MIN_EDGE_WEIGHT``. Zero when there are no
    usable events.
    """
    total = 0.0
    for ev in events:
        at = ev.get("at")
        if not at:
            total += 1.0  # untimestamped (legacy / hand-authored) counts undecayed
            continue
        try:
            # Convert to the local wall-clock date before comparing with
            # ``today`` (also local): two strings for the SAME instant in
            # different offsets must land on the same date, or they'd straddle
            # the weight floor. ``astimezone()`` normalizes aware timestamps and
            # localizes naive ones.
            ev_date = datetime.fromisoformat(at).astimezone().date()
        except (ValueError, TypeError):
            # A malformed timestamp must not anchor the edge at full weight
            # forever — skip the event entirely.
            continue
        age = max(0, (today - ev_date).days)
        total += 0.5 ** (age / COACCESS_HALF_LIFE_DAYS)
    return total


def weight_for(
    slug_a: str,
    slug_b: str,
    *,
    today: date | None = None,
    path: Path | None = None,
) -> float:
    """Decayed co-access weight between two slugs (0.0 when never co-accessed)."""
    a = _norm_slug(slug_a)
    b = _norm_slug(slug_b)
    if a == b:
        return 0.0
    data = load(path)
    rec = data.get("pairs", {}).get(_pair_key(a, b))
    if not rec:
        return 0.0
    return _pair_weight(rec.get("events", []), today or date.today())


def load_edges(
    *,
    today: date | None = None,
    path: Path | None = None,
    min_weight: float = DEFAULT_MIN_EDGE_WEIGHT,
) -> list[dict]:
    """Co-access edges whose decayed weight is at least ``min_weight``.

    Returns a list of ``{"from", "to", "kind": "co_access", "weight"}`` dicts,
    heaviest first. ``from``/``to`` are the sorted slug pair (the edge is
    undirected; a stable order keeps the output deterministic). Consumed by
    ``unified_graph.build`` — slugs are mapped to ``article:<slug>`` node ids
    there, and an edge is kept only when BOTH endpoints exist as nodes.
    """
    today = today or date.today()
    data = load(path)
    edges: list[dict] = []
    for key, rec in data.get("pairs", {}).items():
        weight = _pair_weight(rec.get("events", []), today)
        if weight < min_weight:
            continue
        a, b = _split_key(key)
        if not a or not b:
            continue  # malformed key — skip rather than emit a half-edge
        edges.append({"from": a, "to": b, "kind": "co_access", "weight": weight})
    edges.sort(key=lambda e: (-e["weight"], e["from"], e["to"]))
    return edges


class CoAccessTracker:
    """In-memory bridge from "retrieved together" to "useful together".

    The signal we actually want is not "these articles were returned by one
    search" (that happens to every result, useful or not) but "two articles
    from one search BOTH proved useful". That needs two facts the store itself
    doesn't hold: which slugs came back from a given search, and which of those
    the agent later rated ``useful``. This tracker holds exactly that, for the
    life of the server process, and calls ``record_coaccess`` only when a
    second member of a remembered result set is marked useful.

    Best-effort by design: state is per-process. A ``useful`` mark with no
    remembered set (server restarted, or the search ran in another process) is
    a silent no-op — it costs a missed edge, never a crash or a wrong edge.
    Bounded: only the newest ``max_sets`` result sets are kept.
    """

    def __init__(self, *, max_sets: int = 64, path: Path | None = None) -> None:
        # Each entry: {"slugs": frozenset[str], "useful": set[str]}.
        self._sets: deque[dict] = deque(maxlen=max_sets)
        self._path = path

    @staticmethod
    def _safe_norm(slug: str) -> str | None:
        """Normalize a slug, returning ``None`` instead of raising.

        The server call sites must never crash a search or a rating because one
        slug was odd (e.g. a foreign linked-project slug carrying a ``|``).
        """
        try:
            return _norm_slug(slug)
        except ValueError:
            return None

    def remember(self, slugs: list[str]) -> None:
        """Record the slugs one search returned. Sets of <2 are ignored."""
        normalized = {s for s in (self._safe_norm(x) for x in slugs) if s}
        if len(normalized) < 2:
            return
        self._sets.append({"slugs": frozenset(normalized), "useful": set()})

    def mark_useful(self, slug: str, *, now: str | None = None) -> int:
        """Mark ``slug`` useful; link it to each set's already-useful members.

        Visits EVERY remembered set containing ``slug`` — not just the newest —
        so interleaved searches don't lose a pair (rate A useful for search 1
        and search 2, then B useful for search 1: the A–B edge from search 1 is
        still recorded). Within each set, the first time ``slug`` is marked
        useful a co-access pair is recorded between it and every slug already
        useful in that set, so a set where ``k`` articles prove useful
        accumulates exactly ``C(k, 2)`` pairs, each once.

        Per set the new pairs are written in ONE call (``_record_pairs``), and
        the set is marked useful only AFTER that write succeeds — so a write
        that raises leaves the slug unmarked and a retry records it instead of
        silently dropping it. Returns the total pairs recorded by this call.
        """
        norm = self._safe_norm(slug)
        if norm is None:
            return 0
        recorded = 0
        for entry in self._sets:
            if norm not in entry["slugs"] or norm in entry["useful"]:
                continue
            prior = list(entry["useful"])
            if prior:
                _record_pairs(
                    [(norm, other) for other in prior], now=now, path=self._path
                )
                recorded += len(prior)
            entry["useful"].add(norm)  # only after a successful write
        return recorded


def prune(
    *,
    today: date | None = None,
    path: Path | None = None,
    min_weight: float = DEFAULT_MIN_EDGE_WEIGHT,
) -> int:
    """Drop pairs whose decayed weight has fallen below ``min_weight``.

    Maintenance helper (safe to never call — ``load_edges`` already filters at
    read time). Returns the number of pairs removed. A pruned pair referencing
    a deleted article simply stops being a weak edge; nothing else depends on
    it, so pruning is non-destructive to the knowledge itself.
    """
    today = today or date.today()
    data = _read_for_write(_store_file(path))
    pairs = data.get("pairs", {})
    doomed = [
        key for key, rec in pairs.items()
        if _pair_weight(rec.get("events", []), today) < min_weight
    ]
    for key in doomed:
        del pairs[key]
    if doomed:
        _save(data, path)
    return len(doomed)
