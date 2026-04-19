"""Three-way parallel retrieval + RRF merge for the whisper pipeline.

Given N queries and a scope list, fan out to the in-process search
impls from knowledge_mcp_server, merge results per-channel via
Reciprocal Rank Fusion, then interleave across channels and convert
to the whisper Hit dataclass with stable citation IDs (c1, c2, ...).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from knowledge_mcp_server import (
    _search_knowledge_impl,
    _search_raw_daily_impl,
)
from whisper.types import Hit

logger = logging.getLogger(__name__)

VALID_SCOPES = {"articles", "code", "daily"}
RRF_K = 60  # Textbook RRF constant; same used by hybrid_search
TOP_N_DEFAULT = 12
PER_QUERY_LIMIT = 5


def _search_codebase_impl(query: str, limit: int = PER_QUERY_LIMIT) -> list[dict[str, Any]]:
    """Stub for codebase search — wired up in a later task.

    Kept as a module-level name so tests can monkeypatch it and so the
    downstream retrieve() call flow is identical to the article/daily
    channels. Returns an empty list until a real code-search backend
    (e.g. a Tree-sitter + BM25 index) is plugged in.
    """
    return []


def rrf_merge(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over multiple ranked lists.

    Each input list is [(key, score), ...] ordered best-first. The
    fused score for a key is sum(1 / (k + rank)) across lists that
    contain it. Items missing from a list contribute nothing.
    Returns a single list sorted by fused score, descending.
    """
    fused: dict[str, float] = defaultdict(float)
    for lst in ranked_lists:
        for rank, (key, _score) in enumerate(lst):
            fused[key] += 1.0 / (k + rank + 1)  # +1 so rank 0 is treated as rank 1
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def _run_knowledge(query: str) -> list[dict[str, Any]]:
    return _search_knowledge_impl(query=query, limit=PER_QUERY_LIMIT)


def _run_codebase(query: str) -> list[dict[str, Any]]:
    return _search_codebase_impl(query=query, limit=PER_QUERY_LIMIT)


def _run_daily(query: str) -> list[dict[str, Any]]:
    return _search_raw_daily_impl(query=query, limit=PER_QUERY_LIMIT)


_CHANNEL_RUNNERS = {
    "articles": _run_knowledge,
    "code": _run_codebase,
    "daily": _run_daily,
}


def _hit_key(channel: str, raw: dict[str, Any]) -> str:
    """Stable identity for a hit, so RRF can de-dupe across queries."""
    if channel == "code":
        return f"code::{raw.get('path', '')}"
    return f"{channel}::{raw.get('slug', '')}"


def _to_hit(channel: str, raw: dict[str, Any], cid: str, score: float) -> Hit:
    if channel == "code":
        return Hit(
            id=cid,
            source="code",
            category=None,
            path=raw.get("path", ""),
            title=(raw.get("symbols") or ["?"])[0] if raw.get("symbols") else raw.get("path", ""),
            snippet=raw.get("preview", ""),
            full_body=None,
            score=score,
            symbols=list(raw.get("symbols") or []),
            metadata={},
        )
    # article or daily — both come from _slim_hit so have {slug, title, snippet, distance, metadata}
    meta = dict(raw.get("metadata") or {})
    return Hit(
        id=cid,
        source="article" if channel == "articles" else "daily",
        category=meta.get("category") if channel == "articles" else None,
        path=raw.get("slug", ""),
        title=raw.get("title") or raw.get("slug", ""),
        snippet=raw.get("snippet", ""),
        full_body=None,
        score=score,
        symbols=[],
        metadata=meta,
    )


def retrieve(queries: list[str], scope: list[str], top_n: int = TOP_N_DEFAULT) -> list[Hit]:
    """Run queries against every channel in scope, merge via RRF, return top_n Hits.

    Args:
        queries: already-expanded retrieval queries (typically 3-5).
        scope: subset of {"articles", "code", "daily"}.
        top_n: maximum number of hits to return after fusion.

    Returns:
        List of Hit with citation ids "c1", "c2", ...
    """
    if not queries:
        return []

    channels = [c for c in scope if c in VALID_SCOPES]
    if not channels:
        return []

    # Resolve runners lazily so monkeypatched module-level names are picked up.
    runners = {
        "articles": _run_knowledge,
        "code": _run_codebase,
        "daily": _run_daily,
    }

    # Fan out: (channel, query) → raw hits, in parallel via threadpool.
    # The search impls are sync and CPU/IO bound via Chroma; threads are fine.
    jobs: list[tuple[str, str]] = [(c, q) for c in channels for q in queries]
    raw_by_job: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def _dispatch(channel: str, query: str) -> list[dict[str, Any]]:
        # Read the current module-level impl each call so monkeypatch wins.
        import whisper.retrieve as _self

        if channel == "articles":
            return _self._search_knowledge_impl(query=query, limit=PER_QUERY_LIMIT)
        if channel == "code":
            return _self._search_codebase_impl(query=query, limit=PER_QUERY_LIMIT)
        return _self._search_raw_daily_impl(query=query, limit=PER_QUERY_LIMIT)

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(jobs)))) as ex:
        future_to_job = {
            ex.submit(_dispatch, c, q): (c, q)
            for c, q in jobs
        }
        for fut in future_to_job:
            c_q = future_to_job[fut]
            try:
                raw_by_job[c_q] = fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("retrieval job %s failed: %s", c_q, exc)
                raw_by_job[c_q] = []

    # For each channel, build a ranked-list-per-query of (hit_key, score), feed to RRF.
    # Also keep a key → (channel, raw) lookup so we can materialize Hits after fusion.
    key_to_source: dict[str, tuple[str, dict[str, Any]]] = {}
    per_channel_fused: list[tuple[float, str]] = []

    for channel in channels:
        ranked_lists_for_channel: list[list[tuple[str, float]]] = []
        for query in queries:
            raw = raw_by_job.get((channel, query), [])
            lst: list[tuple[str, float]] = []
            for item in raw:
                key = _hit_key(channel, item)
                # first-writer-wins; RRF doesn't care about score, only rank
                key_to_source.setdefault(key, (channel, item))
                # we don't actually need per-item score for RRF — pass 0.0
                lst.append((key, 0.0))
            ranked_lists_for_channel.append(lst)
        fused_channel = rrf_merge(ranked_lists_for_channel)
        for key, score in fused_channel:
            per_channel_fused.append((score, key))

    # Now fuse across channels: higher-scored (key, channel) wins. We just sort
    # the flat list by fused-channel score descending and take top_n unique keys.
    per_channel_fused.sort(reverse=True)
    seen: set[str] = set()
    ordered_keys: list[str] = []
    for _score, key in per_channel_fused:
        if key in seen:
            continue
        seen.add(key)
        ordered_keys.append(key)
        if len(ordered_keys) >= top_n:
            break

    hits: list[Hit] = []
    for i, key in enumerate(ordered_keys, start=1):
        channel, raw = key_to_source[key]
        hits.append(_to_hit(channel, raw, cid=f"c{i}", score=float(top_n - i)))
    return hits
