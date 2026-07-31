"""Hybrid retrieval fusing vector (Chroma) and BM25 (rank_bm25) results.

Uses Reciprocal Rank Fusion with the standard k=60 constant — no score
normalization required, which matters because BM25 scores and Chroma
cosine distances live on different scales. Documents that appear in both
ranked lists accumulate RRF contributions from each, surfacing matches
that either path alone would rank below the cutoff.

Entry point is ``search_articles`` which mirrors the signature of both
``vector_store.search_articles`` and ``bm25_store.search_articles`` so
the MCP tool layer can swap modes without caring about the backend.
"""
from __future__ import annotations

from typing import Any

import bm25_store
import vector_store

# RRF constant from Cormack et al. 2009. 60 is the canonical value — moving
# it only shifts how aggressively high ranks dominate, and empirically the
# default serves generic retrieval well.
_RRF_K = 60

# Oversample each path so documents the other path rescues still have a
# rank slot to contribute from. Without the multiplier, an article ranked
# 6th by one path and invisible to the other would never appear even when
# the user asked for limit=5.
_POOL_MULTIPLIER = 3


def fuse_rankings(
    ranked_lists: list[list[dict[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
    """Fuse any number of ranked result lists via Reciprocal Rank Fusion.

    Each list must be ordered best-first and each result must carry an
    ``id``. A document appearing in several lists accumulates one
    ``1 / (k + rank)`` contribution per list, which is what makes
    corroboration across streams outrank a single stream's strong opinion.

    The first list to yield a given id supplies the returned payload; later
    duplicates only add score. Returns the top ``limit`` with an added
    ``rrf_score`` field.

    Extracted so the retrieval benchmark fuses through the identical code
    path as live search — a copied constant would drift silently.
    """
    scores: dict[str, float] = {}
    merged: dict[str, dict[str, Any]] = {}

    for results in ranked_lists:
        # One vote per document per stream. A stream that repeats an id would
        # otherwise cast two votes and outrank a document both streams agreed
        # on — the opposite of what RRF is for. The first (best) rank wins.
        counted: set[str] = set()
        for rank, result in enumerate(results, start=1):
            rid = result["id"]
            if rid in counted:
                continue
            counted.add(rid)
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (_RRF_K + rank)
            merged.setdefault(rid, result)

    ordered = sorted(merged.values(), key=lambda r: -scores[r["id"]])
    for result in ordered:
        result["rrf_score"] = scores[result["id"]]
    return ordered[:limit]


def search_articles(
    query: str,
    limit: int = 5,
    min_confidence: float | None = None,
    type_filter: str | None = None,
    zone_filter: str | None = None,
    include_quarantined: bool = False,
) -> list[dict[str, Any]]:
    """Fuse vector + BM25 results for ``query`` via Reciprocal Rank Fusion.

    Each path is asked for ``limit * _POOL_MULTIPLIER`` candidates so the
    fused top-``limit`` has room to pick winners from either side. Results
    are deduplicated by composite id (``slug::zone``) and returned sorted
    by descending RRF score. The ``rrf_score`` field is added to each
    returned dict for debugging; callers that don't care can ignore it.
    """
    pool = max(limit * _POOL_MULTIPLIER, limit)
    filters = {
        "limit": pool,
        "min_confidence": min_confidence,
        "type_filter": type_filter,
        "zone_filter": zone_filter,
        "include_quarantined": include_quarantined,
    }

    vec_results = vector_store.search_articles(query=query, **filters)
    bm25_results = bm25_store.search_articles(query=query, **filters)

    # Vector first so its payload wins on a first-seen tie; both shapes
    # carry the same slug/text, but the vector path has richer metadata.
    return fuse_rankings([vec_results, bm25_results], limit)
