"""BM25 keyword index over curated articles.

Complements the ChromaDB vector store with a literal-token index so
identifier-shaped queries (class names, function names, file paths, error
strings) return the right article instead of a fuzzy concept match.

Design notes
------------
- Mirrors vector_store's (slug, zone) granularity so zone_filter keeps
  working across both retrieval paths. Each article contributes up to two
  corpus entries, one per non-empty Truth zone.
- Lazy, in-process singleton. First search builds the index from the
  knowledge tree; subsequent searches reuse it until a sentinel mtime
  (``state.json``) advances, at which point the index is rebuilt. The
  sentinel is written by ``reindex.reindex_articles`` on every successful
  embed, so a fresh compile automatically invalidates BM25 too — without
  any inter-process messaging.
- Tokenizer splits camelCase, snake_case, kebab-case, and punctuation so
  ``HybridEmailValidationService`` matches both the full identifier and
  its component words. BM25's IDF already down-weights common tokens, so
  no stopword list is maintained — the corpus decides.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# rank_bm25 (and its numpy dep) import eagerly here, NOT inside _build_index().
# FastMCP dispatches sync tool functions on an asyncio worker thread; deferring
# this import to first-call meant the import ran from that worker, where it
# could deadlock on Python's import lock against the main thread (which is
# blocked inside FastMCP's event loop). Symptom was a >99s hang on the first
# search_knowledge call. Eager import is cheap (~0.1s on Windows) and once
# loaded the module is cached in sys.modules, so we never pay it twice.
from rank_bm25 import BM25Okapi

from config import KNOWLEDGE_DIR, STATE_FILE

_index: Any = None                             # BM25Okapi | None
_docs: list[dict[str, Any]] = []               # parallel to the corpus
_last_sentinel_mtime: float | None = None

# Split any camelCase / PascalCase boundary so "HybridEmailValidationService"
# becomes "Hybrid Email Validation Service" before the lowercasing pass.
_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
_NON_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Normalize text into BM25 tokens.

    Splits camelCase boundaries, lowercases, and breaks on any non-
    alphanumeric separator (underscores, hyphens, slashes, punctuation).
    Single-character tokens are dropped — they never carry discriminative
    weight under BM25 IDF and only slow down the index.
    """
    if not text:
        return []
    decamelled = _CAMEL_RE.sub(r"\1 \2", text)
    lowered = decamelled.lower()
    tokens = _NON_TOKEN_RE.split(lowered)
    return [t for t in tokens if len(t) >= 2]


def _sentinel_mtime() -> float:
    """Return state.json's mtime or 0.0 if the file does not yet exist."""
    try:
        return STATE_FILE.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def _iter_article_zones() -> list[dict[str, Any]]:
    """Yield one corpus record per (slug, zone) pair with its tokenized text.

    Imports live inside the function so the module stays cheap to import
    under pytest (no compile_truth pull-in until the first search).
    """
    from compile_truth import (
        TruthZones,
        extract_fallback_truth,
        extract_zones,
        parse_frontmatter,
    )
    from utils import list_wiki_articles, load_contradictions

    quarantined = load_contradictions()
    records: list[dict[str, Any]] = []

    for article in list_wiki_articles():
        try:
            content = article.read_text(encoding="utf-8")
        except OSError:
            continue

        rel = str(article.relative_to(KNOWLEDGE_DIR)).replace("\\", "/")
        slug = rel.removesuffix(".md")
        fm = parse_frontmatter(content)
        zones = extract_zones(content)

        if not zones.observed and not zones.synthesized:
            legacy = extract_fallback_truth(content)
            if legacy:
                zones = TruthZones(observed=legacy, synthesized="")

        title = fm.get("title") or slug.split("/")[-1]
        base_metadata = {
            "slug": slug,
            "title": title,
            "type": fm.get("type"),
            "confidence": float(fm.get("confidence", 0.5)),
            "quarantined": slug in quarantined,
            "updated": fm.get("updated") or fm.get("created") or "unknown",
            "pinned": bool(fm.get("pinned", False)),
        }

        for zone_name, zone_text in (("observed", zones.observed), ("synthesized", zones.synthesized)):
            if not zone_text.strip():
                continue
            # Fold the title into the tokenized stream so title terms
            # contribute to the BM25 score without inflating the stored
            # excerpt text.
            tokens = tokenize(f"{title} {zone_text}")
            if not tokens:
                continue
            records.append({
                "id": f"{slug}::{zone_name}",
                "slug": slug,
                "text": zone_text,
                "tokens": tokens,
                "metadata": {**base_metadata, "zone": zone_name},
            })

    return records


def _build_index() -> None:
    """Rebuild the BM25 corpus from the current knowledge tree."""
    global _index, _docs

    _docs = _iter_article_zones()
    if not _docs:
        _index = None
        return
    _index = BM25Okapi([d["tokens"] for d in _docs])
    # Pre-compute a set per doc for O(1) query-token presence checks.
    # BM25Okapi's IDF can go negative when N is small or a term appears
    # in most docs, so we can't use "score > 0" as a match filter.
    for doc in _docs:
        doc["_token_set"] = set(doc["tokens"])


def _ensure_index() -> None:
    """Build or rebuild the index if the sentinel mtime has advanced."""
    global _last_sentinel_mtime
    current = _sentinel_mtime()
    if _index is None or current != _last_sentinel_mtime:
        _build_index()
        _last_sentinel_mtime = current


def invalidate() -> None:
    """Drop the cached index so the next search rebuilds.

    Called by reindex.py after a bulk re-embed so processes that share
    the knowledge tree via a long-running server (the MCP server) pick
    up freshly-compiled articles on their next query.
    """
    global _index, _docs, _last_sentinel_mtime
    _index = None
    _docs = []
    _last_sentinel_mtime = None


def _passes_filters(
    meta: dict[str, Any],
    min_confidence: float | None,
    type_filter: str | None,
    zone_filter: str | None,
    include_quarantined: bool,
) -> bool:
    if not include_quarantined and meta.get("quarantined"):
        return False
    if min_confidence is not None and float(meta.get("confidence", 0.0)) < min_confidence:
        return False
    if type_filter is not None and meta.get("type") != type_filter:
        return False
    if zone_filter is not None and meta.get("zone") != zone_filter:
        return False
    return True


def search_articles(
    query: str,
    limit: int = 5,
    min_confidence: float | None = None,
    type_filter: str | None = None,
    zone_filter: str | None = None,
    include_quarantined: bool = False,
) -> list[dict[str, Any]]:
    """Return BM25-ranked articles matching ``query``.

    Result shape mirrors ``vector_store.search_articles`` with one added
    ``score`` field (higher is better, native BM25 score). ``distance`` is
    set to ``-score`` so callers that sort by ascending distance still get
    sensible ordering.
    """
    _ensure_index()
    if _index is None or not _docs:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    query_set = set(query_tokens)

    scores = _index.get_scores(query_tokens)
    # Gate on token presence (not score) because BM25Okapi's IDF can be
    # negative — score alone doesn't tell us whether a doc actually
    # contained any of the query terms. BM25 still decides the rank order
    # among the gated docs.
    scored: list[tuple[float, dict[str, Any]]] = []
    for score, doc in zip(scores, _docs):
        if not (query_set & doc["_token_set"]):
            continue
        meta = doc["metadata"]
        if not _passes_filters(meta, min_confidence, type_filter, zone_filter, include_quarantined):
            continue
        scored.append((float(score), doc))

    scored.sort(key=lambda pair: -pair[0])

    out: list[dict[str, Any]] = []
    for score, doc in scored[:limit]:
        out.append({
            "id": doc["id"],
            "slug": doc["slug"],
            "text": doc["text"],
            "metadata": doc["metadata"],
            "score": score,
            "distance": -score,
        })
    return out


def stats() -> dict[str, int]:
    """Return a tiny summary for diagnostic tooling."""
    _ensure_index()
    return {
        "documents": len(_docs),
        "built": _index is not None,
    }
