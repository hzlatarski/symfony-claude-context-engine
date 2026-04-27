"""ChromaDB wrapper for the memory compiler.

Two collections:
- articles:     curated concept articles, one document per (slug, zone) pair.
                zone in {"observed", "synthesized"}. Metadata filters on type,
                confidence, quarantined, updated.
- daily_chunks: raw daily-log chunks, never summarized. One document per H2
                section. Metadata filter on date.

Uses ChromaDB's default ONNX embedder (all-MiniLM-L6-v2) — fully local,
no API key, ~90 MB one-time download on first instantiation.
"""
from __future__ import annotations

import threading
from typing import Any

# chromadb (and the onnx + numpy chain it pulls in) is imported eagerly here,
# NOT lazily inside _get_client(). FastMCP dispatches sync tool functions on
# an asyncio worker thread; deferring this import to the first call meant the
# import chain ran from that worker, where it deadlocks on Python's import
# lock against the main thread (blocked in FastMCP's event loop). Symptom was
# a >99s hang on the first search_knowledge call. Eager import costs ~1s of
# startup but the module is then in sys.modules and threaded use is safe.
import chromadb

from chroma_lock import chroma_write_lock
from config import (
    CHROMA_COLLECTION_ARTICLES,
    CHROMA_COLLECTION_DAILY,
)

_client: Any = None
_lock = threading.RLock()


def _get_client():
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is not None:
            return _client
        import config

        config.CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))
    return _client


def _articles_collection():
    return _get_client().get_or_create_collection(
        name=CHROMA_COLLECTION_ARTICLES,
        metadata={"hnsw:space": "cosine"},
    )


def _daily_collection():
    return _get_client().get_or_create_collection(
        name=CHROMA_COLLECTION_DAILY,
        metadata={"hnsw:space": "cosine"},
    )


def _composite_id(slug: str, zone: str) -> str:
    if "::" in slug:
        raise ValueError(f"slug must not contain '::' separator, got {slug!r}")
    return f"{slug}::{zone}"


def _flatten_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Flatten caller metadata into Chroma-compatible primitives.

    Chroma only accepts str/int/float/bool/None in metadata values.
    Lists and tuples are joined with commas. Unknown types are dropped
    silently — callers should stick to primitives and sequences.
    """
    flat: dict[str, Any] = {}
    for k, v in metadata.items():
        if isinstance(v, bool) or isinstance(v, (str, int, float)) or v is None:
            flat[k] = v
        elif isinstance(v, (list, tuple)):
            flat[k] = ",".join(str(x) for x in v)
            # Add dedicated boolean flags to support exact-match filtering in ChromaDB
            for x in v:
                flat[f"{k}_{x}"] = True
    return flat


def _and_or_single(conditions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Build a Chroma where clause from a list of conditions.

    Chroma's $and with a single element is rejected on some 0.5.x
    versions, so we use the bare condition when there's exactly one.
    """
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _iso_to_date_int(value: Any) -> int | None:
    """Convert a YYYY-MM-DD string to an int like 20260412 for range queries.

    Chroma's $gte/$lte only accept int/float, so ISO date strings must be
    encoded as ints to be filterable. Returns None if the value isn't a
    parseable ISO date.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, str) and len(value) == 10 and value[4] == "-" and value[7] == "-":
        try:
            return int(value[0:4] + value[5:7] + value[8:10])
        except ValueError:
            return None
    return None


def upsert_article(
    slug: str,
    title: str,
    zone: str,
    text: str,
    metadata: dict[str, Any],
) -> None:
    """Insert or update one (slug, zone) document.

    Chroma needs primitive metadata (str/int/float/bool/None). Lists are
    joined with commas; unknown types are dropped. See _flatten_metadata.
    """
    if zone not in {"observed", "synthesized"}:
        raise ValueError(f"zone must be 'observed' or 'synthesized', got {zone!r}")
    if not text.strip():
        return  # empty zone — skip

    flat = _flatten_metadata(metadata)
    flat.update({"slug": slug, "title": title, "zone": zone})

    with chroma_write_lock(CHROMA_COLLECTION_ARTICLES):
        _articles_collection().upsert(
            ids=[_composite_id(slug, zone)],
            documents=[text],
            metadatas=[flat],
        )


def upsert_chunk(
    chunk_id: str,
    source_file: str,
    text: str,
    metadata: dict[str, Any],
) -> None:
    """Insert or update one verbatim daily-log chunk.

    If metadata contains a `date` field in YYYY-MM-DD form, we also inject
    `date_int` (an int like 20260412) so search_daily can filter by range —
    Chroma's $gte/$lte reject strings.
    """
    if not text.strip():
        return

    flat = _flatten_metadata(metadata)
    flat["source_file"] = source_file
    date_int = _iso_to_date_int(flat.get("date"))
    if date_int is not None:
        flat["date_int"] = date_int

    with chroma_write_lock(CHROMA_COLLECTION_DAILY):
        _daily_collection().upsert(ids=[chunk_id], documents=[text], metadatas=[flat])


def delete_article(slug: str) -> None:
    """Remove both zones of an article, if present."""
    with chroma_write_lock(CHROMA_COLLECTION_ARTICLES):
        _articles_collection().delete(
            ids=[_composite_id(slug, "observed"), _composite_id(slug, "synthesized")],
        )


def delete_chunks_for_daily(source_file: str) -> None:
    """Remove all chunks belonging to a daily file before re-chunking it."""
    with chroma_write_lock(CHROMA_COLLECTION_DAILY):
        _daily_collection().delete(where={"source_file": {"$eq": source_file}})


def search_articles(
    query: str,
    limit: int = 5,
    min_confidence: float | None = None,
    type_filter: str | None = None,
    zone_filter: str | None = None,
    include_quarantined: bool = False,
) -> list[dict[str, Any]]:
    """Semantic search over curated articles with metadata filters."""
    conditions: list[dict[str, Any]] = []
    if not include_quarantined:
        conditions.append({"quarantined": {"$eq": False}})
    if min_confidence is not None:
        conditions.append({"confidence": {"$gte": float(min_confidence)}})
    if type_filter is not None:
        conditions.append({"type": {"$eq": type_filter}})
    if zone_filter is not None:
        conditions.append({"zone": {"$eq": zone_filter}})

    with _lock:
        result = _articles_collection().query(
            query_texts=[query],
            n_results=limit,
            where=_and_or_single(conditions),
        )
    return _flatten_results(result)


def search_daily(
    query: str,
    limit: int = 5,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    """Semantic search over verbatim daily-log chunks with date-range filter.

    date_from / date_to are YYYY-MM-DD strings; filtering happens against
    the int-encoded `date_int` metadata field that upsert_chunk injects.
    """
    conditions: list[dict[str, Any]] = []
    if date_from:
        lower = _iso_to_date_int(date_from)
        if lower is not None:
            conditions.append({"date_int": {"$gte": lower}})
    if date_to:
        upper = _iso_to_date_int(date_to)
        if upper is not None:
            conditions.append({"date_int": {"$lte": upper}})

    with _lock:
        result = _daily_collection().query(
            query_texts=[query],
            n_results=limit,
            where=_and_or_single(conditions),
        )
    return _flatten_results(result)


def _flatten_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Zip Chroma's query result shape into a flat list of dicts.

    Defensive: Chroma has returned None (not []) for optional slots in some
    versions; use `or [[]]` on each key so zip never sees None.
    """
    out: list[dict[str, Any]] = []
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    for id_, doc, meta, dist in zip(ids, docs, metas, dists):
        out.append({
            "id": id_,
            "slug": (meta or {}).get("slug") or id_,
            "text": doc,
            "metadata": meta or {},
            "distance": dist,
        })
    return out


def stats() -> dict[str, int]:
    return {
        "articles": _articles_collection().count(),
        "daily_chunks": _daily_collection().count(),
    }
