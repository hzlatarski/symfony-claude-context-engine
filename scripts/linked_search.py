"""Cross-project linked search — fan a single query across multiple KBs.

Mirrors SocratiCode's ``includeLinked`` pattern. With
``MEMORY_COMPILER_LINKED_PROJECTS`` set, ``search_knowledge`` can search
the local knowledge base PLUS the ``knowledge/chroma/`` collections of
sibling projects in a single call. Hits are tagged with a ``project``
label and RRF-merged across all sources.

Assumptions:
  - Each embedder has its own collection name, so a sibling's vectors are
    always queried with the model that made them: ``articles_gemma2_q4``
    with EmbeddingGemma 2 when the sibling has it, else the legacy
    ``articles`` with Chroma's default MiniLM. Fusion uses ranks only, so
    distances on different scales do not matter.
  - Linked-project paths point at the **project root** (the directory
    above ``.claude/memory-compiler/``). The Chroma store is resolved
    as ``<project_root>/knowledge/chroma/`` to match this project's
    layout.

Per-path Chroma clients are cached for the process lifetime — opening
a ``PersistentClient`` is non-trivial (~100ms) and the MCP server is
long-lived so the cache pays for itself fast.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import chromadb

import chroma_client
import config
import embedder
import hybrid_search
from chroma_path import ensure_active_chroma_dir
from config import (
    CHROMA_COLLECTION_ARTICLES,
    CHROMA_COLLECTION_ARTICLES_GEMMA,
    LINKED_PROJECTS,
)

# RRF constant — must match hybrid_search._RRF_K so local hits and linked
# hits live on the same fusion scale.
_RRF_K = 60

_LOCAL_LABEL = "<local>"

_client_cache: dict[str, Any] = {}


def _client_for(project_root: Path) -> Any:
    """Return a cached PersistentClient for the linked project, or None.

    None means the project has no Chroma store yet — silently skipped at
    search time so a misconfigured path doesn't break the local search.
    """
    chroma_root = project_root / "knowledge" / "chroma"
    if not chroma_root.exists():
        return None
    chroma_path = ensure_active_chroma_dir(chroma_root)
    key = str(chroma_path.resolve())
    cached = _client_cache.get(key)
    # A local write resets Chroma's process-wide cache (chroma_client.reset),
    # which orphans every client object, including these read-only ones.
    if cached is not None and chroma_client._alive(cached):
        return cached
    client = chromadb.PersistentClient(path=str(chroma_path))
    _client_cache[key] = client
    return client


def _linked_candidates() -> list[tuple[str, Any]]:
    """Collections to try in a sibling store, best first.

    Follows MC_EMBEDDER, so ``minilm`` really is a rollback for linked search
    too. Under Gemma the legacy collection is the fallback for a sibling that
    has not migrated (or has only an empty, half-built Gemma collection).
    """
    legacy = (CHROMA_COLLECTION_ARTICLES, None)
    if config.active_embedder() == config.EMBEDDER_MINILM:
        return [legacy]
    return [(CHROMA_COLLECTION_ARTICLES_GEMMA, embedder.GemmaChromaEmbeddingFunction()), legacy]


def _open(client: Any, name: str, ef: Any) -> Any:
    """``get_collection``, never ``get_or_create``: searching must not create
    collections in another project's store."""
    if ef is None:
        return client.get_collection(name)
    return client.get_collection(name, embedding_function=ef)


def _linked_articles_collection(client: Any) -> Any:
    """The first non-empty candidate collection, or None."""
    for name, ef in _linked_candidates():
        try:
            collection = _open(client, name, ef)
            if collection.count() > 0:
                return collection
        except Exception:
            continue
    return None


def _query_linked(client: Any, query: str, pool: int, where: Any) -> Any:
    """Query the best candidate; fall back to the next one if it errors."""
    for name, ef in _linked_candidates():
        try:
            collection = _open(client, name, ef)
            if collection.count() == 0:
                continue
            return collection.query(query_texts=[query], n_results=pool, where=where)
        except Exception:
            continue
    return None


def _build_where(
    *,
    min_confidence: float | None,
    type_filter: str | None,
    zone_filter: str | None,
    include_quarantined: bool,
) -> dict[str, Any] | None:
    """Build the Chroma where clause — duplicated from vector_store on purpose.

    Cross-project clients live in different process states; copying the
    11 lines here is preferable to taking a cross-module dependency on
    vector_store's private filter helpers.
    """
    conditions: list[dict[str, Any]] = []
    if not include_quarantined:
        conditions.append({"quarantined": {"$eq": False}})
    if min_confidence is not None:
        conditions.append({"confidence": {"$gte": float(min_confidence)}})
    if type_filter is not None:
        conditions.append({"type": {"$eq": type_filter}})
    if zone_filter is not None:
        conditions.append({"zone": {"$eq": zone_filter}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _flatten(result: dict[str, Any], project_label: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    for id_, doc, meta, dist in zip(ids, docs, metas, dists):
        out.append({
            "id": f"{project_label}::{id_}",
            "slug": (meta or {}).get("slug") or id_,
            "text": doc,
            "metadata": meta or {},
            "distance": dist,
            "project": project_label,
        })
    return out


def search_articles_linked(
    query: str,
    limit: int = 5,
    min_confidence: float | None = None,
    type_filter: str | None = None,
    zone_filter: str | None = None,
    include_quarantined: bool = False,
) -> list[dict[str, Any]]:
    """Hybrid local search + vector-only linked-project search, RRF-merged.

    Local hits come through ``hybrid_search.search_articles`` (BM25 +
    vector). Linked hits are vector-only — running BM25 cross-process
    would require either a shared corpus index or a remote service, and
    neither pulls its weight relative to the cost. Vector recall on
    linked projects is good enough for the "have we hit this elsewhere?"
    use case the feature is meant to serve.

    Returns top-``limit`` after fusion. Each hit carries a ``project``
    field — ``"<local>"`` for the current project, otherwise the last
    directory name of the linked project's path.
    """
    pool = max(limit * 3, limit)

    # Local first — full hybrid path. Already RRF-fused internally.
    local = hybrid_search.search_articles(
        query=query,
        limit=pool,
        min_confidence=min_confidence,
        type_filter=type_filter,
        zone_filter=zone_filter,
        include_quarantined=include_quarantined,
    )
    for r in local:
        r["project"] = _LOCAL_LABEL
        r["id"] = f"{_LOCAL_LABEL}::{r.get('id')}"

    # Linked — one query per project, vector-only.
    linked_by_project: dict[str, list[dict[str, Any]]] = defaultdict(list)
    where = _build_where(
        min_confidence=min_confidence,
        type_filter=type_filter,
        zone_filter=zone_filter,
        include_quarantined=include_quarantined,
    )
    for project_path in LINKED_PROJECTS:
        # Held per project: a local write's reset() must not stop the
        # system under a query that is still running.
        with chroma_client.lock:
            client = _client_for(project_path)
            if client is None:
                continue
            res = _query_linked(client, query, pool, where)
        if res is None:
            continue
        label = project_path.name or str(project_path)
        linked_by_project[label].extend(_flatten(res, label))

    # Sort each linked-project list by distance (lower = better) so RRF
    # ranks reflect that project's local quality.
    for label, items in linked_by_project.items():
        items.sort(key=lambda r: r.get("distance") if r.get("distance") is not None else 1.0)

    # Fuse local + each project as separate rank lists.
    scores: dict[str, float] = {}
    merged: dict[str, dict[str, Any]] = {}

    for rank, r in enumerate(local, start=1):
        rid = r["id"]
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (_RRF_K + rank)
        merged.setdefault(rid, r)

    for label, items in linked_by_project.items():
        for rank, r in enumerate(items, start=1):
            rid = r["id"]
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (_RRF_K + rank)
            merged.setdefault(rid, r)

    ordered = sorted(merged.values(), key=lambda r: -scores[r["id"]])
    for r in ordered:
        r["rrf_score"] = scores[r["id"]]
    return ordered[:limit]
