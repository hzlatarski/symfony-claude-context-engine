"""ChromaDB wrapper for the source-code index collection.

Collection: codebase
  Indexed: src/**/*.php, assets/controllers/**/*.js,
           templates/**/*.twig, config/**/*.yaml
  Chunked: 150-line windows, 30-line overlap
  Metadata: rel_path, file_type, start_line, end_line,
            symbols (comma-joined PHP class/method names)
  chunk_id: "{rel_path}::{chunk_index}"

Uses the same ChromaDB PersistentClient path as vector_store
(config.CHROMA_DB_DIR). ChromaDB deduplicates the backend connection
via its SharedSystemClient registry, so two modules holding separate
_client singletons pointing to the same db dir is safe.
"""
from __future__ import annotations

import threading
from typing import Any

from config import CHROMA_COLLECTION_CODEBASE

_client: Any = None
# RLock (not Lock) so callers that hold it across a _get_client() call do not
# self-deadlock on the lazy init path. Mirrors vector_store.py.
_lock = threading.RLock()


def _get_client():
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is not None:
            return _client
        import chromadb
        import config

        config.CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))
    return _client


def _codebase_collection():
    return _get_client().get_or_create_collection(
        name=CHROMA_COLLECTION_CODEBASE,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunk(
    chunk_id: str,
    rel_path: str,
    text: str,
    metadata: dict[str, Any],
) -> None:
    """Insert or update one code chunk. Skips empty text silently."""
    if not text.strip():
        return
    flat: dict[str, Any] = {
        k: v for k, v in metadata.items()
        if isinstance(v, (str, int, float, bool)) or v is None
    }
    flat["rel_path"] = rel_path
    _codebase_collection().upsert(ids=[chunk_id], documents=[text], metadatas=[flat])


def delete_chunks_for_file(rel_path: str) -> None:
    """Remove all chunks for a file before re-chunking it."""
    _codebase_collection().delete(where={"rel_path": {"$eq": rel_path}})


def search_codebase(
    query: str,
    limit: int = 5,
    file_type: str | None = None,
) -> list[dict[str, Any]]:
    """Semantic search over indexed source code chunks."""
    conditions: list[dict[str, Any]] = []
    if file_type is not None:
        conditions.append({"file_type": {"$eq": file_type}})

    where: dict[str, Any] | None = None
    if len(conditions) == 1:
        where = conditions[0]
    elif conditions:
        where = {"$and": conditions}

    result = _codebase_collection().query(
        query_texts=[query],
        n_results=limit,
        where=where,
    )
    return _flatten_results(result)


def _flatten_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    for id_, doc, meta, dist in zip(ids, docs, metas, dists):
        out.append({
            "id": id_,
            "rel_path": (meta or {}).get("rel_path", ""),
            "text": doc,
            "metadata": meta or {},
            "distance": dist,
        })
    return out


def stats() -> dict[str, int]:
    return {"codebase_chunks": _codebase_collection().count()}


def type_stats() -> dict[str, int]:
    """Return chunk count per file_type (php/js/twig/yaml) for the browse view."""
    counts: dict[str, int] = {}
    for ft in ("php", "js", "twig", "yaml"):
        try:
            result = _codebase_collection().get(
                where={"file_type": {"$eq": ft}},
                include=[],
            )
            counts[ft] = len(result["ids"])
        except Exception:
            counts[ft] = 0
    return counts
