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

import hashlib
import json
import threading
from typing import Any

from chroma_lock import chroma_write_lock
from chroma_path import ensure_configured_chroma_dir
from config import CHROMA_COLLECTION_CODEBASE

_client: Any = None
# RLock (not Lock) so callers that hold it across a _get_client() call do not
# self-deadlock on the lazy init path. Mirrors vector_store.py.
_lock = threading.RLock()
CHROMA_UPSERT_BATCH_SIZE = 256


def _get_client():
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is not None:
            return _client
        import chromadb
        import config

        db_dir = ensure_configured_chroma_dir(config.CHROMA_DB_DIR)
        _client = chromadb.PersistentClient(path=str(db_dir))
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
    with chroma_write_lock(CHROMA_COLLECTION_CODEBASE):
        _codebase_collection().upsert(ids=[chunk_id], documents=[text], metadatas=[flat])


def delete_chunks_for_file(rel_path: str) -> None:
    """Remove all chunks for a file before re-chunking it."""
    with chroma_write_lock(CHROMA_COLLECTION_CODEBASE):
        _codebase_collection().delete(where={"rel_path": {"$eq": rel_path}})


def _upsert_prepared_chunks(
    prepared: list[tuple[str, str, dict[str, Any]]],
) -> None:
    collection = _codebase_collection()
    for start in range(0, len(prepared), CHROMA_UPSERT_BATCH_SIZE):
        batch = prepared[start : start + CHROMA_UPSERT_BATCH_SIZE]
        collection.upsert(
            ids=[item[0] for item in batch],
            documents=[item[1] for item in batch],
            metadatas=[item[2] for item in batch],
        )


def replace_chunks_for_file(
    rel_path: str,
    chunks: list[dict[str, Any]],
) -> None:
    """Replace one file only after its complete new generation is staged."""
    pending_path = f"{rel_path}::pending-replacement"
    prepared: list[tuple[str, str, dict[str, Any]]] = []
    for chunk in chunks:
        text = chunk["text"]
        if not text.strip():
            continue
        metadata = {
            key: value
            for key, value in chunk["metadata"].items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        digest_input = (
            chunk["chunk_id"]
            + "\0"
            + text
            + "\0"
            + json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        )
        digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:16]
        metadata["rel_path"] = pending_path
        prepared.append(
            (f"{chunk['chunk_id']}::v{digest}", text, metadata)
        )

    with chroma_write_lock(CHROMA_COLLECTION_CODEBASE):
        collection = _codebase_collection()
        old = collection.get(
            where={"rel_path": {"$eq": rel_path}},
            include=[],
        )
        old_ids = set(old.get("ids") or [])
        staged_ids = [item[0] for item in prepared]
        staged_id_set = set(staged_ids)
        fresh_ids = list(staged_id_set - old_ids)
        overlapping = [
            (item_id, metadata)
            for item_id, _, metadata in prepared
            if item_id in old_ids
        ]
        try:
            if prepared:
                _upsert_prepared_chunks(prepared)
                collection.update(
                    ids=staged_ids,
                    metadatas=[
                        {**metadata, "rel_path": rel_path}
                        for _, _, metadata in prepared
                    ],
                )
            stale_ids = list(old_ids - staged_id_set)
            if stale_ids:
                collection.delete(ids=stale_ids)
        except Exception:
            if fresh_ids:
                collection.delete(ids=fresh_ids)
            if overlapping:
                collection.update(
                    ids=[item[0] for item in overlapping],
                    metadatas=[
                        {**item[1], "rel_path": rel_path}
                        for item in overlapping
                    ],
                )
            raise


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
