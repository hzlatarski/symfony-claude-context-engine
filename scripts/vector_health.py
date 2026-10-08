"""Find and repair records whose vector is missing from Chroma's index.

A record can be in Chroma's metadata (its id, document and metadata read
fine) while its vector is gone from the HNSW index: reading its embedding
raises "Error finding id" and vector search never returns it. That is what
the stale-index writes fixed in chroma_client left behind (392 of 2,503
article zones on 2026-10-08).

    uv run python scripts/vector_health.py            # check, exit 1 if damaged
    uv run python scripts/vector_health.py --repair   # re-embed damaged records

Repair deletes each damaged record and adds it back with its stored document
and metadata, so it is re-embedded into the index. Ids, texts and metadata
do not change. Verified on a copy of the live store: 392 -> 0 unreadable,
392/392 found again by searching their own text.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

import chroma_client
import config
import embedder

BATCH = 200


def unreadable_ids(collection: Any, ids: list[str], progress: bool = False) -> list[str]:
    """Ids whose embedding cannot be read, found by bisecting failing batches."""
    bad: list[str] = []

    def probe(batch: list[str]) -> None:
        try:
            collection.get(ids=batch, include=["embeddings"])
        except Exception:
            if len(batch) == 1:
                bad.append(batch[0])
                return
            mid = len(batch) // 2
            probe(batch[:mid])
            probe(batch[mid:])

    for start in range(0, len(ids), BATCH):
        probe(ids[start : start + BATCH])
        if progress and (start // BATCH) % 25 == 0:
            print(f"    checked {start + BATCH}/{len(ids)}, damaged so far {len(bad)}", flush=True)
    return bad


def _collections(only: str | None = None) -> list[tuple[str, Any]]:
    """Every collection in the store, opened with the embedder that built it."""
    client = chroma_client.get_client()
    names = [c.name if hasattr(c, "name") else c for c in client.list_collections()]
    if only:
        names = [n for n in names if n == only]
    opened = []
    for name in names:
        if name == config.CHROMA_COLLECTION_ARTICLES_GEMMA:
            opened.append((name, client.get_collection(
                name, embedding_function=embedder.GemmaChromaEmbeddingFunction()
            )))
        else:
            opened.append((name, client.get_collection(name)))
    return opened


def check(only: str | None = None) -> dict[str, tuple[int, list[str]]]:
    report: dict[str, tuple[int, list[str]]] = {}
    with chroma_client.lock:
        for name, collection in _collections(only):
            ids = collection.get(include=[])["ids"]
            print(f"  {name}: {len(ids)} records", flush=True)
            report[name] = (len(ids), unreadable_ids(collection, ids, progress=True))
    return report


def repair_collection(name: str, damaged: list[str]) -> int:
    """Delete + re-add damaged records under the safe write session."""
    if not damaged:
        return 0
    for start in range(0, len(damaged), BATCH):
        batch = damaged[start : start + BATCH]
        with chroma_client.write_session(name) as client:
            if name == config.CHROMA_COLLECTION_ARTICLES_GEMMA:
                collection = client.get_collection(
                    name, embedding_function=embedder.GemmaChromaEmbeddingFunction()
                )
            else:
                collection = client.get_collection(name)
            got = collection.get(ids=batch, include=["documents", "metadatas"])
            ids, docs = got["ids"], got["documents"]
            metas = [m or None for m in got["metadatas"]]
            extra: dict[str, Any] = {}
            if name == config.CHROMA_COLLECTION_ARTICLES_GEMMA:
                titles = [(m or {}).get("title") for m in metas]
                extra["embeddings"] = list(embedder.get_model().embed_documents(titles, docs))
            collection.delete(ids=ids)
            collection.add(ids=ids, documents=docs, metadatas=metas, **extra)
    return len(damaged)


def _print(report: dict[str, tuple[int, list[str]]]) -> int:
    damaged_total = 0
    for name, (total, bad) in sorted(report.items()):
        damaged_total += len(bad)
        status = "ok" if not bad else f"{len(bad)} DAMAGED"
        print(f"  {name:<22} {total:>7} records  {status}", flush=True)
    return damaged_total


def main() -> int:
    parser = argparse.ArgumentParser(description="Check / repair missing Chroma vectors")
    parser.add_argument("--repair", action="store_true", help="Re-embed damaged records")
    parser.add_argument("--collection", help="Only this collection")
    args = parser.parse_args()

    print("Checking vector index health...", flush=True)
    report = check(args.collection)
    damaged = _print(report)
    if not damaged:
        return 0
    if not args.repair:
        print(f"\n{damaged} records have no vector. Run with --repair to re-embed them.")
        return 1

    for name, (_total, bad) in report.items():
        if bad:
            print(f"Repairing {len(bad)} records in {name}...", flush=True)
            repair_collection(name, bad)
    print("\nRe-checking...")
    remaining = _print(check(args.collection))
    return 0 if not remaining else 1


if __name__ == "__main__":
    sys.exit(main())
