"""Find and repair records whose vector is missing from Chroma's index.

A record can be in Chroma's metadata (its id, document and metadata read
fine) while its vector is gone from the HNSW index: reading its embedding
raises "Error finding id" and vector search never returns it. That is what
the stale-index writes fixed in chroma_client left behind (392 of 2,503
article zones on 2026-10-08).

    uv run python scripts/vector_health.py            # check, exit 1 if damaged
    uv run python scripts/vector_health.py --repair   # re-embed damaged records
    uv run python scripts/vector_health.py --all      # include unused collections

Repair re-embeds each damaged record from its stored document and upserts it
in place: nothing is deleted, and ids, texts and metadata do not change.
Verified on a copy of the live store: 392 -> 0 unreadable, 392/392 found
again by searching their own text, still fixed after reopening.

Only the collections search reads are checked by default. Abandoned ones
(the unreadable legacy ``daily_chunks``, MiniLM ``articles`` while Gemma is
active) can hang on read, so they need ``--all`` or ``--collection``.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np

import chroma_client
import config
import embedder

BATCH = 200

Report = dict[str, tuple[int, list[str], "str | None"]]


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
            print(f"    checked {min(start + BATCH, len(ids))}/{len(ids)}, damaged so far {len(bad)}", flush=True)
    return bad


def in_use() -> list[str]:
    """The collections live search reads."""
    return [
        config.articles_collection_name(),
        config.CHROMA_COLLECTION_DAILY,
        config.CHROMA_COLLECTION_CODEBASE,
    ]


def _names(client: Any) -> list[str]:
    return sorted(c.name if hasattr(c, "name") else c for c in client.list_collections())


def _open(client: Any, name: str) -> Any:
    if name == config.CHROMA_COLLECTION_ARTICLES_GEMMA:
        return client.get_collection(name, embedding_function=embedder.GemmaChromaEmbeddingFunction())
    return client.get_collection(name)


def check(names: list[str]) -> Report:
    """name -> (records, damaged ids, error). One broken collection never
    stops the report for the others."""
    report: Report = {}
    with chroma_client.lock:
        client = chroma_client.get_client()
        existing = set(_names(client))
        for name in names:
            if name not in existing:
                report[name] = (0, [], "missing (not built yet)")
                continue
            try:
                collection = _open(client, name)
                ids = collection.get(include=[])["ids"]
                print(f"  {name}: {len(ids)} records", flush=True)
                report[name] = (len(ids), unreadable_ids(collection, ids, progress=True), None)
            except Exception as exc:
                report[name] = (0, [], f"UNREADABLE: {type(exc).__name__}: {exc}"[:300])
    return report


def _embed(name: str, docs: list[str], metas: list[Any]) -> list[Any]:
    """New vectors from the stored text, made by the collection's own embedder."""
    if name == config.CHROMA_COLLECTION_ARTICLES_GEMMA:
        titles = [(m or {}).get("title") for m in metas]
        return list(np.asarray(embedder.get_model(download=True).embed_documents(titles, docs), dtype=np.float32))
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    # A list of numpy rows: Chroma rejects lists of np.float32 scalars.
    return list(np.asarray(DefaultEmbeddingFunction()(docs), dtype=np.float32))


def repair_collection(name: str, damaged: list[str]) -> int:
    """Re-embed damaged records in place with ``upsert``. Nothing is deleted.

    Vectors are computed before the write lock is taken: Gemma needs about
    1-2 s per long document, and other writers wait for this lock.
    """
    repaired = 0
    for start in range(0, len(damaged), BATCH):
        batch = damaged[start : start + BATCH]
        with chroma_client.lock:
            got = _open(chroma_client.get_client(), name).get(ids=batch, include=["documents", "metadatas"])
        ids, docs = got["ids"], got["documents"]
        metas = [m or None for m in got["metadatas"]]
        if not ids:
            continue
        vectors = _embed(name, docs, metas)
        with chroma_client.write_session(name) as client:
            collection = _open(client, name)
            # The snapshot is from before the lock. A writer may have deleted
            # or rewritten a record meanwhile; upserting the old copy would
            # resurrect or revert it. Write back only what is still unchanged.
            now = collection.get(ids=ids, include=["documents", "metadatas"])
            current = {
                i: (d, m or None)
                for i, d, m in zip(now["ids"], now["documents"], now["metadatas"])
            }
            keep = [k for k, i in enumerate(ids) if current.get(i) == (docs[k], metas[k])]
            if keep:
                collection.upsert(
                    ids=[ids[k] for k in keep],
                    documents=[docs[k] for k in keep],
                    metadatas=[metas[k] for k in keep],
                    embeddings=[vectors[k] for k in keep],
                )
        repaired += len(keep)
    return repaired


def _print(report: Report) -> tuple[int, int]:
    damaged_total = errors = 0
    for name, (total, bad, error) in report.items():
        damaged_total += len(bad)
        if error:
            errors += error.startswith("UNREADABLE")
            status = error
        else:
            status = "ok" if not bad else f"{len(bad)} DAMAGED"
        print(f"  {name:<22} {total:>7} records  {status}", flush=True)
    return damaged_total, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check / repair missing Chroma vectors")
    parser.add_argument("--repair", action="store_true", help="Re-embed damaged records (upsert, no deletes)")
    parser.add_argument("--collection", help="Only this collection")
    parser.add_argument("--all", action="store_true", help="Also check collections search does not use")
    args = parser.parse_args()

    if args.collection:
        names = [args.collection]
    elif args.all:
        with chroma_client.lock:
            names = _names(chroma_client.get_client())
    else:
        names = in_use()

    print("Checking vector index health...", flush=True)
    report = check(names)
    damaged, errors = _print(report)
    if not damaged:
        return 1 if errors else 0
    if not args.repair:
        print(f"\n{damaged} records have no vector. Run with --repair to re-embed them.", flush=True)
        return 1

    for name, (_total, bad, _error) in report.items():
        if bad:
            print(f"Repairing {len(bad)} records in {name}...", flush=True)
            repair_collection(name, bad)
    print("\nRe-checking...", flush=True)
    remaining, recheck_errors = _print(check([n for n, (_t, bad, _e) in report.items() if bad]))
    # Collections that were UNREADABLE before are not repaired here; they
    # still count, or a fixed articles index would hide a broken daily one.
    return 0 if not remaining and not errors and not recheck_errors else 1


if __name__ == "__main__":
    sys.exit(main())
