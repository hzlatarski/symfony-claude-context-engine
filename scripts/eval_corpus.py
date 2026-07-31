"""An in-memory corpus that ranks through the project's real retrieval path.

The benchmark needs to score *our* retrieval over *someone else's* corpus
(LongMemEval sessions, a regression fixture, a synthetic set). The live
stores can't do that: ``bm25_store`` and ``vector_store`` are singletons
bound to the knowledge tree and a persistent Chroma directory.

So this module supplies the corpus and borrows the algorithms:

* keyword ranking via ``bm25_store.TokenIndex`` — the project tokenizer
  and the negative-IDF presence gate, not a reimplementation;
* vector ranking via a Chroma **ephemeral** client using the same default
  ONNX embedder (all-MiniLM-L6-v2) the persistent store uses;
* fusion via ``hybrid_search.fuse_rankings`` — the same RRF constant and
  the same first-seen dedup as live search.

If any of those three change, the benchmark moves with them. That is the
entire point: a harness that copies the ranking code measures the copy.

Scope, stated precisely: this exercises the **ranking primitives**, not the
whole of ``search_articles``. The live store indexes one document per Truth
zone with the article title folded in and applies confidence/type/zone/
quarantine filters to both streams; none of that is reproduced here. A good
benchmark score says the tokenizer, the negative-IDF gate and the fusion are
sound — it does not certify live knowledge-base search.
"""
from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import chromadb

from bm25_store import TokenIndex
from hybrid_search import _POOL_MULTIPLIER, fuse_rankings

MODES = ("bm25", "vector", "hybrid")

# Chroma rejects oversized add() calls; mirror the persistent store's batch.
_ADD_BATCH_SIZE = 256

# Blank documents are excluded from the vector index entirely rather than
# embedded behind a placeholder. Nearest-neighbour search always returns a
# ranking, so a placeholder vector competes for top-k slots against real
# evidence — LongMemEval-S has 1228 zero-turn sessions, about 5% of the
# corpus, so that noise measurably depressed vector and hybrid recall. BM25
# already drops them naturally (they tokenize to nothing), so excluding them
# also makes the two streams agree on what exists.


class EphemeralCorpus:
    """A throwaway index over ``docs``, searchable through our real stack.

    ``docs`` is a sequence of mappings with ``id`` and ``text``. Nothing is
    persisted: the Chroma client is in-memory and dies with the instance,
    so a 500-question run builds and discards 500 indexes without touching
    the knowledge base's Chroma directory.
    """

    def __init__(self, docs: Sequence[Mapping[str, Any]]) -> None:
        self._ids: list[str] = []
        texts: list[str] = []
        seen: set[str] = set()

        for doc in docs:
            doc_id = str(doc["id"])
            if doc_id in seen:
                raise ValueError(f"duplicate document id in corpus: {doc_id!r}")
            seen.add(doc_id)
            self._ids.append(doc_id)
            texts.append(str(doc.get("text") or ""))

        self._texts = texts
        self._bm25 = TokenIndex(texts)
        self._collection: Any = None  # built lazily; embedding is the slow part
        self._client: Any = None
        # A unique name per corpus is mandatory, not hygiene. chromadb's
        # EphemeralClient resolves through a shared in-process SystemClient,
        # so a fixed name makes every corpus in a run the SAME collection —
        # each question's haystack piling onto the last until recall
        # collapses. This bug produced a full set of plausible-looking and
        # entirely invalid benchmark numbers before it was caught.
        self._collection_name = f"eval_{uuid.uuid4().hex}"
        self._embeddable_count = 0
        self.blank_documents = sum(1 for text in texts if not text.strip())

    def __len__(self) -> int:
        return len(self._ids)

    def __enter__(self) -> EphemeralCorpus:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Drop this corpus's collection. Safe to call more than once.

        Without this a 500-question run leaves 500 collections alive in the
        shared client for the life of the process.
        """
        if self._client is not None and self._collection is not None:
            try:
                self._client.delete_collection(self._collection_name)
            except Exception:
                # Already gone, or the shared client was torn down first —
                # either way there is nothing left to release.
                pass
        self._collection = None
        self._client = None

    # -- streams ---------------------------------------------------------

    def _ensure_collection(self) -> Any:
        """Embed the corpus into an ephemeral Chroma collection, once.

        Deferred so a ``mode="bm25"`` run never pays for embeddings —
        which is what makes the keyword baseline cheap enough to re-run
        on every tuning change.
        """
        if self._collection is not None:
            return self._collection

        client = chromadb.EphemeralClient()
        collection = client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._client = client
        embeddable_ids = [
            doc_id for doc_id, text in zip(self._ids, self._texts) if text.strip()
        ]
        embeddable_texts = [text for text in self._texts if text.strip()]
        if embeddable_ids:
            for start in range(0, len(embeddable_ids), _ADD_BATCH_SIZE):
                stop = start + _ADD_BATCH_SIZE
                collection.add(
                    ids=embeddable_ids[start:stop],
                    documents=embeddable_texts[start:stop],
                )
        self._embeddable_count = len(embeddable_ids)
        self._collection = collection
        return collection

    def _bm25_ranked(self, query: str, limit: int) -> list[dict[str, Any]]:
        return [
            {"id": self._ids[position], "score": score}
            for position, score in self._bm25.rank(query)[:limit]
        ]

    def _vector_ranked(self, query: str, limit: int) -> list[dict[str, Any]]:
        if not self._ids:
            return []
        collection = self._ensure_collection()
        if not self._embeddable_count:
            return []
        result = collection.query(
            query_texts=[query],
            n_results=min(limit, self._embeddable_count),
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            {"id": doc_id, "distance": distance}
            for doc_id, distance in zip(ids, distances)
        ]

    # -- public API ------------------------------------------------------

    def search(self, query: str, limit: int, mode: str = "hybrid") -> list[str]:
        """Return up to ``limit`` document ids, best first."""
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        if not self._ids or limit <= 0:
            return []

        if mode == "bm25":
            return [r["id"] for r in self._bm25_ranked(query, limit)]
        if mode == "vector":
            return [r["id"] for r in self._vector_ranked(query, limit)]

        # Hybrid: oversample each stream exactly as live search does, so a
        # document rescued by only one stream still has a rank slot to
        # contribute from.
        pool = max(limit * _POOL_MULTIPLIER, limit)
        fused = fuse_rankings(
            [self._vector_ranked(query, pool), self._bm25_ranked(query, pool)],
            limit,
        )
        return [r["id"] for r in fused]
