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

# Separates a parent document id from its chunk ordinal. Chosen because no
# LongMemEval session id contains it; results are always mapped back before
# they leave this module.
_CHUNK_SEPARATOR = "##"


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping word windows.

    ``all-MiniLM-L6-v2`` truncates at ~190 words, so a single embedding of a
    1600-word session represents only its opening — any fact stated later is
    invisible to vector search. Splitting the document and ranking it by its
    best-matching window recovers that.

    The overlap exists so a fact straddling a window boundary is intact in at
    least one window; without it, splitting introduces its own blind spots.
    """
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")

    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]

    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks

# Chroma rejects oversized add() calls; mirror the persistent store's batch.
_ADD_BATCH_SIZE = 256

# When chunking, one document can occupy many consecutive ranks, so ask for
# well more than `limit` rows to be sure `limit` distinct parents emerge.
_CHUNK_OVERSAMPLE = 12

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

    def __init__(
        self,
        docs: Sequence[Mapping[str, Any]],
        chunk_words: int | None = None,
        chunk_overlap: int = 0,
    ) -> None:
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

        # Chunking applies to the vector index only. BM25 reads whole
        # documents and has no window limit, so splitting would only dilute
        # its term statistics for no gain.
        self._chunk_words = chunk_words
        self._chunk_overlap = chunk_overlap
        self.chunked = bool(chunk_words)
        if self.chunked:
            chunk_text("probe", size=chunk_words, overlap=chunk_overlap)  # fail fast

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
        embeddable_ids: list[str] = []
        embeddable_texts: list[str] = []
        for doc_id, text in zip(self._ids, self._texts):
            if not text.strip():
                continue
            if not self.chunked:
                embeddable_ids.append(doc_id)
                embeddable_texts.append(text)
                continue
            for ordinal, piece in enumerate(
                chunk_text(text, size=self._chunk_words, overlap=self._chunk_overlap)
            ):
                embeddable_ids.append(f"{doc_id}{_CHUNK_SEPARATOR}{ordinal}")
                embeddable_texts.append(piece)

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

        # With chunking, `limit` counts parent documents but the index holds
        # chunks — over-fetch so `limit` distinct parents can still emerge
        # even when one document's chunks dominate the head of the ranking.
        wanted = limit * _CHUNK_OVERSAMPLE if self.chunked else limit
        result = collection.query(
            query_texts=[query],
            n_results=min(wanted, self._embeddable_count),
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        ranked: list[dict[str, Any]] = []
        seen: set[str] = set()
        for doc_id, distance in zip(ids, distances):
            # A document is ranked by its single best-matching chunk, which is
            # simply the first one Chroma returns for it.
            parent = doc_id.split(_CHUNK_SEPARATOR)[0] if self.chunked else doc_id
            if parent in seen:
                continue
            seen.add(parent)
            ranked.append({"id": parent, "distance": distance})
            if len(ranked) >= limit:
                break
        return ranked

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
