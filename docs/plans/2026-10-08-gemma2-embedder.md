# EmbeddingGemma 2 embedder for the articles collection: spec and plan

Branch `feat/gemma2-embedder`, from `feat/kb-expand-neighbors` @ d4f8d0a (the branch the live
install runs). Evidence: `knowledge/research/embeddinggemma-2-vs-minilm.md` in the AiTutor repo.

## Why

On 400 English questions over the live KB, hybrid search finds the right article in the top 5
in 81.0% of cases today (MiniLM). With EmbeddingGemma 2 (ONNX q4) it is 88.8%: 48 wins,
17 losses, p<0.001. The ONNX q4 model loads in 0.3 s, embeds a query in 38 ms on CPU, and
needs no new dependencies (onnxruntime, tokenizers and huggingface_hub already ship with chromadb).

## Scope

In: the `articles` collection (curated articles + scenarios), and everything that reads it.
Out: `daily_chunks` and `codebase` stay on MiniLM in this slice. Images, audio, video.

## Spec

**REQ-1 Embedder.** A shared embedder produces EmbeddingGemma 2 text vectors from
`onnx-community/embeddinggemma-2-ONNX`, file `onnx/model_q4.onnx`, pinned to one HF revision.
- AC-1.1 Queries are embedded as `task: search result | query: {q}`.
- AC-1.2 Documents are embedded as `title: {title} | text: {text}`, or `title: none | text: {text}` with no title.
- AC-1.3 Output is 768-d and L2-normalized. Input is truncated at 2,048 tokens.
- AC-1.4 For 3 fixed texts, cosine vs the recorded reference vectors (ST fp32) is >= 0.98.
- AC-1.5 The model is downloaded once, with `local_dir=` (not the HF symlink cache), to
  `~/.cache/memory-compiler/embeddinggemma-2-onnx/<revision>/`, under a file lock.
- AC-1.6 With no cached model and no network, the error names the missing path and the fix command.

**REQ-2 Switchable collection.** The articles collection name depends on the embedder.
- AC-2.1 Default embedder `gemma2-q4` uses collection `articles_gemma2_q4`.
- AC-2.2 `MC_EMBEDDER=minilm` uses the old `articles` collection with Chroma's default embedder.
  Rollback is that env var. Nothing deletes or writes the old `articles` collection.
- AC-2.3 Each collection is opened with its own embedding function. A query never mixes models.

**REQ-3 Writes use titles.** `upsert_article` and `replace_article` pass explicit embeddings
built from `title` + `text`. The stored `documents` stay the plain text (search results do not change shape).

**REQ-4 Search never breaks because of the embedder.**
- AC-4.1 If the vector query raises, `hybrid_search.search_articles` returns the BM25 results and logs one warning.
- AC-4.2 The UserPromptSubmit hook keeps working, cold, in <= 3 s end to end (today ~1.3 s).

**REQ-5 All readers follow.** `linked_search`, `search_scenarios`, `ingest_scenarios`,
`whisper/retrieve`, `dedup` and `eval_corpus` reach the articles collection only through
the REQ-2 accessor. Linked projects that still have only `articles` are queried with MiniLM.

**REQ-6 Dedup thresholds are per embedder.** `DEFAULT_THRESHOLD` (0.88) and
`IDENTICAL_THRESHOLD` (0.97) were tuned on MiniLM. Gemma gets its own values, picked from
measured similarity data and recorded with the evidence in a code comment.

**REQ-7 Cut-over is verified.** After a full rebuild of the new collection:
- AC-7.1 The doc count equals the old `articles` count (articles x zones + scenarios) +/- known deltas.
- AC-7.2 The 400-question benchmark, run through the live `hybrid_search.search_articles`,
  reaches top-5 >= 86% (lower bound of the measured interval).
- AC-7.3 Cold hook latency is measured and is <= 3 s.

## Plan (7 tasks)

1. **Embedder module** — `scripts/embedder.py`: model fetch (REQ-1), `GemmaOnnxEmbedding`
   with `embed_documents(titles, texts)`, `embed_query(texts)`, and a Chroma
   `EmbeddingFunction` adapter (`__call__` = documents, `embed_query` = queries). Imported
   eagerly, like chromadb, to avoid the FastMCP import-lock deadlock noted in `vector_store.py`.
   Test: `tests/test_embedder.py` (prefixes, 768-d, unit norm, truncation, AC-1.4 parity
   against stored reference vectors, AC-1.6 offline error).
2. **Config + vector_store** — `MC_EMBEDDER`, `articles_collection_name()`, one accessor
   `articles_collection()` that passes the right embedding function; explicit title-aware
   embeddings on write (REQ-2, REQ-3). Test: `tests/test_vector_store.py` (name switch,
   minilm path untouched, stored documents unchanged, query uses the query prefix).
3. **BM25 fallback** — `hybrid_search.search_articles` catches the vector-path error (REQ-4.1).
   Test: `tests/test_hybrid_search.py::test_vector_failure_falls_back_to_bm25`.
4. **Readers** — route `linked_search`, `search_scenarios`, `ingest_scenarios`,
   `whisper/retrieve`, `eval_corpus` through the accessor (REQ-5). Test: existing tests for
   each + one new test that linked `articles`-only projects still answer.
5. **Dedup thresholds** — measure Gemma cosine for the pairs MiniLM flags today and for random
   pairs; pick values; make thresholds per embedder (REQ-6). Test: `tests/test_dedup.py`.
6. **Rebuild tooling** — `reindex.py --all --articles-only` and `ingest_scenarios` write into the
   new collection; progress output; resumable through the existing hash cache. Test:
   `tests/test_reindex_reconciliation.py` with a fake embedder.
7. **Cut-over (manual, on the live install)** — merge, run the rebuild (~35 min CPU), then verify
   REQ-7 with real numbers: counts, the 400-question benchmark through the live search path,
   and the cold hook timing. Rollback: `MC_EMBEDDER=minilm`.

After task 6: full `pytest`, then an adversarial review (Codex + Fable subagent) against this
spec, before merge.

## Addendum (2026-10-08): lost-vector bug found during the work

While calibrating dedup, the live store turned out to be damaged:
- `articles`: 392 of 2,503 zones are in Chroma's metadata but missing from the HNSW index
  (reading their embedding fails with "Error finding id"; vector search never returns them).
- `daily_chunks`: the index is ~8,100 log entries behind and Chroma can no longer apply the log
  ("Failed to apply logs to the hnsw segment writer"); every read hangs or fails, so
  `search_raw_daily` never answers (verified: a live call hung 30 min).
- `codebase`: healthy.

Root cause, reproduced with real processes: Chroma's local store keeps each HNSW index in
process memory and saves it over the files on disk. A process that loaded the index early and
wrote late (compile.py / ingest.py wait minutes on the LLM) saved a stale copy, erasing vectors
other processes had added; the write log was purged past that point. `chroma_write_lock`
serialized the calls, not the in-memory copies.

**REQ-8 Writes never save a stale index.** Every write reloads the index from disk under the
collection's write lock (`chroma_client.write_session`). All stores share one process client.
- AC-8.1 Two real processes (stale loader writes after another process wrote and saved): no
  vector of the other process is lost. The same scenario without the fix still loses them.
- AC-8.2 Every write function in vector_store and codebase_store reloads exactly once.

**REQ-9 Damage can be found and repaired.** `scripts/vector_health.py` lists records whose vector
is unreadable and, with `--repair`, re-embeds them (delete + re-add of the stored document and
metadata). AC-9.1 On a copy of the live store: articles 392 -> 0 damaged, 392/392 self-retrieved.

**REQ-10 Daily chunks are rebuilt.** The unreadable `daily_chunks` is replaced by
`daily_chunks_v2` (own hash key `vector_daily_hashes_v2`), rebuilt from the daily files by
`reindex.py --daily-only`. The broken collection is left in place, not deleted.

Cut-over order matters: merge first (old code would re-damage the store), then rebuild
articles (Gemma) and daily (v2), then run `vector_health.py` as the check.
