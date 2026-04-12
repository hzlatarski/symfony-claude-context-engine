# Architecture

This document describes the retrieval pipeline, the four-store data model, and how the MCP tools route queries.

## Four Stores, One System

| Store | Location | Contents | Mutation point |
|---|---|---|---|
| **Daily logs (raw)** | `knowledge/daily/*.md` | Verbatim session transcripts, never summarized | `flush.py` (automatic, session-end) |
| **Concept articles (curated)** | `knowledge/concepts/*.md`, `knowledge/connections/*.md`, `knowledge/qa/*.md` | LLM-compiled Truth + Timeline articles with source anchors | `compile.py`, `ingest.py` |
| **Compiled truth (excerpt)** | `knowledge/compiled-truth.md` | Priority-scored top-N articles, always injected into sessions | `compile_truth.py` (pure Python, zero cost) |
| **Vector index** | `knowledge/chroma/` | ChromaDB: `articles` collection (by zone) + `daily_chunks` collection (by section) | `vector_store.upsert_*` called from compile / ingest / flush |

## Retrieval Routing

When Claude Code needs information, it has three escalating paths:

### Level 0 — Always Injected (SessionStart hook)

- `index.md` (~8 KB) — one line per article, acts as a map
- `compiled-truth.md` (~40 KB) — priority-scored excerpt, recency + linkedness + access + confidence
- Symfony codebase shape (~500 B) — file counts, top hotspots
- `wip.md` — resume-here state from the last session

### Level 1 — On-Demand MCP Tools

Claude picks these when Level 0 doesn't answer the question.

**Knowledge surface (`knowledge-compiler` server):**

- `search_knowledge(query, type, min_confidence, zone, include_quarantined)` — semantic search over curated articles
- `search_raw_daily(query, date_from, date_to)` — semantic search over verbatim drawer chunks
- `get_article(slug)` — fetch full article by slug
- `list_contradictions()` — current quarantine list

**Code surface (`symfony-code-intel` server):**

- `get_codebase_overview()`
- `get_file_deps(path)`
- `get_route_map(prefix)`
- `get_template_graph(template)`
- `get_stimulus_map(controller)`
- `get_hotspots(top_n)`

### Level 2 — Fallback to Read / Grep

If no tool exists for what Claude needs, it falls back to the agent's built-in `Read` and `Grep` tools. This is the escape hatch, not the norm.

## Why Two MCP Servers

Knowledge retrieval and code structure are philosophically different:

- **Knowledge** is eventually-consistent, compiler-LLM-written, confidence-scored, and has drift concerns.
- **Code** is the ground truth, parsed from the filesystem at query time, with mtime-based cache invalidation.

Mixing them forces one cache strategy onto both, and one failure mode (e.g. a bad LLM compile) onto queries that should never fail. Keeping them separate means the code-intel server stays rock-solid even if the knowledge pipeline has a bad day.

## The Two-Collection Vector Split

### `articles` collection

- **Document granularity:** one per `(slug, zone)` pair. The composite id is `{slug}::{zone}` so `delete_article(slug)` can wipe both zones in a single call.
- **Metadata:** `type`, `confidence`, `quarantined`, `updated`, `pinned`, `zone`, `slug`, `title`.
- **Filters:** `type_filter`, `min_confidence`, `zone_filter`, `include_quarantined` — all exposed through `search_knowledge`.
- **Purpose:** curated semantic search — "what have I decided about X?"

### `daily_chunks` collection

- **Document granularity:** one per `##` / `###` section of a daily log. Real daily logs nest `### Session (HH:MM)` under `## Sessions` — splitting on H3 gives one chunk per session event.
- **Metadata:** `source_file`, `section`, `date`, `date_int` (YYYYMMDD as int, since Chroma's `$gte` / `$lte` reject strings).
- **Filters:** date range via `date_from` / `date_to`.
- **Purpose:** verbatim drill-down — "what exactly did I say about X on 2026-04-10?"

The two collections are split because the datasets have wildly different durability semantics: curated articles get rewritten; daily logs never change after the day ends.

## Embedding

- **Model:** `all-MiniLM-L6-v2` via ONNX runtime — fully local, ~90 MB one-time download on first instantiation. Zero API cost, zero network dependency after the first run.
- **Vector dimension:** 384
- **Distance metric:** cosine (set via `hnsw:space: "cosine"` on collection creation)
- **Chunking:** markdown-native via `chunk_daily.py`. Splits on `##` + `###` headings; empty sections (header only) are dropped; duplicate titles inside one file get a numeric suffix.

## State & Idempotency

- `scripts/state.json` tracks per-file hashes for source ingestion **and** vector index state (`vector_article_hashes`, `vector_daily_hashes`). `reindex.py` uses these for incremental updates.
- `vector_store.upsert_*` is idempotent by `(slug, zone)` / `chunk_id`. Safe to call repeatedly.
- `embed_article_file` always calls `delete_article` before upserting both zones, so stale zones from an older version of the article can't linger after the article shrinks.
- `reindex_articles` / `reindex_daily` wrap their loops in `try/finally` so a mid-run failure persists the hash cache for everything embedded up to that point, preventing drift between Chroma and the cache.
- Failures in the vector layer never block the compile / ingest / flush pipelines — they're logged to stderr (or `flush.log` for the detached background flush) and skipped, because a missing vector index is recoverable (run `reindex.py`) while a broken compile is not.

## Compile-Time Flow

```
daily/YYYY-MM-DD.md ──► compile.py
                            │
                            ├──► LLM writes/updates concepts/*.md
                            │    (with skeptical merge vs existing Truth,
                            │     flagging CONTRADICTION: lines to lint)
                            │
                            ├──► reindex_articles(force=False)
                            │    (hash-detects the LLM's writes, embeds
                            │     changed articles into Chroma)
                            │
                            └──► compile_truth.py
                                 (pure Python — reads all articles,
                                  applies confidence decay + priority
                                  scoring, writes compiled-truth.md)
```

## Flush-Time Flow (Session End)

```
Claude Code ──► SessionEnd / PreCompact hook ──► flush.py (detached bg proc)
                                                     │
                                                     ├──► Haiku extracts WIP
                                                     │    + session facts
                                                     │
                                                     ├──► append_to_daily_log()
                                                     │    writes to knowledge/daily/YYYY-MM-DD.md
                                                     │
                                                     ├──► embed_daily_file()
                                                     │    re-chunks + re-embeds the whole day
                                                     │    into Chroma's daily_chunks collection
                                                     │
                                                     └──► update_wip_file() (if non-empty)
                                                          writes wip.md
```

## Cost Model

Per-operation costs are constant regardless of knowledge base size, because the LLM only ever sees:

1. `index.md` (grows slowly — one line per article)
2. `compiled-truth.md` (fixed character budget, default 40 KB)
3. On-demand article fetches via `get_article` / `Read`

Vector store operations are free (local ONNX embeddings — never touches the network after the first model download). LLM calls happen in `compile.py`, `ingest.py`, `flush.py`, `query.py`, `lint.py` (full mode), and `canary.py` — all of which run on fixed inputs per invocation.

## Failure Modes and Recovery

| Failure | Impact | Recovery |
|---|---|---|
| ChromaDB file corruption | Search returns nothing | `rm -rf knowledge/chroma/` + `reindex.py --all` |
| Hash cache drift (Chroma has data the cache doesn't know about, or vice versa) | Re-embeds look "clean" but aren't | Delete `state.json["vector_*_hashes"]` + `reindex.py --all` |
| Flush can't reach Chroma (background fail) | That session's daily chunks are missing from the store | Next flush catches up via `embed_daily_file` re-chunking the whole file |
| compile.py crashes mid-run | State partially advanced, some articles embedded, others not | `try/finally` in `reindex_articles` preserves the cache; subsequent compile re-embeds what's missing |
| An article's LLM compile generates a CONTRADICTION marker | Article gets quarantined via `lint.py` and excluded from search + compiled-truth | Human review, then `lint.py --resolve` to clear |
| Canary questions start failing | Early warning for compiler drift | Investigate the failing canary(s); typically indicates a regression in compile.py or the knowledge it references |
