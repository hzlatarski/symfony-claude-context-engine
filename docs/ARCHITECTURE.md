# Architecture

This document describes the retrieval pipeline, the four-store data model, and how the MCP tools route queries.

## Four Stores, One System

| Store | Location | Contents | Mutation point |
|---|---|---|---|
| **Session evidence** | `knowledge/daily/*.md`, `knowledge/daily/transcripts/*.jsonl` | Readable lossy summaries plus byte-for-byte raw Claude transcripts | `flush.py` and transcript hooks |
| **Concept articles (curated)** | `knowledge/concepts/*.md`, `knowledge/connections/*.md`, `knowledge/qa/*.md` | LLM-compiled Truth + Timeline articles with source anchors | `compile.py`, `ingest.py` |
| **Compiled truth (excerpt)** | `knowledge/compiled-truth.md` | Priority-scored top-N articles, always injected into sessions | `compile_truth.py` (pure Python, zero cost) |
| **Vector index** | `knowledge/chroma/active/` | ChromaDB: `articles`, `daily_chunks`, and `codebase` collections | staged replacements called from compile / ingest / flush / reindex |

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
- `search_raw_daily(query, date_from, date_to)` — literal-first plus semantic search over daily and raw-transcript chunks
- `get_raw_daily_chunk(chunk_id)` — fetch the complete, unclipped source chunk
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

- **Document granularity:** one per `(slug, zone)` pair. Replacement IDs include a content version; `delete_article(slug)` removes every version by `slug` metadata.
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

- `scripts/state.json` tracks per-file hashes for source ingestion **and** vector index state (`vector_article_hashes`, `vector_daily_hashes`). Every mutation reloads the current file under a cross-process lock, updates only its section, fsyncs a temporary file, and atomically replaces the destination.
- Article, daily/transcript, and codebase replacement stages a complete content-versioned generation before deleting stale chunks. Promotion is idempotent even when old and new IDs overlap, and a failed replacement retains the last complete searchable generation.
- Full reindex reconciles stored sources against disk, removing hashes and chunks for deleted or emptied files.
- `reindex_articles` / `reindex_daily` wrap their loops in `try/finally` so a mid-run failure persists the hash cache for everything embedded up to that point, preventing drift between Chroma and the cache.
- Curated-vector refresh failures remain recoverable through `reindex.py`. Raw transcript archive/index failure is stricter: it blocks cursor advancement so source information can never be marked consumed before it is durable and searchable.

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
                                                     ├──► archive original JSONL
                                                     │    + index every raw record
                                                     │    (failure leaves cursor unchanged)
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

The cursor window is extracted from the immutable archived snapshot, not the
still-growing live transcript. Per-session locks prevent duplicate cursor
windows, while a separate per-date lock serializes daily-file creation,
append, and replacement indexing across different sessions.

## Cost Model

Per-operation costs are constant regardless of knowledge base size, because the LLM only ever sees:

1. `index.md` (grows slowly — one line per article)
2. `compiled-truth.md` (fixed character budget, default 40 KB)
3. On-demand article fetches via `get_article` / `Read`

Vector store operations are free (local ONNX embeddings — never touches the network after the first model download). LLM calls happen in `compile.py`, `ingest.py`, `flush.py`, `query.py`, `lint.py` (full mode), and `canary.py` — all of which run on fixed inputs per invocation.

## Failure Modes and Recovery

| Failure | Impact | Recovery |
|---|---|---|
| ChromaDB file corruption | Search fails or returns nothing | Stop writers, rename `knowledge/chroma/active` to a uniquely named recovery directory under `knowledge/chroma/`, then run `reindex.py --all`; verify the rebuild before deleting any retained recovery store |
| Hash cache drift (Chroma has data the cache doesn't know about, or vice versa) | Re-embeds look "clean" but aren't | Delete `state.json["vector_*_hashes"]` + `reindex.py --all` |
| Flush can't reach Chroma (background fail) | That session's daily chunks are missing from the store | Next flush catches up via `embed_daily_file` re-chunking the whole file |
| compile.py crashes mid-run | The current daily source remains unacknowledged unless its article mutation completed | The serialized next compile retries the unacknowledged source; staged vector replacement retains the last complete searchable article generation |
| An article's LLM compile generates a CONTRADICTION marker | Article gets quarantined via `lint.py` and excluded from search + compiled-truth | Human review, then `lint.py --resolve` to clear |
| Canary questions start failing | Early warning for compiler drift | Investigate the failing canary(s); typically indicates a regression in compile.py or the knowledge it references |
