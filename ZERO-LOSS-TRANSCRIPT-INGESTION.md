# Zero-Loss Transcript Ingestion and Recovery

Date: 2026-07-28

## Purpose

This document records the transcript-ingestion failure, the changes made to
prevent the original loss mechanism, the historical recovery performed, and
the operational checks needed to maintain the transcript-preservation
guarantees. It does not certify the rest of Memory Compiler as lossless; the
post-change review findings are recorded below.

The change separates two concerns that were previously coupled:

1. A byte-for-byte archive and raw searchable index preserve everything.
2. The LLM-generated daily summary remains a convenient, intentionally lossy
   view of the conversation.

The summary is no longer the only copy of session information.

## Incident and Root Cause

The old transcript reader kept only the newest 30 user/assistant text turns and
then truncated the resulting context to roughly 15,000 characters. Despite
omitting earlier fresh turns, it returned the total transcript turn count as the
next cursor. The hook therefore marked omitted turns as consumed. In affected
sessions, ranges such as turns 129-259 could never be presented to a later
flush.

There was a second independent loss mechanism: even when a URL or identifier
reached the summarization prompt, the LLM was free to omit, shorten, or
paraphrase it. Searching only the summary could therefore fail even though the
original assistant message contained the exact value.

The failure was not just an undersized limit. Increasing 30 to a larger fixed
number would only move the loss boundary. The fix had to ensure that no omitted
content could be marked as processed and that exact source data remained
retrievable without relying on summarization.

## Required Invariants

The transcript archive and raw-index implementation enforces these rules:

- The original Claude transcript is archived byte-for-byte before a flush can
  advance its cursor.
- A stale, shorter snapshot cannot truncate an existing longer archive.
- A divergent snapshot is rejected instead of silently overwriting history.
- Every non-empty JSONL record is indexed, including tool inputs, tool results,
  and records without user/assistant text.
- Default summary extraction has no turn or character limit.
- If a caller deliberately requests a bounded batch, turns are processed
  oldest-first and the cursor advances only through turns fully included in the
  batch.
- Failure to archive or index raw data prevents cursor advancement.
- Before a detached flush is spawned, an fsynced pending-job record preserves
  its complete context, cursor, and archive arguments. A missing materialized
  context file is reconstructed from that self-contained record. Failed jobs
  return non-zero, remain queued, and are retried by the next capture hook.
- Runtime filenames use a digest of the untrusted session ID plus a UUID, so
  hook input cannot redirect pending writes outside the state directory.
- Concurrent cursor writers merge monotonically under a file lock.
- Replacing indexed chunks for one source cannot destroy the last complete
  searchable version if a write fails partway through.
- Exact URLs and long identifiers are indexed independently and can be fetched
  in full.

## Data Flow After the Change

At `PreCompact` or `SessionEnd`:

1. The hook locates the live Claude JSONL transcript.
2. `archive_transcript()` atomically copies it to
   `knowledge/daily/transcripts/<session-id>.jsonl`.
3. The hook extracts every fresh text turn from the saved cursor onward.
4. The hook starts `flush.py`, passing both the summary context and transcript
   archive.
5. `flush.py` indexes every raw transcript record before it calls the summary
   model.
6. If raw indexing fails, the flush stops and leaves the old cursor unchanged.
7. On success, the daily summary is written and the cursor advances.

Sessions containing only tool activity still run the raw-index path. They do
not require a user/assistant text turn to preserve their information.

## Code Changes and Why

### `scripts/transcript.py`

- Removed the default 30-turn and 15,000-character limits.
- Changed optional bounded extraction to process oldest-first and return the
  cursor immediately after the last included turn.
- Added monotonic, atomic, file-locked byte-for-byte transcript archiving.
- Added raw JSONL indexing for all record types.
- Added overlapping 6,000-character chunks so large records remain searchable.
- Added dedicated exact-reference chunks for URLs and long identifiers so
  chunk boundaries and semantic ranking cannot hide them.
- Preserved each record's own date where available, rather than assigning an
  entire multi-day session one date.

### `hooks/pre-compact.py` and `hooks/session-end.py`

- Archive the raw transcript before extracting or scheduling a summary.
- Refuse to proceed when archive validation fails.
- Reduced the minimum summary threshold to one text turn.
- Schedule raw indexing even when there are zero new text turns.
- Pass the archive path to `flush.py`.

### `scripts/flush.py`

- Index the raw archive before summary generation, deduplication, or cursor
  advancement.
- Stop without advancing the cursor if raw indexing fails.
- Append exact URLs from the source context and tool drawer to the generated
  daily summary deterministically. This improves summary usability, while the
  raw archive remains the authoritative copy.

### `scripts/vector_store.py`

- Added batch upserts for transcript-scale indexing.
- Added literal raw-chunk search and full-chunk retrieval.
- Added loss-safe source replacement:
  - new content receives immutable content-version IDs;
  - the complete replacement is staged under a pending source name;
  - old source chunks are deleted only after staging completes;
  - staged metadata is promoted to the real source;
  - pending leftovers are cleaned up.

If the process stops during replacement, either the previous complete source or
the complete staged source remains searchable.

### `scripts/chroma_lock.py`

- Updated lock placement for the new replaceable active database directory.
- Existing cross-process writer serialization continues to cover Chroma
  mutations.

### `scripts/flush_cursor.py`

- Added a cross-process file lock around the complete read/merge/write cycle.
- Cursor values merge using `max()` so a late writer cannot move a session
  backward.
- Cursor writes remain atomic and the tracked-session cap remains enforced.

### `scripts/knowledge_mcp_server.py`

- Changed `search_raw_daily` to combine literal-first and semantic results.
- Preserved source, session, date, and section metadata in slim results.
- Added `get_raw_daily_chunk(chunk_id)` to return the complete unclipped text
  for a search result.

### `scripts/backfill_transcripts.py`

- Added historical discovery and ingestion for Claude project transcripts.
- Supports all sessions or selected `--session` IDs.
- Continues processing other sessions after a per-session failure.
- Returns a non-zero process status when any requested session fails.

### `scripts/reindex.py` and `scripts/utils.py`

- Included archived transcript JSONL files in full and daily-only rebuilds.

### `scripts/config.py`

- Moved the configured live Chroma database to
  `knowledge/chroma/active`.

Using a replaceable child directory allows a verified rebuild to be switched
into service without deleting the previous database or other recovery
artifacts.

### `README.md` and `.gitignore`

- Documented the raw transcript archive, historical backfill commands, and
  full raw-chunk MCP retrieval.
- Kept transcript archives out of Git because they can contain private
  conversation content, tool payloads, file paths, and other sensitive data.

## Historical Recovery

The following affected sessions were re-archived and re-indexed:

- `8f013255-edee-4b5f-b461-63de65dcd0ab`
- `d63eb469-13fb-4dc1-9287-3d545015107d`
- `96f67ab9-152d-4acb-9a3f-707b86d04720`
- `f73a3afb-aa58-4dd0-8f56-0076b0761364`
- `7c3b97ef-d154-4dd4-a69f-2720fd309692`
- `5aa7671e-2f70-42e7-b2e9-cae0d070c33c`

For all six sessions, the archived file's SHA-256 digest matched the original
transcript. Exact artifact URLs were found through literal search and retrieved
unclipped through full-chunk lookup.

The existing live Chroma database was crashing inside native code during
writes, so it was not modified in place. A new single-writer database was built
and verified with:

- 1,186 article zones
- 29,010 raw daily/transcript chunks
- 2,849 codebase chunks

The verified database is:

`C:\wamp64\www\AiTutor\knowledge\chroma\active`

Old and failed databases were preserved under `knowledge/chroma`:

- `.failed-rebuild-zero-loss-20260728`
- `.recovery-corrupt-active-20260728`
- `.recovery-old-root-leftovers-20260728`

They were intentionally not deleted.

## Verification Performed

The original combined focused suite covering transcript extraction, archives,
hooks, backfill, cursor concurrency, raw MCP retrieval, and loss-safe vector
replacement passed 113 tests. A post-documentation rerun of the six directly
affected test files passed 88 tests.

Additional checks completed successfully:

- Python bytecode compilation for the changed scripts and hooks
- byte-for-byte archive hash comparison for all six recovered sessions
- exact URL search and full, unclipped retrieval for all six sessions
- semantic article and codebase searches against the rebuilt store
- live write/search/delete smoke test against the rebuilt store
- targeted adversarial tests for stale snapshots, cursor-writer races,
  multi-day dates, raw-only records, partial replacements, and backfill failure
  handling

That initial repository-wide run was not green: 772 tests passed, 5 failed,
and 46 errored. The remediation pass replaced obsolete Whisper SDK mocks with
tests at the current CLI subprocess boundary, corrected stale tray and
orchestrator expectations, and restored the full suite as a release gate.

## Repository-Wide Remediation

The repository-wide review found the following adjacent defects. The
remediation completed on 2026-07-28 addressed each one:

- Compiler and Whisper subprocesses now require exit code zero; ingestion and
  compilation also verify the expected knowledge mutation before advancing a
  source hash.
- Compilation is serialized oldest-first under a process lock, shared state
  uses locked atomic read-modify-write operations, and overlapping hooks use a
  whole-transaction per-session flush lock.
- Cursor windows are extracted from the exact archived snapshot passed to raw
  indexing, and a per-date transaction lock prevents different sessions from
  truncating or stale-indexing their shared daily summary file.
- Hooks and child processes use the active Python executable instead of
  relying on `uv` being inherited on `PATH`.
- Capture hooks surface missing transcripts, archive failures, and spawn
  failures through stderr and a non-zero process status. Detached flush jobs
  are removed from the durable retry queue only after raw indexing, summary
  persistence, and cursor advancement succeed.
- Daily summaries and cursor state are fsynced before pending work is removed,
  so a successful exit never acknowledges only operating-system cache state.
- Article, daily/transcript, and codebase replacements retain the last complete
  generation on write failure and reconcile deleted or emptied sources.
- Chroma clients migrate an unambiguous legacy-only store to `active` under a
  lock, resume an interrupted marked migration, and reject genuinely ambiguous
  mixed layouts.
- Viewer/MCP paths are containment-checked, raw Markdown HTML is disabled, and
  the viewer emits a restrictive nonce-based Content Security Policy.
- Upgrade dependency or migration failures now return non-zero without writing
  a success marker.

The full finding-by-finding record, test plan, and final evidence are in
`MEMORY-COMPILER-REVIEW-AND-REMEDIATION-PLAN.md`.

## Operations

### Restart after deployment

The old MCP process trees that held the corrupt database open were stopped.
Existing editor/agent connector transports cannot reconnect to a replaced MCP
server within the same session. Restart Claude Code, Codex, or the IDE once so
the connector starts against `knowledge/chroma/active` and discovers the
`get_raw_daily_chunk` tool.

### Backfill all discoverable sessions

```powershell
uv run python scripts/backfill_transcripts.py
```

### Backfill selected sessions

```powershell
uv run python scripts/backfill_transcripts.py `
  --session <session-id> `
  --session <session-id>
```

### Rebuild the index

```powershell
uv run python scripts/reindex.py --all
```

Do not delete the current active database before a rebuild. Build and validate
the replacement separately, then switch it into the `active` path while
preserving the previous copy.

### Retrieve exact information

1. Call `search_raw_daily` with the exact URL fragment, identifier, command, or
   phrase.
2. Use the returned result `id` with `get_raw_daily_chunk`.
3. Treat the full raw chunk and archived JSONL as authoritative when they
   disagree with an LLM-generated summary.

## Tradeoffs and Data Handling

The zero-loss path uses more disk space and embedding time than summary-only
storage. That cost is intentional: fixed-size summary windows cannot guarantee
retention.

Raw transcripts may contain sensitive content and tool results. They are
gitignored but remain local files and searchable Chroma documents. Access to
the project data directory should therefore be treated as access to the
original conversations, not merely to sanitized summaries.
