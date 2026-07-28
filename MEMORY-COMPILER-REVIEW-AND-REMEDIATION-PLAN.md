# Memory Compiler Review Findings and Remediation Plan

Date: 2026-07-28

## Goal

Remove every confirmed data-loss, security, indexing, installation, and test
reliability defect found in the repository-wide review, while preserving the
byte-for-byte transcript recovery already completed.

## Constraints

- Existing transcript archives are authoritative and must never be rewritten
  from summaries.
- A failed external process must never be recorded as successful.
- A replacement index must remain complete and searchable if embedding fails.
- Filesystem paths received from HTTP or MCP callers must remain inside the
  intended knowledge root.
- Every behavior change follows a red-green regression-test cycle.
- Existing unrelated working-tree changes and recovery artifacts are preserved.
- No production LLM calls are made by the tests.

## Confirmed Findings

### F01 — Failed daily compilation is recorded as successful (P1)

`scripts/compile.py` rejects a non-zero Claude result only when stdout is
empty. A failed process that prints an error is allowed to update
`ingested_daily`, causing future runs to skip the source.

**Fix:** require exit code zero, use the compact wiki index, and verify that a
successful run produced an observable knowledge-base mutation before recording
the source hash.

**Tests:** non-zero stdout is rejected; zero-without-mutation is rejected;
zero-with-mutation is recorded.

### F02 — Concurrent compiler agents can overwrite shared articles (P1)

`compile.py` launches one unrestricted writer per changed daily log with
`asyncio.gather`. All writers edit the same concepts, connections, index, and
compile log from the same initial snapshot.

**Fix:** serialize daily compilation in deterministic oldest-first order and
hold a process-level compiler lock so separate hook processes cannot overlap.

**Tests:** multiple daily logs execute in order with maximum concurrency one;
two compiler processes cannot enter the critical section together.

### F03 — Shared state writes are non-atomic and lose concurrent updates (P1)

`scripts/utils.py` writes the entire `state.json` without a process lock,
reload/merge, or atomic replacement.

**Fix:** provide a locked state-update API that reloads current state, mutates
only the requested section, fsyncs a temporary file, and atomically replaces
the destination. Migrate all state writers to it.

**Tests:** concurrent updates to different sections both survive; readers never
observe partial JSON; obsolete keys can still be deliberately removed.

### F04 — Codebase replacement deletes the only good index first (P1)

`index_codebase.py` deletes prior chunks before all replacement embeddings are
available.

**Fix:** stage a complete per-file replacement with generation-tagged chunk
IDs, promote it only after all upserts succeed, then delete the previous
generation.

**Tests:** an injected embedding failure leaves all old chunks searchable and
removes partial staged chunks; success removes the old generation.

### F05 — Stored XSS in the local knowledge viewer (P1)

Markdown rendering permits raw HTML and templates mark the result safe.
Conversation- or LLM-derived `<script>` content therefore executes in the
viewer origin.

**Fix:** disable raw HTML in MarkdownIt and add a restrictive response Content
Security Policy as defense in depth.

**Tests:** script/iframe/event-handler input is escaped and the CSP disallows
inline scripts.

### F06 — Viewer and MCP article path traversal (P1)

Caller-controlled slugs are joined directly below `knowledge/`, allowing
`../` traversal to readable Markdown outside the knowledge root.

**Fix:** centralize canonical slug validation and resolved-path containment;
allow only the documented article directories and Markdown suffix.

**Tests:** normal nested articles work; encoded and direct traversal, absolute
paths, and disallowed directories are rejected by viewer and MCP operations.

### F07 — Deleted and emptied sources remain searchable (P2)

Codebase, article, daily, and transcript rebuilds only visit existing files.
An emptied code file records a new hash without deleting old chunks.

**Fix:** reconcile stored source paths with the current scan, delete missing or
empty-source chunks, and remove their hashes atomically.

**Tests:** deleted and emptied files lose all chunks and hashes; unchanged
existing files remain untouched.

### F08 — Chroma active-directory migration is missing (P2)

The configured database moved from `knowledge/chroma` to
`knowledge/chroma/active`, but existing populated stores are not migrated or
rejected.

**Fix:** add an idempotent migration that moves recognized legacy Chroma
artifacts into `active` under a migration lock, refuses ambiguous mixed stores,
and is invoked before clients open the database.

**Tests:** legacy-only migrates; active-only is unchanged; mixed layouts fail
loudly; an empty root creates `active`.

### F09 — Linked-project search uses the legacy Chroma path (P2)

`linked_search.py` opens sibling `knowledge/chroma`, bypassing the active-store
configuration.

**Fix:** resolve linked stores through the same active-directory helper.

**Tests:** linked search opens `<project>/knowledge/chroma/active`.

### F10 — Upgrade reports success after required setup failure (P2)

Dependency synchronization and migrations are treated as warnings, but the
upgrade marker is written and the command exits zero.

**Fix:** make required migration and dependency failures fatal and write the
version marker only after both succeed.

**Tests:** either failure returns non-zero and leaves no success marker; full
success writes it.

### F11 — Hook/flush child commands rely on bare `uv` (P2)

The installer may emit bare `uv`, both transcript hooks spawn it, and
`flush.py` uses it to start compilation. Restricted shells therefore archive
but fail to index or compile.

**Fix:** hooks execute `flush.py` with `sys.executable`; flush executes
`compile.py` with `sys.executable`; installer resolves known user-install
locations and refuses to install a broken command if no executable exists.

**Tests:** generated and runtime child commands contain an absolute/current
Python executable and work with `uv` removed from `PATH`.

### F12 — The same non-zero-with-stdout bug exists in Whisper (P2)

`enhance.py` and `expand_query.py` can treat failed Claude CLI processes as
successful content.

**Fix:** require exit code zero before parsing stdout and convert failures to
the existing typed fallback/error behavior.

**Tests:** non-zero stdout never becomes enhanced text or parsed expansion;
zero stdout follows the success path.

### F13 — Overlapping flushes process the same cursor window (P3)

The cursor read and final cursor write are individually safe, but
check-summary-append-advance is not serialized per session.

**Fix:** hold a per-session flush lease for the complete transaction. A second
process rechecks the cursor after acquiring the lease and exits if the window
was already handled.

**Tests:** two concurrent flush workers produce one summary/model call and one
cursor advancement; failure releases the lease without advancing.

### F14 — Viewer statistics read an obsolete state schema (P3)

The viewer counts `state["ingested"]`, while current state uses
`ingested_daily` and `ingested_sources`.

**Fix:** load migrated state and count the current sections without
double-counting.

**Tests:** current-schema fixtures report the expected counts; legacy fixtures
are migrated.

### F15 — Full test suite is not a trustworthy release gate (P2)

The suite has five failures and 46 errors. Whisper tests mock the removed SDK
client rather than the current Claude CLI boundary; installer and tray
expectations also describe obsolete behavior.

**Fix:** replace stale SDK mocks with subprocess-boundary fakes, correct
orchestrator/tray expectations to current contracts, and keep new regression
tests in the full suite.

**Tests:** the complete pytest suite exits zero.

### F16 — Recovery and storage documentation is contradictory (P3)

Architecture documentation recommends deleting all `knowledge/chroma`, which
now includes the active database and preserved recovery stores. Other passages
still describe summaries as raw transcripts or refer to the legacy store path.

**Fix:** document `knowledge/chroma/active`, non-destructive staged rebuilds,
the transcript archive as the raw source, and explicit recovery-directory
retention.

**Tests:** documentation is reviewed against the runtime constants and recovery
commands; no destructive whole-root deletion remains.

## Independent Adversarial Review Findings

The required read-only Codex review found six gaps in the first remediation
implementation. Reconciliation also found one adjacent metadata race. All were
treated as release blockers:

### IR01 — Identical codebase replacement deletes its promoted IDs (P1)

Deterministic staged IDs overlapped the prior generation on an identical
`--all` reindex, but cleanup deleted every old ID including the promoted IDs.

**Resolution:** delete only `old_ids - staged_ids`; on failure, delete only
fresh staged IDs and restore overlapping IDs to their live metadata.

**Evidence:** repeated-identical success and interrupted-identical replacement
tests both retain one complete searchable generation.

### IR02 — Daily and tool viewer routes allow Windows traversal (P2)

Article routes used canonical containment, but caller-controlled `date` values
were directly joined into daily and tool paths. Encoded backslashes escape the
daily directory on Windows.

**Resolution:** accept only exact `YYYY-MM-DD` route keys before filesystem
access; invalid keys return 404. Contradiction-view article reads also use the
canonical resolver.

**Evidence:** encoded-backslash traversal tests for both routes cannot disclose
sentinel files outside the intended directory.

### IR03 — Different sessions race on the shared daily file (P2)

Per-session flush locks do not serialize the first-of-day create/append/index
transaction shared by all sessions.

**Resolution:** hold a per-date file lock across header creation, append, and
replacement indexing so neither file truncation nor a stale vector snapshot can
win after a newer append.

**Evidence:** concurrent distinct-session append transactions retain both
entries and never overlap their embedding critical sections.

### IR04 — Cursor and archive can describe different snapshots (P2)

Hooks archived the transcript, then extracted the cursor window from the
still-growing live path.

**Resolution:** extract exclusively from the immutable archive path handed to
`flush.py`; later growth is handled by the next archive/window.

**Evidence:** both hook tests assert the extraction input is the archive.

### IR05 — Active-store migration is not crash-resumable (P2)

A crash after creating `active` but before moving every legacy artifact left a
mixed layout that the next run rejected as ambiguous.

**Resolution:** write a migration marker before the first move, resume only
marker-owned mixed layouts, remove the marker after completion, and continue to
reject unmarked mixed stores.

**Evidence:** an interrupted marked migration resumes; unmarked mixed layouts
still fail loudly.

### IR06 — Compiler reports success when daily compiles fail (P2)

Failure and success both returned cost `0.0`; the command printed completion
and exited zero even though no failed source hash advanced.

**Resolution:** return explicit per-source booleans, report succeeded/total,
and return process status 1 if any daily compile failed.

**Evidence:** subprocess, zero-without-mutation, and command-exit regressions
distinguish failure from successful credited output.

### IR07 — Cross-session flush metadata updates can clobber costs (P3)

`last-flush.json` used an unlocked stale read followed by a direct write.

**Resolution:** update it under a cross-process lock with reload, fsync, and
atomic replacement.

**Evidence:** concurrent session updates retain both cost-history entries.

### IR08 — Failed codebase rollback can delete the only generation (P1)

The first staged-replacement repair unconditionally deleted pending IDs in a
`finally` block. If restoring an overlapping live ID also failed, that cleanup
deleted the only remaining copy.

**Resolution:** never delete pending overlap IDs after a failed restoration.
Pending metadata remains searchable by normal code queries, so an interrupted
cleanup degrades metadata cleanliness instead of losing content.

**Evidence:** a failure-in-failure regression forces restoration to raise and
confirms the original document ID and content remain searchable.

### IR09 — Daily Markdown replacement deletes the good index first (P1)

Daily Markdown embedding still called delete before upsert even after article
and transcript replacement became staged.

**Resolution:** build the complete new chunk set first, replace it through the
shared loss-safe staged source transaction, and hold the same per-date lock
used by daily appenders during manual reindex.

**Evidence:** the daily embedding regression asserts the staged replacement
boundary is used, and concurrent append tests retain both sessions.

### IR10 — Capture hooks report archive and spawn failures as success (P2)

`SessionEnd` and `PreCompact` logged missing transcripts, archive failures, and
child-spawn failures but returned process status 0.

**Resolution:** both hook entry points now return explicit status codes, print
capture failures to stderr, and exit through `SystemExit(main())`.

**Evidence:** both hook variants return non-zero for missing transcripts and
failed process creation.

### IR11 — Detached flush failures have no durable retry record (P2)

A raw transcript indexing failure returned from the detached process with
status 0. The context file happened to remain, but no durable record retained
the cursor and archive arguments needed to discover and retry it.

**Resolution:** persist an fsynced pending-job sidecar before spawning. Every
capture hook retries all pending jobs; `flush.py` returns non-zero and retains
the job on raw-index, summary, or cursor failure, removing it only after the
transaction succeeds.

**Evidence:** regressions verify non-zero raw-index failure with retained
context, durable job retention after spawn failure, and job removal after a
successful raw-only retry.

### IR12 — Retry marker can outlive an unflushed context payload (P2)

The first durable-queue implementation fsynced its marker but created the
referenced context with `Path.write_text()`. A power loss could therefore
publish a retry job whose payload was empty or absent.

**Resolution:** create the context through a temporary file, flush and fsync
it, and atomically publish it. The later IR14 reconciliation also made the
marker self-contained so either durable artifact can recover the payload.

**Evidence:** a publication-failure regression confirms the self-contained
marker reconstructs a missing context file.

### IR13 — Hook session IDs can escape the runtime directory (P2)

Hook input supplied `session_id`, which was interpolated directly into the new
context filename. Path separators or `..` could redirect the write.

**Resolution:** filenames contain only a fixed prefix, a SHA-256 session
digest, and a UUID job ID. The original ID remains data inside the marker.

**Evidence:** both hook variants keep a traversal-shaped session ID directly
under the configured state directory.

### IR14 — Marker failure can orphan an undiscoverable context (P2)

Publishing context first still left a gap: if marker publication failed, later
hooks could not discover the otherwise durable context file.

**Resolution:** publish one fsynced, self-contained marker containing the
complete context, cursor, archive path, and session ID before materializing the
separate context file. The loader atomically reconstructs a missing context
from that marker.

**Evidence:** an injected context-publication failure leaves a readable marker,
and the next queue scan rebuilds the exact context.

### IR15 — Daily append and cursor were acknowledged before fsync (P2)

The daily entry and cursor replacement were closed/renamed but not explicitly
flushed to stable storage before the pending job was removed.

**Resolution:** flush and fsync the daily append, then write the cursor through
an fsynced temporary file and atomic replacement; only afterward remove pending
work.

**Evidence:** regressions assert daily fsync and cursor fsync-before-replace.

## Execution Plan

### Task 1 — Compilation and shared-state correctness

**Files:** `scripts/compile.py`, `scripts/utils.py`, their callers,
`tests/test_compile.py`, and `tests/test_state_updates.py`.

- [x] Add failing tests for F01–F03.
- [x] Implement strict process-success and output verification.
- [x] Serialize compilation under a process lock.
- [x] Implement atomic locked state mutation and migrate writers.
- [x] Run the focused compilation/state tests.

### Task 2 — Viewer and MCP security boundaries

**Files:** `scripts/viewer.py`, `scripts/knowledge_mcp_server.py`,
`scripts/utils.py`, templates, `tests/test_viewer.py`, and
`tests/test_knowledge_mcp.py`.

- [x] Add failing XSS and traversal tests for F05–F06.
- [x] Disable raw HTML and add response CSP.
- [x] Add canonical article path resolution and use it in all callers.
- [x] Run focused viewer/MCP tests.

### Task 3 — Loss-safe indexes and storage migration

**Files:** `scripts/index_codebase.py`, `scripts/codebase_store.py`,
`scripts/reindex.py`, `scripts/vector_store.py`, `scripts/config.py`,
`scripts/linked_search.py`, and their tests.

- [x] Add failing replacement/reconciliation/migration tests for F04 and
      F07–F09.
- [x] Implement staged code replacement and source reconciliation.
- [x] Implement and integrate the idempotent active-store migration.
- [x] Correct linked-project path resolution.
- [x] Run focused index and vector-store tests.

### Task 4 — Hook, flush, installer, upgrade, and viewer operations

**Files:** hooks, `scripts/flush.py`, `scripts/flush_cursor.py`,
`install.py`, `scripts/upgrade.py`, `scripts/viewer.py`, and related tests.

- [x] Add failing tests for F10–F11 and F13–F14.
- [x] Remove runtime dependence on bare `uv`.
- [x] Add a whole-transaction per-session flush lock.
- [x] Make required upgrade steps fail closed.
- [x] Correct viewer statistics.
- [x] Run focused operational tests.

### Task 5 — Whisper boundary and suite repair

**Files:** `scripts/whisper/`, `tests/whisper/`, `whisper_tray/`, and
`tests/whisper_tray/`.

- [x] Add failing non-zero-stdout tests for F12.
- [x] Implement strict Claude CLI result handling.
- [x] Replace stale SDK mocks and obsolete expectations for F15.
- [x] Run Whisper and tray tests.

### Task 6 — Documentation and release verification

**Files:** `README.md`, `docs/ARCHITECTURE.md`,
`ZERO-LOSS-TRANSCRIPT-INGESTION.md`, and this document.

- [x] Correct storage, raw-source, recovery, and operational documentation for
      F16.
- [x] Run focused tests after each subsystem.
- [x] Run Python compilation and the complete pytest suite.
- [x] Run an independent adversarial review against this plan and the diff.
- [x] Reconcile and fix confirmed review findings.
- [x] Record exact final verification output and check off completed tasks.

### Task 7 — Independent-review reconciliation

**Files:** `scripts/codebase_store.py`, `scripts/utils.py`,
`scripts/reindex.py`, `scripts/flush.py`, `scripts/pending_flush.py`, both
capture hooks, and their regression tests.

- [x] Make overlapping codebase rollback loss-safe (IR08).
- [x] Move daily Markdown indexing to staged source replacement (IR09).
- [x] Propagate hook and detached-flush failures as non-zero statuses (IR10).
- [x] Add durable, automatically retried pending flush jobs (IR11).
- [x] Publish fsynced context before durable retry metadata (IR12).
- [x] Remove untrusted session IDs from writable paths (IR13).
- [x] Make each pending marker self-contained and context-recoverable (IR14).
- [x] Fsync daily output and cursor state before queue cleanup (IR15).
- [x] Run the focused reconciliation tests.
- [x] Obtain a clean independent follow-up review.
- [x] Rerun the complete final verification gates.

## Definition of Done

- Every finding F01–F16 has a regression test or an explicit documentation
  verification.
- No failed external process advances ingestion, compilation, upgrade, or
  cursor state.
- Concurrent writers cannot lose state or overwrite knowledge compilation.
- Failed indexing retains the last complete searchable generation.
- Viewer and MCP callers cannot execute stored HTML or escape the knowledge
  root.
- Deleted sources cannot remain searchable after reconciliation.
- Existing Chroma data is migrated or rejected loudly, never silently ignored.
- The complete pytest suite, bytecode compilation, and diff checks pass.
- Independent review finds no unresolved P1/P2 defect in the implemented scope.

## Final Verification Evidence

- Initial repository baseline: **772 passed, 5 failed, 46 errors**.
- Whisper and tray repair: **128 passed**.
- Viewer and MCP combined regression run: **71 passed**.
- Post-review focused reconciliation run: **82 passed**.
- Complete suite before final hook retry reconciliation: **839 passed in
  69.17s**.
- Focused capture/flush reconciliation run: **48 passed in 1.58s**.
- Final complete suite after all review fixes: **850 passed in 68.14s**.
- `python -m compileall -q scripts hooks install.py`: passed.
- `git diff --check`: passed; only configured LF-to-CRLF conversion warnings
  were emitted.
- Documentation scan found no destructive whole-Chroma deletion command or
  obsolete claim that daily summaries are verbatim transcripts.
- Ruff was not run because this environment has no Ruff executable installed.

The independent reviewer initially reported one P1 and five P2 gaps (IR01–IR06).
Reconciliation added IR07. Follow-up reviews then found two staged-replacement
P1 failures (IR08–IR09) and six capture durability/security P2 failures
(IR10–IR15). Every finding was reproduced by control-flow inspection and fixed;
the final follow-up reviewer reported **no unresolved P1/P2 findings**.
