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

## Post-Release Verification Round (2026-07-30)

A verification pass re-derived every claim above from the shipped code rather
than from the checklists. The suite result reproduced exactly (**850 passed**),
and F01–F16 and IR01–IR15 were each traced to concrete implementations. The
core zero-loss mechanism verified clean: both extraction limits are `None` and
`next_cursor = start + len(selected)`, so the cursor cannot outrun the content.

Nine defects were found in the *claims*, not in the core guarantee. All are now
fixed with regression tests (`tests/test_review_fix_hardening.py`).

The fixes themselves were then put through the same adversarial review, twice.
The first pass found six problems in them — including a genuine self-inflicted
data-loss risk (PR03's original hand-out claim) and a false-accept path (PR05's
original log gate). The second pass, run against the reconciled result, found
three more: attempts not counted when a worker raised rather than returned, a
lone `Source:` line still satisfying the compile gate, and duplicate spawns
still possible because the queue lock guards only the scan. All were fixed
before this document was finalized; the details are recorded inline with the
finding they belong to.

### PR01 — The authoritative archive was the only artifact never fsynced (P1)

IR15 fsynced the daily append and the cursor so a successful exit could not
acknowledge cache state, but `archive_transcript()` published via
`write_bytes()` + rename. A power loss after cursor publication could leave a
cursor that consumed turns the archive no longer held — the exact failure IR15
existed to prevent, on the one file the design calls authoritative.

**Fix:** write, flush, and `os.fsync()` the archive before the rename, then
best-effort fsync the containing directory so the new entry is durably
published too. Directory fsync is a no-op on Windows, which has no equivalent;
the file-content fsync is the portable part of the guarantee.

### PR02 — One corrupt marker permanently halted every future capture (P2)

`load_pending_flushes()` raised on any unusable record, and both hooks call it
inside a handler that spawns nothing on error. A single bad marker therefore
stopped all raw indexing and summarization, for every session, forever —
falsifying IR11's automatic-retry claim. Archives still survived on disk, so
this was availability, not loss.

**Fix:** structurally unusable markers are moved aside as `.corrupt` and
reported on stderr; the queue keeps draining. An I/O error is deliberately not
treated as corruption — a briefly unreadable marker (antivirus, sharing
violation, permissions) is left untouched for the next run — and a failed
quarantine rename never falls back to deleting the marker, because the marker
is the only record of the session, cursor, and archive path.

### PR03 — Pending jobs had no retry ceiling or real ordering (P2)

Hooks spawned one process per marker with no backoff, so a permanently-failing
job was relaunched by every capture hook indefinitely and piled up processes
blocked on the session lease. Separately, the loader's documented "oldest-first"
order sorted filenames built from a session digest and a random UUID (IR13's own
scheme), making the order effectively random.

**Fix:** `MAX_FLUSH_ATTEMPTS` retires a hopeless job as `.exhausted`, and an
explicit `created_at` provides the real ordering.

Attempts are counted by the **worker that actually ran and failed**
(`flush.py` records one on a non-zero exit *and* on an unhandled exception),
never at hand-out time. The first implementation of this fix claimed an attempt
when the hook loaded the job, which the adversarial reviewer correctly
rejected: a job whose process was never created — hook crash, `Popen` failure,
resource exhaustion — would burn its whole budget without executing once and
then be discarded as exhausted. That would have made the durable queue lose
exactly the work it exists to protect. Counting real failures keeps the ceiling
meaningful while an un-spawned job retains its full budget.

The second round of review caught the matching gap: counting only non-zero
*returns* left the most likely failure mode — an exception raised before any
return — with a budget that never moved, so the job would be respawned forever.
The accounting now wraps the whole run, records the attempt, and re-raises so
the traceback and non-zero exit are preserved.

Duplicate spawns are suppressed by a **soft lease**. `SessionEnd` and
`PreCompact` can fire back to back for the same session, and the queue lock only
guards the scan, so both could previously launch a worker for one job. A
handed-out job records `spawned_at` and is skipped while the lease is live. This
is deliberately not an ownership transfer: the marker is never renamed or
hidden, an expired lease always re-offers the job, and a confirmed failure
releases the lease immediately. The cost of a worker that never started is a
delayed retry, never a lost one — the failure mode a claim-and-hide design would
have reintroduced.

### PR04 — The Chroma migration marker was not durable (P2)

IR05 promised crash-resume, but the marker used `write_text()` with no fsync
while the artifact moves could become durable first. A crash in that window
leaves a mixed layout with no marker, which the same function then refuses as
ambiguous — wedging the store permanently.

**Fix:** `_write_marker_durably()` fsyncs the marker before the first move.

### PR05 — A legitimate no-op compile could be rejected forever (P2)

F01's gate accepted a run only if a `concepts/`/`connections/` article changed
*and* carried the source anchor. A daily log already fully covered by existing
articles legitimately produces no article change, so its hash was never
recorded: it recompiled on every run at LLM cost while the command stayed
permanently red.

**Fix:** a credited `knowledge/log.md` entry also counts as observable
knowledge — both forms still require the source to be *named*, so a failed CLI
that only printed an error satisfies neither. `MAX_COMPILE_ATTEMPTS` and
`failed_daily` bound the retries, mirroring `ingest.record_failure`.

The log credit is deliberately narrow, after the reviewer showed a looser
version could false-accept. Only a **strict append** counts: if `log.md` was
rewritten or truncated the result is rejected outright, so a pre-existing entry
from an earlier run cannot satisfy the gate. The appended text must also carry
the instructed compile heading (`compile | <file>`, with or without the
`daily/` prefix). An earlier revision also accepted a lone
`- Source: daily/<file>` line; the reviewer showed that line can appear in
other kinds of entry, so it no longer suffices on its own.

### PR09 — Set-aside sources were reported as "up to date" (P2)

Once every remaining source was set aside by PR05's ceiling, the command
printed "Nothing to compile - all daily logs are up to date" and exited 0 — a
never-compiled daily log silently reported as success.

**Fix:** exhausted sources are named on stderr and the command returns 1 while
any exist. They are still not re-attempted, so the repeated LLM cost PR05
removed does not come back. This is honest rather than permanently red: with
the gate broadened, a source only reaches the ceiling after three genuine
failures, so a red exit now means something is really wrong.

### PR06 — A swallowed daily-index failure left no repair record (P3)

The per-date lock gives mutual exclusion, not atomicity, so the Markdown can
land while the embed fails. The failure was logged and discarded and the cursor
advanced, leaving the daily search index quietly stale. Not a zero-loss
violation — the raw archive is authoritative and separately indexed — but the
comment claimed a transaction it did not provide.

**Fix:** failures record `stale_daily_index` in state for `reindex.py --daily`
to repair, success clears it, and the comment now states the real guarantee.
The clear path reads before writing so the hot path takes no extra locked write.

### PR07 — Distinct session IDs could share one archive (P3)

IR13 hardened pending-job filenames against untrusted session IDs but
`archive_transcript()` still collapsed every unsafe character to `_`, so two
distinct IDs mapped to one archive and the second was rejected as divergent —
unable to be archived at all.

**Fix:** `_archive_stem()` passes safe IDs through unchanged (so all seven
existing archives keep their names) and appends a SHA-256 digest otherwise.

### PR08 — Bounded extraction ignored its own ceiling for the first turn (P3)

The overflow check only applied once a turn had been admitted, so a limit
smaller than the first turn returned that whole turn *and* advanced the cursor
past it. Unreachable in production — both defaults are `None` and no caller
passes a limit — but it contradicted the documented bounded-batch contract.

**Fix:** the ceiling is checked before admitting any turn, including the first.

### Post-release verification evidence

- Independent rerun of the shipped suite before any change: **850 passed**.
- Red baseline for the new regressions: **15 failed, 3 passed** (the 3 are
  guard tests pinning behavior that must not change).
- Full suite after the first round of fixes: **868 passed**.
- After reconciling the first adversarial review of those fixes: **874 passed**.
- After reconciling the second: **879 passed in 66.30s**. No test was weakened
  or skipped to accommodate an implementation.
- Two tests were rewritten because the *implementation* changed underneath
  them, not to make a failure go away: both had pinned the hand-out attempt
  claim that the review overturned. They now assert the replacement invariants
  (a hand-out never consumes budget; an expired lease always re-offers the job).
- One regression surfaced during the work and was fixed at the source rather
  than in the test: bookkeeping on the daily-append success path added a second
  global `fsync`, breaking `test_daily_append_is_fsynced_before_success`. The
  clear path now reads before writing, which removes a needless locked write
  from the hot path and restores the original assertion untouched.
- `python -m compileall -q scripts hooks install.py`: passed.
- Real-behavior smoke runs (no mocks, all three rounds): archive
  fsync/idempotence/growth-append and divergence rejection; corrupt-marker
  quarantine with a valid neighbour still served; a loaded job never consuming
  budget; the lease suppressing an immediate duplicate while expiry reclaims the
  job; a confirmed failure counting and releasing the lease at once; the ceiling
  reached only through real worker failures, with the retired job's payload
  still intact in its `.exhausted` marker; `created_at` ordering beating
  filename order; context recovery from a self-contained marker; end-to-end
  legacy Chroma migration preserving `.recovery-*` directories and cleaning its
  marker; and the **live 545 MB store** resolving as a no-op with no stray
  marker.
- All seven existing transcript archives are UUID-named, so PR07 orphans none.

One earlier claim was checked and found overstated rather than wrong: the store
runs `journal_mode=delete`, not WAL, so migration has no `-wal`/`-shm` sidecar
to drop — only a transient hot journal, which is a far narrower window than a
WAL would be.
