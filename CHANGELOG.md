# Changelog

All notable changes to the Claude Context Engine — Symfony Edition are tracked here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The version recorded in `VERSION` at the repo root is the source of truth. The `check_update.py` helper compares it against `https://raw.githubusercontent.com/hzlatarski/symfony-claude-context-engine/main/VERSION` to surface upgrade prompts.

## [0.8.0] — 2026-07-31

Two things the new retrieval benchmark turned up while it was being built: a data-loss bug that had been destroying the ingest history on every run, and a measured improvement to vector retrieval.

### Fixed

- **`ingest.py` truncated `state.json` to `{}` on every run.** Its migration mutator called `migrate_state_schema(current)` — which mutates in place and returns *the same object* — then `current.clear()`, which emptied the "migrated" copy too, then updated from the now-empty dict. The damage compounds: with the ingest history gone the next run re-queues all 497 sources at LLM cost, and that run wipes it again. It destroyed 496 source records and 83 daily records twice in one week before being found. After the fix the queue is 69 files rather than 499, because the skip logic can finally see the history.
- **The test suite could write to the developer's live `state.json`.** Two tests exercised production code paths without redirecting it. `tests/conftest.py` now redirects `STATE_FILE` to a tmp file for every test by default, and a backstop fails any test where a top-level key shrank or vanished — restoring the previous content first. The backstop checks shrinkage rather than byte equality on purpose: a legitimate background ingest may be appending while the suite runs, and a byte comparison would both fail spuriously and restore a stale snapshot over that writer's progress.

### Added

- **Chunked embedding (`--chunk-words`, `--chunk-overlap`).** `all-MiniLM-L6-v2` truncates at roughly 190 words while LongMemEval sessions run a median of 1633, so a whole-document embedding only ever represents the opening. Chunking splits documents into overlapping windows and ranks a document by its best-matching window.

  Hybrid, first 150 questions, same subset both rows:

  | | recall@5 | recall@10 | ndcg@10 | rr@20 | runtime |
  |---|---|---|---|---|---|
  | whole-session | 0.9467 | 0.9867 | 0.8512 | 0.8353 | ~285s |
  | chunked 150w/30o | **0.9600** | **1.0000** | **0.8926** | **0.8904** | 2790s |

  Better on every metric, and recall@10 is perfect. The gain is largest on the *ranking* metrics rather than raw recall, which is what you would expect: chunking mostly helps documents the vector stream already reached but ranked poorly, by letting the matching passage speak for the document instead of its first paragraph. The cost is roughly 10× the runtime. Chunking applies to the vector index only — BM25 reads whole documents and has no window limit — with a test pinning that its score is unchanged.

### Notes

- Stopping a background shell does not necessarily kill the Python process it launched. Two orphaned `ingest.py` runs survived their wrappers here and kept rewriting `state.json` for hours, one of them from before the migrate fix — which is why the loss appeared to recur after each restore. Check for surviving `python.exe` children before concluding a fix did not work.

## [0.7.0] — 2026-07-31

Retrieval quality is now measured rather than argued about. Until this release every ranking knob — the RRF constant, the pool multiplier, the tokenizer, the embedding model — was set by reasoning alone; the 950-test suite checked behavior but never asked whether search actually finds the right thing. The new harness scores our real ranking path against LongMemEval-S (500 questions, ~50 sessions per haystack) and reports recall@k, NDCG@10 and reciprocal rank.

Baseline on all 500 questions: **bm25 0.9640 / vector 0.9240 / hybrid 0.9680 recall@5**. Hybrid wins every recall cutoff, but bm25 alone edges it on both NDCG@10 and rr@20 — fusion buys coverage, not precision, at roughly 60× the runtime. Sanity controls put a random ranking at 0.180 against a 0.191 chance rate.

Building it surfaced two defects in live code that the existing suite could not see, and an adversarial review round caught nine more in the harness itself before any number was trusted.

### Added

- **`scripts/eval_metrics.py`, `scripts/eval_corpus.py`, `scripts/eval_longmemeval.py`** — the benchmark: metrics over ranked id lists, an ephemeral per-question index, and the LongMemEval-S loader/scorer/CLI. `--mode bm25` needs no embeddings and scores all 500 in ~16s, which makes it a practical regression check after any tokenizer or ranking change.
- **`docs/RETRIEVAL-BENCHMARK.md`** — how to run it, the baseline table, what the harness deliberately does *not* measure, and the two traps below written up so they do not recur.
- **`bm25_store.TokenIndex` and `hybrid_search.fuse_rankings`** — extracted from previously inlined code so the benchmark shares one implementation with live search instead of copying it. A harness that copies the ranking code measures the copy.

### Fixed

- **The BM25 index was built from the wrong string.** `_iter_article_zones` deliberately folds the article title into the token stream while the stored `text` omits it. Rebuilding the index from `text` silently dropped every title term from live knowledge-base search — and the entire suite still passed, because every existing fixture happens to repeat title words in its body. Records now carry an explicit `index_text`.
- **`load_state` handed writers a dict with missing keys.** It returned raw JSON with no normalization, so a state file that had lost a key crashed its writer — `ingest.py` raised `KeyError` only *after* re-processing all 497 source files, discarding the whole run. Observed live on 2026-07-30 with `ingested_sources` gone entirely and 496 records unrecoverable from the file. Defaults are now backfilled *and* a warning is printed, because silence is what let the loss go unnoticed for days.
- **Blank documents competed for vector ranks.** They were embedded behind a placeholder on the assumption that a document with no content "cannot match anything" — false for nearest-neighbour search, which always returns a ranking. LongMemEval-S contains 1228 zero-turn sessions, about 5% of the indexed corpus. They are now excluded from the vector index and counted in the report.
- **`fuse_rankings` let a single stream vote twice** for a repeated id, letting one retriever's duplicate outrank a document both streams agreed on. RRF assumes one rank per document per ranking.
- **Reciprocal rank was truncated but labelled as MRR.** Results are only materialized to `depth`, so gold below it contributed 0. Reported as `rr@<depth>` now, and not comparable to published full-ranking MRR.
- **`aggregate()` skipped absent metric keys**, dividing by a smaller denominator than the question count it reported — a partial or resumed run would have looked complete, and better than it was. Ragged input is now rejected.
- **NDCG could exceed 1.0** when a ranking repeated a gold id, and a negative cutoff silently meant "all but the last N" rather than failing. Empty questions were accepted, becoming an arbitrary nearest-neighbour lookup under vector search.

### Notes

- `chromadb.EphemeralClient()` is **not** isolated — it resolves through a shared in-process system client, so two clients requesting the same collection name get the same collection. A fixed name merged all 500 haystacks into one growing pool and reported vector recall of 0.354 instead of 0.924. Anything building an ephemeral Chroma index must use a unique collection name and delete it; `TestIsolationBetweenCorpora` guards this.
- Chunking sessions to 150-word windows before embedding lifts vector-only from 0.820 to 0.940 recall@5 on a 50-question subset — the largest single improvement measured so far, and a candidate for a real mode.
- The LongMemEval-S dataset is not vendored (278 MB); the benchmark takes a `--dataset` path.

## [0.6.1] — 2026-07-30

Hardening from a post-release verification pass over 0.6.0. That release's own suite reproduced cleanly (850 passed) and its core zero-loss mechanism verified correct — both extraction limits are `None`, the cursor advances only through included turns, and raw-index failure still blocks it. But nine claims in its two design documents were not fully satisfied by the code. Each is now fixed with a regression test; the suite is 879 passed.

The fixes were themselves put through two adversarial review rounds, which caught a self-inflicted data-loss risk in the first retry-ceiling design and a false-accept in the first compile gate. Both were corrected before release; `MEMORY-COMPILER-REVIEW-AND-REMEDIATION-PLAN.md` records the reasoning inline (PR01–PR09).

### Fixed

- **The authoritative transcript archive is now fsynced before publication.** 0.6.0 fsynced the daily log and the cursor but published the archive with a plain write plus rename — so a power loss could leave a cursor that had consumed turns the archive no longer contained. The one file the design treats as authoritative was the only one without the guarantee.
- **A corrupt pending-flush marker no longer halts every future capture.** The loader raised on any unusable record and both hooks spawn nothing when it does, so a single bad file stopped all raw indexing and summarization for every session indefinitely. Unusable records are quarantined as `.corrupt` and the queue keeps draining.
- **Flush retries are bounded.** `MAX_FLUSH_ATTEMPTS` retires a hopeless job as `.exhausted` instead of letting every capture hook relaunch it forever, and jobs carry an explicit `created_at` so the documented oldest-first order is real rather than a sort over random UUIDs. Attempts are counted by the worker that actually ran and failed — on a non-zero exit or an unhandled exception — never when the hook hands the job out. A job whose process was never created keeps its full budget, so a hook crash or a failed spawn cannot silently exhaust and discard work that never ran once.
- **Duplicate flush workers are suppressed by a soft lease.** `SessionEnd` and `PreCompact` firing back to back for one session could each launch a worker for the same job. A handed-out job now records `spawned_at` and is skipped while the lease is live. The marker is never renamed or hidden, so an expired lease always re-offers the job and a confirmed failure releases it immediately — a worker that never started costs a delayed retry, never a lost one.
- **A transient I/O error no longer looks like corruption.** An unreadable pending marker is left in place for the next run, and a failed quarantine rename never falls back to deleting it. The marker is the only record of the session, cursor, and archive path, so destroying it on a sharing violation would lose the job outright.
- **Set-aside daily logs are no longer reported as "up to date".** Once every remaining source hit the retry ceiling, the compiler printed a success message and exited 0 for logs it had never compiled. They are now named on stderr and the command exits non-zero while any exist — without re-attempting them, so the repeated LLM cost stays gone.
- **The Chroma active-store migration marker is fsynced.** Without it a crash could make the artifact moves durable while the marker was still cached, leaving a mixed layout that the next run refuses as ambiguous — permanently wedging the store the crash-resume logic existed to recover.
- **A legitimate no-op compile no longer retries forever.** The mutation gate accepted only a source-anchored article change, so a daily log already covered by existing articles never recorded its hash: it recompiled every run at LLM cost while the command stayed permanently red. A credited `knowledge/log.md` entry now also counts, and `MAX_COMPILE_ATTEMPTS` bounds retries the way `ingest.py` already did.
- **A failed daily embed leaves a repair record.** The per-date lock provides mutual exclusion, not atomicity, so the Markdown can land while indexing fails. That stays non-fatal by design, but it now records `stale_daily_index` for `reindex.py --daily` instead of only writing a log line.
- **Two distinct session IDs can no longer share one archive.** Unsafe IDs were collapsed character-by-character, so a second session mapping to the same name was rejected as divergent and could never be archived. Safe IDs (Claude's UUIDs) still pass through verbatim, so no existing archive is renamed.
- **Bounded transcript extraction honors its ceiling for the first turn.** A limit smaller than the first turn returned that whole turn and advanced the cursor past it. Unreachable in production — no caller passes a limit — but it contradicted the documented contract.
- **The session-start update check no longer depends on bare `uv`.** F11 removed that dependency from the capture path but not from here, where the failure is swallowed, so update checks simply stopped happening in restricted shells.

## [0.6.0] — 2026-07-28

Sessions can now be recovered and searched without relying on a fixed-size summary window. The original transcript remains authoritative, while summaries continue to provide a readable daily history.

### Added

- **Zero-loss transcript archive and search.** `PreCompact` and `SessionEnd` archive the original Claude JSONL byte-for-byte and independently index message text, tool inputs, tool results, URLs, identifiers, and exact references.
- **Historical transcript backfill.** `scripts/backfill_transcripts.py` discovers and indexes previous Claude sessions, with support for selecting individual session IDs.
- **Durable flush retries.** Detached flush work is recorded before launch, survives process and power failures, and is retried without advancing the transcript cursor until every required stage succeeds.

### Changed

- **Transcript extraction no longer defaults to 30 turns.** Fresh turns are consumed oldest-first without a fixed limit; explicitly bounded windows advance only through complete included turns.
- **Chroma storage uses a guarded active layout.** Legacy stores migrate under a lock with crash-resume markers, while ambiguous mixed layouts fail instead of silently selecting incomplete data.
- **Raw retrieval exposes complete evidence.** Search results can be expanded through `get_raw_daily_chunk` to retrieve the full unclipped indexed record.

### Fixed

- Prevented cursor advancement when transcript archiving, raw indexing, summary persistence, or durable cursor storage fails.
- Replaced delete-before-write vector updates with staged source replacement for code, articles, daily logs, and transcripts, preserving the last searchable generation on failure.
- Serialized compiler, state, daily-log, and per-session writers so concurrent hooks and reindex processes cannot clobber each other.
- Made ingestion, compilation, Whisper, upgrade, and child-process commands report non-zero exits honestly instead of recording failed work as successful.
- Reconciled deleted and emptied sources so stale code, article, daily, and transcript chunks no longer remain searchable.
- Blocked viewer and MCP path traversal, disabled executable raw Markdown HTML, and added a restrictive nonce-based Content Security Policy.

### Tests

- Added regression coverage for transcript completeness, archive integrity, cursor concurrency, durable retries, staged replacement, migration recovery, stale-source reconciliation, subprocess failures, traversal, and stored XSS. The complete suite passes 850 tests.

## [0.5.0] — 2026-07-12

Fixes for defects found while comparing this engine against [TencentDB Agent Memory](https://github.com/TencentCloud/TencentDB-Agent-Memory), plus the duplicate-detection mechanism it had and this one lacked. Its layered-memory half is largely what this engine already does; its context-offload half needs host hooks that rewrite the live message array, which Claude Code does not expose. What transferred is the *principle* that a lossy summary must ship a handle back to the full text — applied here to compiled-truth excerpts.

### Fixed

- **compiled-truth had collapsed to 8 of 678 articles.** Entries were full article bodies, and at a ~4,000-char mean a 40,000-char budget fit almost nothing — the priority scoring was choosing between a handful of verbose articles instead of ranking the knowledge base. Entries are now capped to a 1,200-char excerpt (`MAX_ARTICLE_CHARS`, `--max-article-chars 0` to disable) carrying a `get_article(slug)` pointer to the full text. 8 → 37 articles in the file; ~1.5 → 18 in the session-start injection.
- **The session-start heading lied.** It read "Compiled Truth (all current knowledge)" while injecting the top 25%, so the agent believed it had the whole KB and skipped `search_knowledge`. Heading and truncation notice now state what is and isn't included.
- **Duplicate and per-turn flushes.** `settings.local.json` bound the flush hook to `Stop` (fires every assistant turn) and duplicated the `PreCompact` binding. Both removed; `settings.json` already wired `SessionEnd` and `PreCompact` correctly.
- **Overlapping flush windows.** Every flush re-read the last 30 turns of the whole transcript, so PreCompact and SessionEnd re-summarized the same turns into duplicate daily-log entries. New per-session turn cursor (`flush_cursor.py`) advances only after a flush succeeds, so a failed flush retries its window instead of losing it.
- **Flush sentinels were substring-matched.** A real summary that merely *mentioned* `FLUSH_OK` was discarded and replaced with "Nothing worth saving" — and sessions working on this compiler mention it constantly. Now matched as a prefix of the stripped response.
- **`compile.py`'s prompt said "one of the six"** memory types while listing all eight, contradicting `config.MEMORY_TYPES`. A drift-guard test now fails if the prompts and the config taxonomy disagree.

### Added

- **`dedup.py` — near-duplicate detection.** All-pairs cosine sweep over the article vectors already in Chroma; zero LLM, zero network, zero cost. The compiler prompt had always *asked* the LLM not to create near-duplicates; nothing ever checked. It had not been working: the live KB held byte-identical articles under two slugs, three HeyGen v3 articles, and three CSRF articles. Surfaced as the `near_duplicate` lint check. Deliberately does not auto-merge — choosing which facts survive a merge is not a decision to automate.
- **Duplicate prevention at compile time.** `compile.py` and `ingest.py` now name the existing articles a source is semantically closest to ("prefer UPDATING one of these"), instead of telling the model to scan a 678-line index. The source is chunked before embedding, since the embedder truncates at 256 tokens and would otherwise see only the first session of a daily log.
- **Stale-vector detection and pruning.** Deleting or renaming an article left its embedding behind, so `search_knowledge` could return a slug that `get_article` cannot open. `dedup.py --prune-stale` removes them; surfaced as the `stale_vector` lint check. Six were found and pruned in this repo.

## [0.4.1] — 2026-07-11

### Fixed

- **`upgrade.py` no longer exits non-zero on Windows.** The final "Upgrade complete" line and the progress lines use non-ASCII glyphs (arrow, ellipsis). On a Windows cp1252 console these raised `UnicodeEncodeError` on the closing print — *after* the upgrade had already fully succeeded — corrupting the exit code to 1 and tripping the skill's failure path (and any auto-upgrade wrapper that checks the return code). `sys.stdout`/`stderr` are now reconfigured to UTF-8 with `errors="replace"` at startup (guarded for non-reconfigurable streams), so glyphs render on capable terminals and un-encodable ones degrade to a replacement char instead of crashing.


## [0.4.0] — 2026-07-11

Four improvements adapted from [Graphify](https://github.com/Graphify-Labs/graphify)'s code-intelligence graph — the ideas it had that this engine lacked, filtered to what fits a curated + retrieval hybrid (its "no embeddings" thesis and multi-LLM backends were deliberately *not* adopted). All four are pure-Python and add zero LLM cost.

### Added

- **Retrieval-outcome feedback loop** — `scripts/retrieval_feedback.py` + `record_retrieval_outcome` MCP tool (knowledge server) + `scripts/reflect.py`. The one signal time-based confidence decay can't capture: *was an article actually useful when retrieved?* The agent marks a retrieval `useful` / `dead_end` / `corrected`; outcomes are recency-weighted (45-day half-life) and feed a new `WEIGHT_FEEDBACK` axis in `compile_truth.py` priority scoring. Backward-compatible: an unrated article scores a neutral `0.5`, so the relative ranking of never-rated articles is unchanged. `reflect.py` aggregates the store into `knowledge/LESSONS.md` (trusted vs. contested articles); `kb_health` gains a `feedback` block surfacing the most-contested articles for review. Graphify's `save-result` / `reflect` pattern.
- **Inline rationale nodes** — the call-graph parser (`parsers/call_graph.py`) now lifts design-intent comments (`// WHY:`, `// HACK`, `// TODO`, `// FIXME`, `/** @deprecated */`, and 5 more tags) out of `src/**/*.php` and attaches them to the method or class they annotate. `unified_graph.py` materializes them as `note:<file>:<line>` leaf nodes with `annotates` edges (degree-1, so they cluster into their owner's Leiden community without adding hub noise). Surfaced via a new `find_rationale(tag, query)` MCP tool and an **Inline Rationale** section in `get_file_deps`. Answers "where are the known hacks / deprecations / TODOs?" and gives `trace_route` the *why* next to the *what*.
- **`trace_path(from_node, to_node)` MCP tool** (code-intel server) — shortest connection between any two unified-graph nodes via BFS over the undirected projection, with each hop annotated by edge kind / relation / confidence. Answers "how is this controller symbol related to that article?" — complements `get_unified_neighbors` (radius around one node) by returning the actual chain *between* two.
- **`merge_order_risk(base, branches)` MCP tool** (code-intel server) + `scripts/merge_risk.py`. For this project's chronically-dirty-`main` + many-worktrees reality: auto-detects branches ahead of `base` (skipping `backup/`/`archive/`/`wip/` noise), diffs each, maps changed files into unified-graph communities, and reports **direct file conflicts** (same file on 2+ branches — git *will* conflict) and **community-overlap risk** (different files in the same semantic cluster — a coupling risk plain git can't see). Detects **stacked branches** via `git merge-base --is-ancestor` so a child restating its parent's files isn't a false conflict. Graphify's `prs --conflicts` signal.

### Changed — richer, more automatic context surface

Prompted by "is the compiler actually being called enough, and is what it returns useful?":

- **`trace_route` collapses entity accessor noise** — childless getter/setter calls to `\Entity\` classes (`User::setEmail`, `getId`, …) folded into one summary line per parent, so the architecturally-meaningful service/repository hops aren't buried. New `collapse_accessors` param (default True) on the tool.
- **Auto-injection hook now surfaces the knowledge base** — `hooks/user-prompt-submit.py` previously injected code-intel *only* when the prompt named a concrete file/route/class; a conceptual "why/decision/how" prompt got nothing. It now also runs a `search_knowledge` (hybrid) when the prompt matches conceptual triggers and injects the top curated-KB matches — closing the gap where the KB was never auto-surfaced.
- **Auto-injection hook adds related code** — when a file is named it now also injects semantically-related code (`search_codebase`), excluding the named file and sibling-worktree/vendor duplicates. `get_file_deps`' inline-rationale section now parses just the one named file (a ~ms tree-sitter pass) instead of the whole call graph, so the per-prompt hook stays cheap. (`impact_of_change` was intentionally left OUT of the hook — in a fresh cold process it forces a whole-graph parse for niche value; call it on-demand instead.)

### Tests

- ~55 new tests (`test_retrieval_feedback.py`, `test_rationale.py`, `test_trace_path.py`, `test_merge_risk.py`, `test_trace_route_collapse.py`, `test_prompt_hook_triggers.py`) plus a rationale PHP fixture. `test_integration.py`'s file-deps timing test now warms the call-graph cache (a new `get_file_deps` dependency). All pure-Python (no network/Chroma).

### Notes

Every finding from an adversarial review of this release was fixed before merge: summary-first `@deprecated` docblocks (per-line scan), docblock attribution from the comment's end line, uppercase-only bare tags (no "Note that…" false positives), proportional weight rebalance (unrated-article ranking provably preserved), corrupt-feedback-store backup instead of clobber, stacked-branch ancestry, and the `get_file_deps` cold-process latency regression (disk-cached parse).

## [0.3.1] — 2026-06-17

Three features cherry-picked from the [obsidian-wiki](https://github.com/Ar9av/obsidian-wiki) LLM-wiki framework — the ideas it had that this engine lacked. All three are pure-Python and add zero LLM cost. Plus a Windows quality-of-life fix for the flashing `claude` console window.

### Added

- **Multi-agent history mining** — `scripts/import_agent_history.py` + `scripts/agent_adapters/` (registry mirroring `source_handlers`). Reads Codex (`~/.codex/sessions`) and other Claude Code projects (`~/.claude/projects`) transcript stores, normalizes each session to ingestible markdown under `knowledge/imported/<agent>/`, and feeds the existing `ingest.py` pipeline. Project-scoped by `cwd`, idempotent (content-hash skip), `--dry-run`/`--since`/`--limit`/`--project all` flags. Closes the blind spot where work done through agents other than this project's Claude Code never reached the KB. The adapter registry is the extension point for more agents (Hermes deferred — format unconfirmed). Adds the `imported-agent-history` source group to `sources.yaml.example`.
- **Wikilink backfill** — `scripts/crosslink.py`. Scans articles for unlinked prose mentions of other articles' titles/aliases and appends `[[wikilinks]]` to `### Related Concepts` (graph-safe against the path-based wikilink format; no inline prose rewriting). Conservative whole-word case-insensitive matching, masks frontmatter/code/existing links, dry-run by default (`--apply` to write). Densifies the graph feeding `compiled-truth.md` scoring, `get_unified_neighbors`, and Leiden communities.
- **Graph exports** — `scripts/export_graph.py`. Serializes the unified knowledge graph to GraphML (Gephi/yEd), Neo4j Cypher, a self-contained interactive HTML viewer (vis-network via CDN), or raw JSON. CLI `--format`/`--out`; default output under `knowledge/exports/`.

### Fixed

- **Flashing `claude` console window on Windows** — `flush.py`, `compile.py`, `ingest.py`, and the whisper spawners (`enhance.py`, `expand_query.py`) launched the `claude` CLI via `subprocess` without `CREATE_NO_WINDOW`, so each spawn allocated a visible console window (most visibly on every session-end flush). The Python launcher in `session-end.py`/`pre-compact.py` already suppressed its own window, but the grandchild CLI did not inherit it. All five subprocess spawners now pass a shared `NO_WINDOW_CREATIONFLAGS` (defined in `config.py`; `CREATE_NO_WINDOW` on win32, `0` no-op elsewhere). The Agent-SDK spawners (`canary.py`, `lint.py`, `query.py`) are unaffected by this fix — they spawn through the SDK, which does not expose `creationflags`.

### Tests

- 53 new tests (`test_import_agent_history.py` ·12, `test_crosslink.py` ·23, `test_export_graph.py` ·18), all pure-Python (no network/Chroma). Full suite otherwise unchanged.

## [0.3.0] — 2026-06-16

### Changed

- feat(release): add one-command release helper (scripts/release.py)

## [0.2.0] — 2026-06-16

Per-task code-intelligence context. The knowledge base already had a strong "USE FIRST" directive + deferred-tool unlock at session start; the code-intel MCP did not, so its tools were rarely reached for. This release closes that asymmetry and adds automatic, per-prompt context injection.

### Added

- **UserPromptSubmit auto-context hook** — `hooks/user-prompt-submit.py`. Regex-detects file paths, Symfony routes (`GET /x`), PascalCase classes, and Stimulus controllers in the user's prompt, resolves them to repo paths, and injects the matching code-intel builder output (`get_file_deps` / `trace_route`) under an "Auto-fetched code intelligence" heading. The expensive `mcp_server` import only happens on a match, so conversational prompts pay only the regex cost (~0). All failures degrade to empty context — the hook never blocks or breaks a turn. Wire it in the host project's `.claude/settings.json` under a `UserPromptSubmit` hook.

### Changed

- **`hooks/session-start.py`** — now injects a "Use Code Intelligence before touching code" block, the structural twin of the existing KB directive: its own one-call `ToolSearch` unlock for the code-intel tools plus a trigger table (`get_file_deps` before editing, `trace_route` for request flow, `impact_of_change` before merging, `get_template_graph` before Twig changes).

### Notes

- The auto-context hook is opt-in per project: add the `UserPromptSubmit` entry to that project's `.claude/settings.json`. The session-start directive needs no wiring — it ships inside the existing session-start hook output.

## [0.1.0] — 2026-04-27

First versioned release. Establishes the upgrade contract (`VERSION` + `check_update.py` + `upgrade.py` + `/memory-compiler-upgrade` skill) and ships the SocratiCode-inspired capability port.

### Added

- **Live file watcher** — `scripts/watch.py` debounces filesystem events by 2s and routes them to `reindex_articles` / `reindex_daily` / `index_codebase.reindex_single`.
- **Cross-project linked search** — `MEMORY_COMPILER_LINKED_PROJECTS` env var. `search_knowledge(..., include_linked=True)` fans out via vector search to sibling project Chroma stores and RRF-merges results, tagging each hit with a `project` label.
- **Cross-process locking** — `chroma_lock.chroma_write_lock()` (filelock-backed) wraps every Chroma upsert/delete. One lock per collection under `knowledge/chroma/.locks/`. 60s default acquisition timeout via `CHROMA_LOCK_TIMEOUT_SECONDS`.
- **Resumable, interruptible ingest** — `ingest_state` module writes atomic per-file status snapshots and polls a stop flag at every file boundary.
- **`list_sources` MCP tool** — surfaces the source-group catalog with descriptions and chunk counts (analogous to SocratiCode's `codebase_context`).
- **`kb_health` MCP tool** — one-shot diagnostic combining vector store sizes, articles by memory type, broken `[src:]` anchor count, quarantine count, and last-ingest timestamp.
- **`ingest_status` / `ingest_stop` MCP tools** — live progress + cooperative cancellation for the ingest pipeline.
- **`get_circular_dependencies` MCP tool** — iterative Tarjan SCC over the resolved call graph; supports `scope='all' | 'php' | 'js' | 'vendor-excluded'`.
- **Mermaid output** — `trace_route` and `impact_of_change` accept `output_format='mermaid'` for flowchart rendering.
- **Source-group descriptions** — `SOURCE_PATTERNS` carries an LLM-facing description per group; surfaced as `source_description` on every `search_codebase` result.
- **`MEMORY_COMPILER_EXTRA_EXTENSIONS`** env var — adds custom file extensions to the codebase indexer (scoped to `src/`, `assets/`, `templates/`, `config/` to avoid walking `vendor/`).
- **Upgrade mechanism** — `VERSION` file, `scripts/check_update.py` (cached + snooze-aware version probe against the remote), `scripts/upgrade.py` (`git fetch` + `reset --hard` + `install.py` rerun), `~/.claude/skills/memory-compiler-upgrade/SKILL.md` for the user-facing prompt.

### Changed

- `search_knowledge` accepts `include_linked: bool` and forwards to the linked-search path when set.
- `index_codebase.SOURCE_PATTERNS` now stores `(file_type, [globs], description)` tuples and is extended at runtime with `MEMORY_COMPILER_EXTRA_EXTENSIONS` entries.
- `hooks/session-start.py` injects an `## Update Available` block when `check_update.py` reports `UPGRADE_AVAILABLE`. The block tells the agent to run `/memory-compiler-upgrade` to handle the prompt.

### Notes

- All linked projects must use the default Chroma embedder (bundled ONNX MiniLM). The cross-project fan-out is vector-only — cross-process BM25 indexes are intentionally not exposed.
- `kb_health` against the AiTutor knowledge base surfaced 1573 broken `[src:]` anchors and 89 articles missing a valid `type:` value at first run — real signal, not noise.
- Validation: 359/359 non-whisper tests pass. The two whisper failures predate this release.
