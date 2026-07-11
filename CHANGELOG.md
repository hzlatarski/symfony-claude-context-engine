# Changelog

All notable changes to the Claude Context Engine — Symfony Edition are tracked here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The version recorded in `VERSION` at the repo root is the source of truth. The `check_update.py` helper compares it against `https://raw.githubusercontent.com/hzlatarski/symfony-claude-context-engine/main/VERSION` to surface upgrade prompts.

## [0.4.1] — 2026-07-11

### Changed

- fix(upgrade): reconfigure stdout/stderr to UTF-8 so upgrade never exits non-zero on Windows
- feat: Graphify-inspired improvements + richer auto-injection (v0.4.0)
- fix(subprocess): resolve claude binary explicitly instead of trusting PATH

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
