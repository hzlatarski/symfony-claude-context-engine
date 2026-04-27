# Changelog

All notable changes to the Claude Context Engine — Symfony Edition are tracked here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The version recorded in `VERSION` at the repo root is the source of truth. The `check_update.py` helper compares it against `https://raw.githubusercontent.com/hzlatarski/symfony-claude-context-engine/main/VERSION` to surface upgrade prompts.

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
