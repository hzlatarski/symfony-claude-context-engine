# Claude Context Engine — Symfony Edition

**Your AI conversations, project docs, and codebase compile themselves into a searchable, self-healing knowledge base — with semantic retrieval, memory-type filtering, and drift detection baked in.**

A long-term memory system for Claude Code, purpose-built for Symfony projects. Session transcripts, design specs, and live codebase structure flow into a single knowledge store that Claude queries on demand through dedicated MCP tools. Unlike vector-only systems that store everything and hope for the best, this engine **compiles** raw conversations into structured, source-cited articles — and defends that structure against the drift failure modes typical LLM wikis suffer.

**Target stack:** Symfony 7.x, PHP 8.2+, Twig, Stimulus.js, AssetMapper.

**Lineage:** Forked from [coleam00/claude-memory-compiler](https://github.com/coleam00/claude-memory-compiler), itself inspired by [Andrej Karpathy's LLM Knowledge Base](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f). Substantially extended with a Symfony code-intelligence layer, anti-drift hardening, and a semantic retrieval surface inspired by [MemPalace](https://github.com/MemPalace/mempalace) and [thedotmack/claude-mem](https://github.com/thedotmack/claude-mem).

---

## Key Features

- 🖥 **[Web Viewer UI](#web-viewer)** — Read-only FastAPI dashboard at **<http://127.0.0.1:37778>**. One command (`uv run python scripts/viewer.py`), no build step, no auth. Browse articles, daily logs, tool drawer, contradictions, and cost history.
- 🎙 **[Voice-to-enhanced-prompt](#whisper-prompt)** — Mic page at **<http://127.0.0.1:37778/whisper>**. Speak a question; the pipeline transcribes it locally (faster-whisper, CPU), expands it via Haiku query-expansion, retrieves grounding from the knowledge base, and rewrites it into a fully grounded Claude prompt. Three modes: verbatim, rewrite, clean.
- 🧠 **Curated memory, not a grep-pile** — Sessions compile into structured articles with Truth + Timeline format, `[[wikilinks]]`, and `[src:path]` provenance anchors. Human-readable, diffable, Obsidian-compatible.
- 🏷 **Memory type taxonomy** — Every article is a `fact`, `event`, `discovery`, `preference`, `advice`, or `decision`. First-class filter in `search_knowledge` so "only preferences about testing" is one call.
- 🔍 **Hybrid BM25 + vector search** — Two dedicated MCP servers: `knowledge-compiler` for semantic+lexical retrieval, `symfony-code-intel` for live codebase structure. Fused via Reciprocal Rank Fusion.
- ⚡ **Token-efficient retrieval** — `search_knowledge` returns slim ~220-char snippets; `get_articles([slugs])` batch-fetches full bodies only for the winners. ~10× context savings on multi-hit queries.
- 🛠 **Symfony code intelligence** — Six live-parsing MCP tools: `get_codebase_overview`, `get_file_deps`, `get_route_map`, `get_template_graph`, `get_stimulus_map`, `get_hotspots`. Mtime-cached, sub-second.
- 📚 **Structured tool drawer** — `PostToolUse` hook writes a JSONL log of every tool call to `knowledge/daily/*.tools.jsonl`. `flush.py` reads it as ground-truth input for Haiku summaries — so daily logs cite real file paths and commands, not reconstructions.
- 🧪 **Anti-drift hardening** — Source anchors, confidence decay (90-day half-life), contradiction quarantine, canary questions, skeptical compile prompt, observed-vs-synthesized zones.
- 📉 **O(1) prompt cost** — Priority-scored `compiled-truth.md` keeps session-start context constant from 50 articles to 5,000. Upstream compiler scales linearly.
- 🆓 **Local embeddings, zero API cost** — Bundled `all-MiniLM-L6-v2` ONNX model, ~90 MB one-time download. Chroma runs fully offline.
- 💾 **Knowledge in git** — Entire store is plain markdown under `knowledge/`. Check it into your repo; your team shares memory via normal `git pull`.

---

## Why This Exists

Every Claude Code session starts blank. You re-explain the same things — naming conventions, deployment rules, last week's architectural decisions — over and over. Generic "memory" plugins store raw transcripts and hope vector search surfaces the right chunks. That works, until it doesn't: no type filtering, no confidence, no way to tell a firm decision from a passing remark, no way to know when the system has started hallucinating.

This engine takes a different bet. It **compiles** your sessions into structured, human-readable articles with explicit provenance — then layers **semantic retrieval, confidence decay, and drift canaries** on top of that curated foundation. You get the recall benefits of a vector store *and* the reliability of a hand-curated wiki, without hand-curating anything.

---

## What Makes It Valuable

### 1. Two-Layer Knowledge Store

- **Curated articles** (`knowledge/concepts/`) — LLM-compiled Truth + Timeline format, one file per concept, with `[[wikilinks]]` forming a full knowledge graph.
- **Verbatim drawer** (`knowledge/daily/`) — raw session logs, never summarized, semantically indexed so you can always retrieve the exact words that led to a compiled claim.

### 2. Symfony Code Intelligence (Dedicated MCP Server)

Five pure-Python parsers expose your live codebase as structured data via the `symfony-code-intel` MCP server's six tools:

| Tool | What it returns |
|---|---|
| `get_codebase_overview` | File counts by type, routes, templates, Stimulus orphans, top churn hotspots |
| `get_file_deps(path)` | Imports, reverse deps, routes/templates touched, co-change partners |
| `get_route_map(prefix)` | Route → controller → action → template → injected services table |
| `get_template_graph(t)` | Twig inheritance, includes, Stimulus bindings |
| `get_stimulus_map(c)` | Bidirectional JS ↔ Twig map with orphan detection |
| `get_hotspots(top_n)` | Churn-ranked files with ownership and bus-factor scoring |

Runs live, mtime-cached, under one second. Git intelligence caches to `knowledge/git-intel.json` and invalidates on HEAD change.

### 3. Knowledge MCP Server (Semantic Retrieval)

A **second** MCP server (`knowledge-compiler`) — separate from the code-intel one on purpose — exposes the knowledge store:

| Tool | What it returns |
|---|---|
| `search_knowledge(query, ...)` | Semantic search over curated articles with filters for memory `type`, `min_confidence`, `zone`, quarantine state. Returns **slim snippets** (~220 chars), not full bodies. |
| `search_raw_daily(query, date_from, date_to)` | Semantic search over verbatim drawer chunks (daily logs, never summarized). Slim snippets. |
| `get_article(slug)` | Full markdown + parsed frontmatter for one article |
| `get_articles([slugs])` | Batch-fetch full bodies for multiple slugs in one round trip. Missing slugs return `{slug, error: "not_found"}` so one bad slug doesn't abort the batch. |
| `list_contradictions()` | Current contradiction-quarantine list |

Backed by **ChromaDB** with the bundled `all-MiniLM-L6-v2` ONNX embedder — fully local, zero API cost, ~90 MB one-time model download on first use.

**Token-efficient two-step retrieval** — inspired by claude-mem's `search → get_observations` split, `search_knowledge` returns just `{slug, title, snippet, distance, metadata}` (~50–100 tokens per hit) so the agent can scan cheaply and then fetch full bodies only for the winners via `get_article(s)`. Saves ~10× context on multi-hit queries where most matches turn out to be irrelevant.

### 4. Anti-Drift Hardening (Six Defenses)

Typical LLM wikis decay: compilers hallucinate, facts contradict earlier facts, old claims rot without anyone noticing. This engine ships six concrete mitigations out of the box:

| Defense | What it does |
|---|---|
| **Source anchors** | Every Truth bullet carries a `[src:path]` anchor. `lint.py` verifies targets exist; broken anchors become errors. |
| **Confidence decay** | 90-day exponential half-life on the `confidence:` frontmatter field. Unvalidated old claims sink in priority until re-corroborated. |
| **Contradiction quarantine** | `lint.py` writes contradictions to `knowledge/contradictions.json`. Quarantined articles are excluded from `compiled-truth.md` **and** from `search_knowledge` results until `lint --resolve` clears them. |
| **Canary questions** | `canary.py` runs known-answer questions via Haiku and fails loudly if expected substrings stop appearing — early warning for compiler drift. |
| **Skeptical compile prompt** | `compile.py` compares new info against existing Truth on every update, flags contradictions with a `CONTRADICTION:` marker, and appends `### Conflict` subsections instead of silently overwriting. |
| **Observed / Synthesized zones** | `## Truth` splits into `### Observed` (direct extractions, low hallucination risk) and `### Synthesized` (compiler inferences, higher risk, opt-in via `compile_truth.py --synth`). |

### 5. Memory Type Taxonomy

Every article carries a `type:` — one of `fact`, `event`, `discovery`, `preference`, `advice`, `decision`. Used as a first-class filter in `search_knowledge` so you can ask the agent to surface "only preferences about testing" or "only decisions from the last sprint." Unknown values fail `lint.check_memory_types`.

### 6. Structured Tool Drawer (PostToolUse Capture)

A lossless, machine-readable log of every tool call Claude Code makes during a session, captured live by the `PostToolUse` hook and fed back into the flush pipeline as **ground-truth input for Haiku**.

- **Live capture** — `hooks/post-tool-use.py` fires after every tool invocation, writes one JSONL line to `knowledge/daily/YYYY-MM-DD.tools.jsonl` with `{ts, session_id, tool, input_digest, result_size, ok}`. Pure stdlib, broad-exception-wrapped, ~5s timeout — never breaks the session.
- **Tool-aware digests** — the hook keeps only the load-bearing fields per tool (`file_path` for Edit/Write/Read, `command` for Bash, `pattern`+`path` for Grep, `description`+`subagent_type` for Task, etc.), and caps everything else at 240 chars. The raw transcript still has the full payload if you need it.
- **Ground truth for flush.py** — `flush.py` loads today's drawer filtered by `session_id`, renders a compact ranked summary via `format_tool_events` (tool counts + ranked notable operations with priority tiering), and injects it into the Haiku flush prompt with instructions to trust the tool log over the conversation text when they disagree. Result: flush summaries cite real file paths and commands instead of reconstructing them from transcript prose.
- **Idempotent** — duplicated lines from replayed sessions do no harm; the compile pipeline already de-dupes on content hash.

This closes a hard gap in the upstream compiler: previously, Haiku had to infer "what was done this session" from the conversation text alone, which is lossier than reading the actual tool-call stream. Idea borrowed from [thedotmack/claude-mem](https://github.com/thedotmack/claude-mem)'s `PostToolUse` capture pattern; the drawer format and flush integration are local.

### 7. O(1) Prompt Cost

The upstream compiler dumps every article into every prompt — cost scales linearly with knowledge base size. This fork uses a three-level retrieval pattern:

- **Level 0 (always injected):** `index.md` (map) + `compiled-truth.md` (priority-scored excerpt, default 40KB)
- **Level 1 (on-demand MCP):** `search_knowledge`, `search_raw_daily`, `get_article` — targeted fetches
- **Level 2 (fallback):** Direct `Read` / `Grep` via the agent's built-in tools

Cost stays constant from 50 articles to 5,000.

---

## How It Works

```
                   SESSION LIFECYCLE
                   =================

  ┌─────────────┐  SessionStart hook   ┌──────────────────┐
  │ Claude Code │◄── compiled-truth ───│ session-start.py │
  │   session   │    + index + wip     └──────────────────┘
  └──┬───────┬──┘
     │       │ PostToolUse hook (every tool call)
     │       ▼
     │  ┌───────────────────┐     ┌────────────────────────────┐
     │  │ post-tool-use.py  │────►│ knowledge/daily/            │
     │  └───────────────────┘     │   YYYY-MM-DD.tools.jsonl    │
     │                            │ (structured drawer layer)   │
     │                            └──────────────┬──────────────┘
     │ SessionEnd / PreCompact hook              │
     ▼                                           │
  ┌──────────────┐  background spawn   ┌─────────┴───┐
  │session-end.py│────────────────────►│  flush.py   │
  └──────────────┘  (detached proc)    └──────┬──────┘
                                              │     (loads drawer as
                                              │      ground truth for
                                              │      the Haiku prompt)
                               ┌──────────────┼──────────────┐
                               ▼              ▼              ▼
                        daily log       wip.md        ChromaDB
                       (markdown)    (resume-here)   (verbatim chunks)
                               │
                               │ after 6 PM
                               ▼
                        ┌─────────────┐
                        │  compile.py │──► concepts/*.md ──► ChromaDB
                        └──────┬──────┘    (curated articles)
                               │
                               ▼
                        ┌──────────────────┐
                        │ compile_truth.py │──► compiled-truth.md
                        └──────────────────┘    (zero cost, pure Python)


                   RETRIEVAL DURING A SESSION
                   ==========================

  session prompt ──► always has: index.md + compiled-truth.md + codebase shape

                ──► on-demand:
                     • search_knowledge(query, type, min_confidence, ...)
                     • search_raw_daily(query, date_range, ...)
                     • get_article(slug)
                     • list_contradictions()
                     • get_codebase_overview / get_file_deps / get_route_map / ...
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full retrieval pipeline and four-store data model.

---

## Quick Start

### 1. Clone into your project

```bash
git clone https://github.com/hzlatarski/symfony-claude-context-engine.git .claude/memory-compiler
cd .claude/memory-compiler
uv sync
```

### 2. Configure hooks

Merge into your project's `.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [{"matcher": "", "hooks": [{"type": "command", "command": "cd .claude/memory-compiler && uv run python hooks/session-start.py",  "timeout": 15}]}],
    "PreCompact":   [{"matcher": "", "hooks": [{"type": "command", "command": "cd .claude/memory-compiler && uv run python hooks/pre-compact.py",    "timeout": 10}]}],
    "SessionEnd":   [{"matcher": "", "hooks": [{"type": "command", "command": "cd .claude/memory-compiler && uv run python hooks/session-end.py",    "timeout": 10}]}],
    "PostToolUse":  [{"matcher": "", "hooks": [{"type": "command", "command": "cd .claude/memory-compiler && uv run python hooks/post-tool-use.py",  "timeout": 5 }]}]
  }
}
```

`PostToolUse` fires after every tool call Claude Code makes and writes a structured JSONL drawer (`knowledge/daily/YYYY-MM-DD.tools.jsonl`) that `flush.py` later reads as ground-truth input for the Haiku flush summary — see "Structured Tool Drawer" above.

### 3. Register MCP servers

Create or merge `.mcp.json` at your project root:

```json
{
  "mcpServers": {
    "symfony-code-intel": {
      "command": "uv",
      "args": ["run", "--directory", ".claude/memory-compiler", "python", "scripts/mcp_server.py"]
    },
    "knowledge-compiler": {
      "command": "uv",
      "args": ["run", "--directory", ".claude/memory-compiler", "python", "scripts/knowledge_mcp_server.py"]
    }
  }
}
```

### 4. Seed knowledge and build the vector index

```bash
cp sources.yaml.example sources.yaml
# Edit sources.yaml to point at your project's docs / specs / memories

uv run python scripts/ingest.py           # compile source files into articles
uv run python scripts/reindex.py --all    # backfill ChromaDB
```

### 5. Use it

Sessions accumulate automatically. Ask Claude to "search the knowledge base for X" and watch it call `search_knowledge`. After any doubt, verify with `search_raw_daily` or read the compiled article directly with `get_article`.

### 6. Browse the dashboard

```bash
uv run python scripts/viewer.py
# → http://127.0.0.1:37778
```

See [Web Viewer](#web-viewer) below for everything it shows.

---

## Web Viewer

A read-only FastAPI dashboard over the knowledge store. One command, no build step, no auth, bound to localhost.

```bash
uv run python scripts/viewer.py
# → http://127.0.0.1:37778
```

| Route | What it shows |
|---|---|
| `/` | Overview: article counts, quarantine status, today's tool calls, today's flush cost, memory-type histogram, recently updated articles |
| `/articles` | Filterable article list — by memory type, min confidence, quarantine mode (hide/only/all), and substring search |
| `/articles/{slug}` | Single article with frontmatter badges, rendered markdown, `[[wikilinks]]` rewritten to internal links, raw-markdown drawer |
| `/daily` & `/daily/{date}` | Daily log index + rendered detail |
| `/tools` & `/tools/{date}` | Per-day tool-drawer browser with event counts, error counts, and per-event table |
| `/contradictions` | Current quarantine list |
| `/stats` | Chroma collection sizes, recent 20 flush records with per-session costs |

**Design notes.** Dark "tactical" theme matching the MHB project aesthetic. Memory-type tinting (`fact` blue, `event` amber, `discovery` purple, `preference` pink, `advice` teal, `decision` red) is driven by a single `type_colors` Jinja global so nav chips, badges, and card borders stay in sync — the type-tinted card border idea is borrowed from [thedotmack/claude-mem](https://github.com/thedotmack/claude-mem)'s React viewer. Read-only on principle: no mutation endpoints anywhere, and binding to `127.0.0.1` (not `0.0.0.0`) means it's never accessible from the LAN. Port `37778` is one above claude-mem's `37777` to avoid collision when both tools live on the same box.

---

## whisper-prompt

A voice-to-enhanced-prompt pipeline baked into the Web Viewer. Speak a question; the pipeline transcribes it locally, expands it into multi-angle search queries, retrieves grounding from the knowledge base, and rewrites your words into a precise, citation-grounded Claude prompt.

```bash
uv run python scripts/viewer.py
# → http://127.0.0.1:37778/whisper
```

### How it works

```
Mic → faster-whisper (CPU) → Haiku query expansion
    → parallel BM25+vector retrieval (articles / code / daily)
    → RRF merge → Sonnet grounded rewrite → enhanced prompt + citations
```

1. **Transcribe** — faster-whisper runs fully local (no API call). Model is pre-warmed at viewer startup so the first request has no cold-start penalty.
2. **Expand** — Haiku decomposes the transcript into 3–5 targeted sub-queries (articles, code, daily scopes).
3. **Retrieve** — parallel BM25 + vector searches across the selected scopes; results merged via Reciprocal Rank Fusion.
4. **Rewrite** — Sonnet rewrites the transcript into a complete, grounded prompt with inline `[[wikilink]]` citations.

### Modes

| Mode | What it produces |
|------|-----------------|
| **verbatim** | Your transcript, unchanged — no rewrite, just grounding citations appended |
| **rewrite** | Full Sonnet rewrite grounded in retrieved knowledge — default |
| **clean** | Rewrite without `[[wikilink]]` citations — paste-ready for non-wiki contexts |

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Space` | Toggle recording |
| `Cmd/Ctrl+Enter` | Copy enhanced prompt |
| `Cmd/Ctrl+R` | Re-enhance with current scope |
| `1` / `2` / `3` | Switch mode (verbatim / rewrite / clean) |

### Scope override

After transcription, toggle the **articles**, **code**, and **daily** scope chips to re-run retrieval against only the stores you care about. Pressing **Regenerate** re-calls `/api/whisper/re-enhance` with the cached transcript and new scope — no re-transcription.

### Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `MEMORY_COMPILER_MODEL_FLUSH` | `claude-haiku-4-5-20251001` | Used for query expansion |
| `MEMORY_COMPILER_MODEL_COMPILE` | `claude-sonnet-4-6` | Used for grounded rewrite |
| `WHISPER_MODEL_SIZE` | `base` | faster-whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`) |

### Cost per utterance

| Step | Model | Typical cost |
|------|-------|-------------|
| Transcription | local (CPU) | **$0.00** |
| Query expansion | Haiku | ~$0.001 |
| Grounded rewrite | Sonnet | ~$0.01–0.03 |
| **Total** | | **~$0.01–0.03** |

### Drift canary

`canary.py` includes a dedicated whisper pipeline canary (`whisper:tailwind-rebuild`) that feeds a pre-canned transcript through the grounded rewrite and asserts the result cites the expected command strings from the feedback memory. Run automatically when you run `uv run python scripts/canary.py` without `--id`.

---

## Key Commands

```bash
# Knowledge pipeline
uv run python scripts/compile.py               # compile daily logs → articles
uv run python scripts/ingest.py                # compile source files → articles
uv run python scripts/compile_truth.py         # regenerate compiled-truth.md (pure Python)
uv run python scripts/query.py "question"      # ask the KB (uses Sonnet)

# Vector store
uv run python scripts/reindex.py               # incremental (hash-based)
uv run python scripts/reindex.py --all         # force full rebuild
uv run python scripts/reindex.py --articles-only
uv run python scripts/reindex.py --daily-only

# Drift detection & quality
uv run python scripts/lint.py                  # full lint (structural + contradictions)
uv run python scripts/lint.py --structural-only
uv run python scripts/lint.py --resolve        # clear contradiction quarantine
uv run python scripts/canary.py                # run drift canaries
uv run python scripts/canary.py --dry-run      # list canaries without running
```

---

## Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MEMORY_COMPILER_DISABLED_HOOKS` | _(unset)_ | Disable hooks: `all` or comma-separated (`session-start,session-end,pre-compact`) |
| `MEMORY_COMPILER_MODEL_FLUSH` | `claude-haiku-4-5-20251001` | Model for session flush (cheap) |
| `MEMORY_COMPILER_MODEL_COMPILE` | `claude-sonnet-4-6` | Model for daily-log compilation |
| `MEMORY_COMPILER_MODEL_INGEST` | `claude-sonnet-4-6` | Model for source-file ingestion |
| `MEMORY_COMPILER_MODEL_QUERY` | `claude-sonnet-4-6` | Model for interactive queries |
| `MEMORY_COMPILER_MODEL_CANARY` | `claude-haiku-4-5-20251001` | Model for drift canary checks |
| `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` | `95` | Set to `50` to compact earlier in long sessions |

### `sources.yaml`

See `sources.yaml.example`. Each source group has an `id`, `type` (markdown built-in; PDF / URL planned), glob `include` / `exclude` patterns, a category tag, and a description.

### Article Format

```markdown
---
title: "Concept Name"
type: fact              # fact|event|discovery|preference|advice|decision
confidence: 0.85
sources:
  - "daily/2026-04-01.md"
created: 2026-04-01
updated: 2026-04-03
---

## Truth

### Observed

- Fact A from source [src:daily/2026-04-01.md]
- Fact B corroborated by two sources [src:daily/2026-04-01.md] [src:daily/2026-04-03.md]

### Synthesized

- Inferred pattern: A + B together imply X [src:daily/2026-04-01.md]

### Related Concepts

- [[concepts/related-concept]] — how it connects (one line)

---

## Timeline

### 2026-04-01 | daily/2026-04-01.md
- Initial discovery during project setup
- Decided to use X approach because Y
```

See [AGENTS.md](AGENTS.md) for the complete schema and schema rules.

---

## Cost

All LLM costs use your existing Claude subscription (Max / Team / Enterprise) — no separate API key needed.

| Operation | Model | Cost | When |
|---|---|---|---|
| Session flush | Haiku | ~$0.005-0.02 | Every session end (automatic) |
| Daily compilation | Sonnet | ~$0.20-0.60 | After 6 PM (automatic) |
| Source ingestion | Sonnet | ~$0.20-0.60/file | Manual |
| Compiled truth | _pure Python_ | **$0.00** | After every compile/ingest |
| Structural lint | _pure Python_ | **$0.00** | Manual |
| Contradiction lint | Sonnet | ~$0.15-0.25 | Manual |
| Canary drift check | Haiku | ~$0.05-0.20 | Manual/scheduled |
| Query | Sonnet | ~$0.15-0.40 | Manual |
| **Vector store** | _local ONNX_ | **$0.00** | Always |

Typical automatic cost: **$0.25-0.75 per day** for 10-15 sessions.

### Why Costs Are Stable

The upstream `claude-memory-compiler` dumps all existing wiki articles into every compile/ingest prompt — costs grow linearly with knowledge-base size. This fork fixes that with the three-level retrieval pattern: `index.md` (always, tiny) + `compiled-truth.md` (always, fixed budget) + on-demand `search_knowledge` / `get_article`. Cost per operation is approximately constant from 50 articles to 5,000.

---

## What Makes This Different From MemPalace

MemPalace's thesis is "store everything verbatim, let vector search sort it out." This engine agrees that verbatim matters — that's the drawer layer — but it also believes **curated structure is worth the LLM cost**. The result:

| | MemPalace | This engine |
|---|---|---|
| Storage | Verbatim drawers in ChromaDB | Verbatim daily chunks **and** compiled concept articles |
| Curation | None — LLM extraction rejected on principle | LLM compiles, with anti-drift defenses |
| Retrieval | Semantic search only | Semantic search **plus** priority-scored compiled-truth **plus** knowledge graph wikilinks |
| Drift detection | Fact-checker not wired up | Canaries, confidence decay, contradiction quarantine, skeptical compile prompt |
| Code awareness | Domain-agnostic | Dedicated Symfony code-intel MCP server |
| Metadata filters | Wing/room/hall metadata | Memory type, confidence, zone, quarantine state |
| Cost | ~$0 runtime | ~$0.25-0.75/day automatic |

Neither approach is "right" — they're different tradeoffs. This engine is the tool you want when your knowledge base must **read like documentation**, not like a grep-pile.

---

## Obsidian Integration

The knowledge base is pure markdown with `[[wikilinks]]`. Point an Obsidian vault at `knowledge/` for graph view, backlinks, and search alongside the MCP tools.

---

## Technical Reference

- **[AGENTS.md](AGENTS.md)** — article schema, hook architecture, script internals, source handler API
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — retrieval pipeline, four-store data model, MCP routing

## Credits

- Forked from [coleam00/claude-memory-compiler](https://github.com/coleam00/claude-memory-compiler) by Cole Medin
- Inspired by [Andrej Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
- Verbatim drawer layer concept from [MemPalace](https://github.com/MemPalace/mempalace)
- Built on the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk) and [FastMCP](https://github.com/modelcontextprotocol/python-sdk)

## License

MIT
