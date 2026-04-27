# Claude Context Engine — Symfony Edition

**Your AI conversations, project docs, and codebase compile themselves into a searchable, self-healing knowledge base — with semantic retrieval, memory-type filtering, and drift detection baked in.**

A long-term memory system for Claude Code, purpose-built for Symfony projects. Session transcripts, design specs, and live codebase structure flow into a single knowledge store that Claude queries on demand through dedicated MCP tools. Unlike vector-only systems that store everything and hope for the best, this engine **compiles** raw conversations into structured, source-cited articles — and defends that structure against the drift failure modes typical LLM wikis suffer.

**Target stack:** Symfony 7.x, PHP 8.2+, Twig, Stimulus.js, AssetMapper.

**Lineage:** Forked from [coleam00/claude-memory-compiler](https://github.com/coleam00/claude-memory-compiler), itself inspired by [Andrej Karpathy's LLM Knowledge Base](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f). Substantially extended with a Symfony code-intelligence layer, anti-drift hardening, and a semantic retrieval surface inspired by [MemPalace](https://github.com/MemPalace/mempalace) and [thedotmack/claude-mem](https://github.com/thedotmack/claude-mem).

---

## Key Features

- 🖥 **[Web Viewer UI](#web-viewer)** — Read-only FastAPI dashboard at **<http://127.0.0.1:37778>**. One command (`uv run python scripts/viewer.py`), no build step, no auth. Browse articles, daily logs, tool drawer, contradictions, and cost history.
- 🎙 **[Voice-to-enhanced-prompt](#whisper-prompt)** — Mic page at **<http://127.0.0.1:37778/whisper>**. Speak a question; the pipeline transcribes it locally (faster-whisper, CPU), expands it via Haiku query-expansion, retrieves grounding from the knowledge base, and rewrites it into a fully grounded Claude prompt. Three modes: raw, clean, context.
- 🖱 **[WhisperTray](#whisptertray)** — Standalone Windows system-tray dictation app. Global hotkey (customizable) starts recording from any window; a floating pill overlay shows state; on stop, the audio is transcribed and enhanced, then auto-pasted back into the window that was focused before recording started.
- 🧠 **Curated memory, not a grep-pile** — Sessions compile into structured articles with Truth + Timeline format, `[[wikilinks]]`, and `[src:path]` provenance anchors. Human-readable, diffable, Obsidian-compatible.
- 🏷 **Memory type taxonomy** — Every article is a `fact`, `event`, `discovery`, `preference`, `advice`, or `decision`. First-class filter in `search_knowledge` so "only preferences about testing" is one call.
- 🔍 **Hybrid BM25 + vector search** — Two dedicated MCP servers: `knowledge-compiler` for semantic+lexical retrieval, `symfony-code-intel` for live codebase structure. Fused via Reciprocal Rank Fusion.
- ⚡ **Token-efficient retrieval** — `search_knowledge` returns slim ~220-char snippets; `get_articles([slugs])` batch-fetches full bodies only for the winners. ~10× context savings on multi-hit queries.
- 🛠 **Symfony code intelligence** — Live-parsing MCP tools: `get_codebase_overview`, `get_file_deps`, `get_route_map`, `get_template_graph`, `get_stimulus_map`, `get_hotspots`, `trace_route`, `impact_of_change`, `get_circular_dependencies`. Mtime-cached, sub-second. `trace_route` and `impact_of_change` accept `output_format="mermaid"` for flowchart rendering.
- 👀 **Live file watcher** — `scripts/watch.py` runs alongside the viewer, debounces filesystem events, and incrementally reindexes both knowledge articles and codebase chunks within ~2s of any save. No more "I edited the article and `search_knowledge` still returns the old version" gap.
- 🔗 **Cross-project linked search** — Set `MEMORY_COMPILER_LINKED_PROJECTS=/path/a,/path/b` and pass `include_linked=true` to `search_knowledge` to fan out a single query across multiple project knowledge bases. Results are RRF-merged and tagged with the originating project name.
- 🛡 **Cross-process locking** — Concurrent writers (the file watcher, a `SessionEnd` flush, a manual `ingest.py`) coordinate via `filelock` against per-collection lock files under `knowledge/chroma/.locks/`. No more SQLite "database is locked" surprises.
- ⏸ **Resumable, interruptible ingest** — Per-file content-hash checkpointing makes `ingest.py` crash-safe — re-running skips files whose hash matches the last successful run. Live progress is written to `knowledge/.ingest-status.json` and exposed via the `ingest_status` MCP tool. `ingest_stop` raises a cooperative-cancel flag the next file boundary honors.
- 🩺 **`kb_health` MCP tool** — One-shot diagnostic: collection sizes, articles by memory type, broken `[src:]` anchors, quarantine count, last ingest timestamp. Replaces six manual scripts when you need to know "is the KB healthy?".
- 🌳 **AST-aware code chunking** — PHP and JS files chunk on class/method/function boundaries via tree-sitter, so `search_codebase` hits land on whole units instead of mid-method line slices. Twig and YAML fall back to 150-line windows.
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

Six pure-Python parsers plus a tree-sitter call graph expose your live codebase as structured data via the `symfony-code-intel` MCP server's nine tools:

| Tool | What it returns |
|---|---|
| `get_codebase_overview` | File counts by type, routes, templates, Stimulus orphans, top churn hotspots |
| `get_file_deps(path)` | Imports, reverse deps, routes/templates touched, co-change partners |
| `get_route_map(prefix)` | Route → controller → action → template → injected services table |
| `get_template_graph(t)` | Twig inheritance, includes, Stimulus bindings |
| `get_stimulus_map(c)` | Bidirectional JS ↔ Twig map with orphan detection |
| `get_hotspots(top_n)` | Churn-ranked files with ownership and bus-factor scoring |
| `trace_route(method, path, output_format="text")` | Full call chain a route triggers — controller action down through services, repositories, and rendered templates. Each hop carries a confidence score reflecting how the receiver type was resolved. Set `output_format="mermaid"` for a `flowchart TD` rendering. |
| `impact_of_change(file=None, since_ref="HEAD", output_format="text")` | Reverse-walks the call graph from edited lines (parsed from `git diff -U0`) to surface affected HTTP routes **and** Stimulus controllers, risk-scored by hotspot weight. Crosses the JS↔PHP boundary via resolved `fetch()` URLs. Mermaid output mode renders affected routes + reached methods as a two-tier flowchart. |
| `get_circular_dependencies(scope="vendor-excluded", output_format="text")` | Tarjan's SCC over the resolved call graph. Reports cycles (SCCs of size > 1 plus self-loops) sorted by size. `scope` filters to `php`, `js`, `all`, or `vendor-excluded` (default — keeps only `src/...` symbols). |

Runs live, mtime-cached, under one second. Git intelligence caches to `knowledge/git-intel.json` (HEAD-based invalidation); the symbol-level call graph caches to `knowledge/call-graph.json` (mtime + HEAD invalidation).

#### Confidence-scored call graph

`trace_route` and `impact_of_change` are backed by a tree-sitter call-graph parser that walks `src/**/*.php` and `assets/controllers/**/*_controller.js`. PHP resolution rules:

| Pattern | Confidence |
|---|---|
| Constructor-injected typed properties (promoted or classic) | 1.0 |
| Static calls `Foo::bar()`, `self::`, `parent::` (with `extends`-clause walking) | 1.0 |
| `$this->method()` resolving up the inheritance chain | 1.0 |
| Doctrine `$em->getRepository(X::class)` chained or via local var | 1.0 |
| `$this->render('x.html.twig')` → render edge to `template:...` | 1.0 |
| Typed local var: parameter type or `new X()` assignment | 0.7 |
| Untyped or dynamic dispatch | skipped |

JS resolution rules emit `js:<stimulus-name>::<method>` symbols and resolve `fetch()` calls against the route map: literal URLs at confidence 1.0, template literals (`/api/x/${id}/y`) at 0.7 with `${...}` collapsed to `*` for wildcard matching against route placeholders.

### 3. Knowledge MCP Server (Semantic Retrieval)

A **second** MCP server (`knowledge-compiler`) — separate from the code-intel one on purpose — exposes the knowledge store:

| Tool | What it returns |
|---|---|
| `search_knowledge(query, ..., include_linked=False)` | Semantic search over curated articles with filters for memory `type`, `min_confidence`, `zone`, quarantine state. Returns **slim snippets** (~220 chars), not full bodies. Set `include_linked=True` to also search every project listed in `MEMORY_COMPILER_LINKED_PROJECTS` — hits get a `project` tag. |
| `search_raw_daily(query, date_from, date_to)` | Semantic search over verbatim drawer chunks (daily logs, never summarized). Slim snippets. |
| `search_codebase(query, file_type)` | Hybrid BM25 + vector search over indexed source files. PHP and JS are chunked at class/method/function boundaries via tree-sitter (see below); Twig and YAML use 150-line windows. Returns chunked file excerpts with line ranges, plus the source group's `source_description` so the agent learns *when* to consult that file group. |
| `get_article(slug)` | Full markdown + parsed frontmatter for one article |
| `get_articles([slugs])` | Batch-fetch full bodies for multiple slugs in one round trip. Missing slugs return `{slug, error: "not_found"}` so one bad slug doesn't abort the batch. |
| `list_contradictions()` | Current contradiction-quarantine list |
| `list_sources()` | Source-group catalog: `{file_type, patterns, description, chunk_count}` per group. Use **before** `search_codebase` when you don't know which `file_type` fits — descriptions tell you when to consult each group. Doubles as a freshness check (`chunk_count: 0` for an expected group means the index is stale). |
| `kb_health()` | One-shot diagnostic. Returns `{vector_store, codebase_store, articles{total,by_type}, anchors{broken,articles_missing_anchors}, quarantine, freshness, ingest}`. Use as a CI gate or before answering critical KB questions. |
| `ingest_status()` | Live progress snapshot for the most recent / active `ingest.py` run. Reads `knowledge/.ingest-status.json`. Phases: `idle`, `starting`, `running`, `finished`, `stopped`, `error`. Compare `updated_at` against current time — a stale snapshot during `running` likely means the process crashed silently. |
| `ingest_stop()` | Cooperative halt for a running ingest. The current Sonnet call (if any) finishes and is checkpointed before the run exits — re-running `ingest.py` resumes from the next unprocessed file via the existing hash-based skip logic. |

Backed by **ChromaDB** with the bundled `all-MiniLM-L6-v2` ONNX embedder — fully local, zero API cost, ~90 MB one-time model download on first use.

**Token-efficient two-step retrieval** — inspired by claude-mem's `search → get_observations` split, `search_knowledge` returns just `{slug, title, snippet, distance, metadata}` (~50–100 tokens per hit) so the agent can scan cheaply and then fetch full bodies only for the winners via `get_article(s)`. Saves ~10× context on multi-hit queries where most matches turn out to be irrelevant.

**AST-based code chunking.** `index_codebase.py` chunks PHP and JS source via tree-sitter so each chunk is a complete `class`, `interface`, `trait`, `enum`, `function`, or `method` — not an arbitrary line window that cuts mid-method. Classes larger than 400 lines split into a header chunk plus one chunk per method; methods larger than 400 lines fall back to 150-line windows so no single chunk grows unbounded. Twig, YAML, and any file where the tree-sitter parser fails or finds no top-level declarations transparently fall through to the line-window chunker. The win: `search_codebase` hits return whole, semantically-coherent units the LLM can reason about, instead of half a method's tail spliced to half another method's head. Idea borrowed from [zilliztech/claude-context](https://github.com/zilliztech/claude-context); their AST-splitter pattern is the only piece of that project that wasn't already covered here.

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

### One-command install

```bash
git clone https://github.com/hzlatarski/symfony-claude-context-engine.git .claude/memory-compiler
uv run --directory .claude/memory-compiler python install.py
```

`install.py` does everything in one shot:

1. Merges Claude Code hooks into `.claude/settings.json` (idempotent — safe to re-run)
2. Registers MCP servers in `.mcp.json` at your project root
3. Copies `sources.yaml.example` → `sources.yaml` (skipped if already present)
4. Runs initial ingest + ChromaDB vector reindex

After it finishes, edit `sources.yaml` to point at your project's docs and specs, then open Claude Code — the hooks fire automatically on the next session.

#### VS Code: run setup from the Task Runner

If you use VS Code, the repo includes a `.vscode/tasks.json` that exposes the installer as a runnable task. After cloning:

1. Open the Command Palette → **Tasks: Run Task**
2. Select **Setup Claude Context Engine**

This runs the same `install.py` command in the integrated terminal — no manual typing required.

### Browse the dashboard

```bash
uv run --directory .claude/memory-compiler python scripts/viewer.py
# → http://127.0.0.1:37778
```

### Manual setup (advanced)

If you prefer to run steps individually or need to integrate into an existing `.claude/settings.json` by hand:

<details>
<summary>Expand manual steps</summary>

#### 1. Clone and sync deps

```bash
git clone https://github.com/hzlatarski/symfony-claude-context-engine.git .claude/memory-compiler
cd .claude/memory-compiler
uv sync
```

#### 2. Configure hooks

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

#### 3. Register MCP servers

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

#### 4. Seed knowledge and build the vector index

```bash
cp sources.yaml.example sources.yaml
# Edit sources.yaml to point at your project's docs / specs / memories

uv run python scripts/ingest.py           # compile source files into articles
uv run python scripts/reindex.py --all    # backfill ChromaDB
```

#### 5. Use it

Sessions accumulate automatically. Ask Claude to "search the knowledge base for X" and watch it call `search_knowledge`. After any doubt, verify with `search_raw_daily` or read the compiled article directly with `get_article`.

</details>

See [Web Viewer](#web-viewer) below for everything the dashboard shows.

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

**Design notes.** Dark "tactical" theme — a single-accent tinted dark palette. Memory-type tinting (`fact` blue, `event` amber, `discovery` purple, `preference` pink, `advice` teal, `decision` red) is driven by a single `type_colors` Jinja global so nav chips, badges, and card borders stay in sync — the type-tinted card border idea is borrowed from [thedotmack/claude-mem](https://github.com/thedotmack/claude-mem)'s React viewer. Read-only on principle: no mutation endpoints anywhere, and binding to `127.0.0.1` (not `0.0.0.0`) means it's never accessible from the LAN. Port `37778` is one above claude-mem's `37777` to avoid collision when both tools live on the same box.

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
    → RRF merge → Sonnet context-grounded rewrite → enhanced prompt + citations
```

1. **Transcribe** — faster-whisper runs fully local (no API call). Model is pre-warmed at viewer startup so the first request has no cold-start penalty.
2. **Expand** — Haiku decomposes the transcript into 3–5 targeted sub-queries (articles, code, daily scopes).
3. **Retrieve** — parallel BM25 + vector searches across the selected scopes; results merged via Reciprocal Rank Fusion.
4. **Context** — Sonnet rewrites the transcript into a complete, grounded prompt with inline `[[wikilink]]` citations.

### Modes

| Mode | What it produces |
|------|-----------------|
| **raw** | Your transcript, unchanged — no AI, no rephrasing. Fastest. |
| **clean** | Fix grammar and remove filler words. One quick Haiku call. |
| **context** | Full Sonnet rewrite grounded in retrieved knowledge — default. |

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Space` | Toggle recording |
| `Cmd/Ctrl+Enter` | Copy enhanced prompt |
| `Cmd/Ctrl+R` | Re-enhance with current scope |
| `1` / `2` / `3` | Switch mode (raw / clean / context) |

### Scope override

After transcription, toggle the **articles**, **code**, and **daily** scope chips to re-run retrieval against only the stores you care about. Pressing **Regenerate** re-calls `/api/whisper/re-enhance` with the cached transcript and new scope — no re-transcription.

### Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `MEMORY_COMPILER_MODEL_FLUSH` | `claude-haiku-4-5-20251001` | Used for query expansion |
| `MEMORY_COMPILER_MODEL_COMPILE` | `claude-sonnet-4-6` | Used for context-grounded rewrite |
| `WHISPER_MODEL_SIZE` | `base` | faster-whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`) |

### Cost per utterance

| Step | Model | Typical cost |
|------|-------|-------------|
| Transcription | local (CPU) | **$0.00** |
| Query expansion | Haiku | ~$0.001 |
| Context-grounded rewrite | Sonnet | ~$0.01–0.03 |
| **Total** | | **~$0.01–0.03** |

### Drift canary

`canary.py` includes a dedicated whisper pipeline canary (`whisper:tailwind-rebuild`) that feeds a pre-canned transcript through the context-grounded rewrite and asserts the result cites the expected command strings from the feedback memory. Run automatically when you run `uv run python scripts/canary.py` without `--id`.

---

## WhisperTray

A standalone Windows system-tray dictation app that pairs with the same `whisper.orchestrator` pipeline used by the web viewer's mic page. Press a global hotkey from any application, speak, press stop — the result is transcribed, enhanced, and auto-pasted back into the window you were in before recording started.

```bash
uv run python whisper_tray/main.py
```

### How it works

1. **Global hotkey** — `HotkeyListener` (pynput) fires from any focused window, regardless of which app is in front.
2. **HWND capture** — at the moment recording begins, `GetForegroundWindow()` saves the target window handle before the pill overlay can steal focus.
3. **Pill overlay** — a frameless topmost `tk.Toplevel` appears at the bottom of the screen showing recording state (animated braille spinner), mode selector, cancel (✕) and stop (■) buttons, and a `?` help popup.
4. **Transcribe + enhance** — audio is handed to `whisper.orchestrator.enhance_from_audio()` in a thread pool; the same Haiku/Sonnet pipeline used by the mic page runs in the background.
5. **Auto-paste** — `injector.py` copies the result to the clipboard, calls `SetForegroundWindow(hwnd)` to restore the original window's focus, then fires `Ctrl+V`.

> **Usage tip:** Place your cursor in the target input (chat box, editor, etc.) *before* pressing the hotkey — the destination window is locked in at that moment. You do not need to click back after speaking; focus is restored automatically.

### Modes

| Mode | Pipeline | Cost |
|------|----------|------|
| **raw** | Transcript only — no AI call | $0.00 |
| **clean** | Haiku grammar fix + filler removal | ~$0.001 |
| **context** | Expand → retrieve KB → Sonnet grounded rewrite | ~$0.01–0.03 |

The mode can be changed per-recording from the pill overlay (radio buttons) or locked globally in Settings.

### Settings

Open Settings from the system tray icon. Options:

| Setting | Default | Notes |
|---------|---------|-------|
| Hotkey | `<ctrl>+<cmd>` | Click "Record…" to capture any combo |
| Hotkey mode | `click_toggle` | `click_toggle` = press once to start, again to stop; `hold` = hold to record |
| Enhancement mode | `context` | `raw` / `clean` / `context` |
| Mode lock | off | Hides the per-recording mode selector on the pill |
| Auto-paste | on | Ctrl+V into the source window after enhance |
| Microphone | Auto-detect | Choose any input device by name |
| Language | Auto-detect | Pass language hint to faster-whisper |
| Launch with Windows | off | Writes a `HKCU\...\Run` registry key |

Settings persist to `~/.whisper-tray/settings.json`.

### First-run wizard

On first launch (no `settings.json` found), a setup wizard prompts for the hotkey and auto-paste preference, then writes `settings.json` and starts the listener.

### Installation

```bash
# From the memory-compiler root:
uv sync
uv run python whisper_tray/main.py
```

Requires the same `uv` environment as the rest of the project. No separate install step — all deps are already in `pyproject.toml` (`pystray`, `pynput`, `sounddevice`, `pyautogui`, `pyperclip`, `Pillow`).

### Running tests

```bash
uv run pytest tests/whisper_tray/ -v
```

---

## Key Commands

```bash
# Knowledge pipeline
uv run python scripts/compile.py               # compile daily logs → articles
uv run python scripts/ingest.py                # compile source files → articles
uv run python scripts/ingest.py --all          # force re-ingest (per-file hash checkpoint still saves rerun cost)
uv run python scripts/compile_truth.py         # regenerate compiled-truth.md (pure Python)
uv run python scripts/query.py "question"      # ask the KB (uses Sonnet)

# Vector store
uv run python scripts/reindex.py               # incremental (hash-based)
uv run python scripts/reindex.py --all         # force full rebuild
uv run python scripts/reindex.py --articles-only
uv run python scripts/reindex.py --daily-only

# Live watcher (run alongside viewer.py — auto-reindexes on file change, ~2s debounce)
uv run python scripts/watch.py                 # foreground; Ctrl-C to stop
uv run python scripts/watch.py --quiet         # WARNING-level logging

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
| `MEMORY_COMPILER_LINKED_PROJECTS` | _(unset)_ | Comma-separated absolute paths to other Symfony project roots whose `knowledge/chroma/` collections should be searchable via `search_knowledge(..., include_linked=True)`. Non-existent paths are silently skipped at search time. |
| `MEMORY_COMPILER_EXTRA_EXTENSIONS` | _(unset)_ | Comma-separated file extensions (each with leading dot — e.g. `.dist,.neon`) that the codebase indexer should also pick up. Each extension becomes its own `file_type`; globs are scoped to `src/`, `assets/`, `templates/`, `config/` so vendor trees aren't walked. Files are indexed as plaintext (no AST chunking). |

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

## Cross-Project Linked Search

If you maintain multiple Symfony projects with their own knowledge bases, point them at each other so a single query can answer "have we hit this before *anywhere*?":

```bash
export MEMORY_COMPILER_LINKED_PROJECTS=/c/wamp64/www/ManilvaHandyMan,/c/wamp64/www/eintollesfest
```

Then call `search_knowledge` with `include_linked=true`:

```jsonc
// agent → knowledge-compiler MCP
search_knowledge {
  "query": "Tailwind v4 rebuild after CSS changes",
  "include_linked": true,
  "limit": 5
}
```

Each linked project is searched in parallel (vector-only — cross-process BM25 isn't exposed) and merged into the local results via Reciprocal Rank Fusion. Hits are tagged with a `project` field — `"<local>"` for the current project, otherwise the linked project's directory name. Non-existent paths are silently skipped at search time, so a stale `LINKED_PROJECTS` won't break the call.

**Constraints.** All linked projects must use the default Chroma embedder (this engine's bundled ONNX MiniLM). Cross-process BM25 indexes are intentionally not exposed: vector recall is good enough for the "have we seen this elsewhere?" use case, and a shared BM25 corpus would require either a remote service or a tighter coupling between projects than the env-var contract justifies.

---

## Live File Watcher

Run `scripts/watch.py` alongside `viewer.py` to keep the article and codebase indexes live as you edit:

```bash
uv run python scripts/watch.py
# 2026-04-26 14:32:11 watch INFO Watching /c/wamp64/www/AiTutor/knowledge
# 2026-04-26 14:32:11 watch INFO Watching /c/wamp64/www/AiTutor/src
# 2026-04-26 14:32:11 watch INFO Watcher up. Ctrl-C to stop.
```

The watcher classifies each filesystem event into `article` (markdown under `knowledge/concepts/`), `daily` (markdown under `knowledge/daily/`), or `codebase` (any supported extension under `src/`, `assets/`, `templates/`, `config/`). Events are debounced for **2 seconds** of quiet, then dispatched to `reindex_articles()`, `reindex_daily()`, or `index_codebase.reindex_single()` as appropriate. Cross-process locking (see [Cross-process Safety](#cross-process-safety)) means the watcher and a manual `ingest.py` can run simultaneously without corrupting the SQLite-backed Chroma store.

It's a foreground process — `Ctrl-C` shuts down the observer cleanly. Failure of any single reindex is logged but never crashes the watcher loop.

---

## Self-Upgrade

Inspired by [gstack](https://github.com/garrytan/gstack)'s upgrade flow. The engine ships with a periodic version check that surfaces a one-line notice at session start when a new release lands, and a dedicated skill (`/memory-compiler-upgrade`) that handles the prompt + execution.

### How it works

1. **`VERSION` file** at the repo root is the source of truth for the installed release.
2. **`scripts/check_update.py`** is invoked by the SessionStart hook. It probes `https://raw.githubusercontent.com/hzlatarski/symfony-claude-context-engine/main/VERSION` (cached for 6h, snooze-aware) and emits one of:
   - `UPGRADE_AVAILABLE <old> <new>` — remote ahead of local
   - `JUST_UPGRADED <old> <new>` — surfaced once after a successful upgrade so the agent can render "what's new" from `CHANGELOG.md`
   - silence (offline / up-to-date / snoozed / disabled)
3. **`hooks/session-start.py`** injects the notice as `## Memory Compiler Update Available` into the session context. The agent reads it and runs `/memory-compiler-upgrade`.
4. **`/memory-compiler-upgrade` skill** (installed at `~/.claude/skills/memory-compiler-upgrade/SKILL.md`) prompts via `AskUserQuestion` with four options:
   - **Yes, upgrade now** — runs `scripts/upgrade.py` (stash + `git fetch` + `reset --hard origin/main` + migrations + `uv sync`)
   - **Always keep me up to date** — sets `auto_upgrade: true` in `~/.memory-compiler/config.json`, then upgrades
   - **Not now** — snoozes with escalating backoff (24h → 48h → 1 week)
   - **Never ask again** — sets `update_check: false`; re-enable with `MEMORY_COMPILER_UPDATE_CHECK=1`
5. **After the upgrade**, the next session-start surfaces `JUST_UPGRADED` and the agent renders 5–7 bullets summarising the diff between the old and new `## [version]` blocks in `CHANGELOG.md`.

### State directory

```
~/.memory-compiler/
├── config.json            # {auto_upgrade, update_check}
├── last-update-check      # touch file — 6h cache TTL
├── update-snoozed         # "<version> <level> <epoch>"
└── just-upgraded-from     # written by upgrade.py — consumed once by the skill
```

### Manual usage

```bash
# Check (bypassing cache + snooze) and prompt if anything is available
/memory-compiler-upgrade

# Direct probe — prints UPGRADE_AVAILABLE / JUST_UPGRADED / nothing
uv run python scripts/check_update.py --force

# Force a non-interactive upgrade (skips the AskUserQuestion prompt)
uv run python scripts/upgrade.py
```

### Env overrides

| Variable | Effect |
|---|---|
| `MEMORY_COMPILER_AUTO_UPGRADE=1` | Skip the prompt and upgrade silently when an update is detected |
| `MEMORY_COMPILER_UPDATE_CHECK=0` | Disable update checks entirely (the SessionStart probe stays silent) |
| `MEMORY_COMPILER_REMOTE_VERSION_URL` | Alternate raw `VERSION` URL for testing or self-hosted forks |
| `MEMORY_COMPILER_STATE_DIR` | Alternate state dir (defaults to `~/.memory-compiler/`) |

### Migrations

Place per-version migration scripts under `scripts/migrations/v<X>.<Y>.<Z>.{sh,py}`. The upgrader runs every script whose version is strictly greater than the old `VERSION` and ≤ the new one, in semver order. Scripts must be idempotent — failures are non-fatal so a botched migration doesn't strand the user mid-upgrade.

---

## Cross-process Safety

Concurrent writers — the watcher, a `SessionEnd` flush, a manual `ingest.py`, a `reindex.py --all` — coordinate via per-collection file locks under `knowledge/chroma/.locks/`. Implementation: the [`filelock`](https://py-filelock.readthedocs.io/) library, one lock per Chroma collection (`articles`, `daily_chunks`, `codebase`), held only for the duration of the upsert/delete call. Read paths (queries, counts) are unlocked because Chroma handles concurrent reads safely.

Default acquisition timeout is 60s (configurable via `CHROMA_LOCK_TIMEOUT_SECONDS`). If a process crashes mid-write, the OS releases the lock file handle and the next acquirer reclaims it automatically — no manual cleanup required.

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
