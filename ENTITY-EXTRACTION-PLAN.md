# Entity Extraction & Cross-Referencing — Design Note + Plan

**Status:** proposed (not built). **LLM cost:** $0 — pure Python, same class as `crosslink.py` / `salience.py`.
**Origin:** review of [openaleph/openaleph](https://github.com/openaleph/openaleph), 2026-09-05.

---

## 1. Background — what OpenAleph does that we don't

OpenAleph is an investigative-journalism platform (Elasticsearch + Postgres + Redis + Docker). Almost none of its stack fits this engine — we are deliberately local, markdown-first, zero-server, zero-API-cost, and our graph/retrieval layer is already more sophisticated than a journalism index needs.

**One idea transfers.** OpenAleph's core value is **entity resolution**: it lifts *named things* out of documents — a person, a company, an address, an email — and links every document that mentions the same one, using a fixed entity vocabulary called **FollowTheMoney** (`Person`, `Company`, `Email`, `Address`, …).

Our graph links **articles to articles by title** (`crosslink.py`) and **articles to code by `[src:]` anchor** (`unified_graph.py`). It does **not** link by the *thing inside the prose*. Today, twenty articles that all mention prod server `65.21.4.203`, or `KokoroTtsService`, or the command `app:content:narrate`, share **no graph edge** unless one happens to be an article title. That is the gap.

**The win:** materialize those recurring things as first-class `entity:` graph nodes, so:
- `get_unified_neighbors("entity:host/65.21.4.203")` → every article and file that touches that server.
- `get_salience(kind="hubs")` surfaces the load-bearing entities of the whole KB for free.
- a new `find_entity(...)` MCP tool answers "show me everything about X" in one hop.

---

## 2. Design decision — graph-only, never rewrite articles

`crosslink.py` **writes** `[[wikilink]]` bullets into article bodies. That is its risk surface (masking, section insertion, idempotency).

Entities must **not** do that. The precedent already in the codebase is the **`note:` rationale node** in `unified_graph.py` (`_emit_rationale`): rationale comments are lifted from source and materialized as degree-1 graph nodes **at graph-build time**, and the article/source files are never touched.

**Entities follow the `note:` model exactly:**
- A pure-Python extractor finds entity mentions in article prose.
- `unified_graph.build()` mints `entity:<canonical-id>` nodes and `article → entity` edges (`kind="mentions"`) during the existing Pass-2 loop over `article_contents`.
- No article file is ever modified. Nothing to lint, no masking bugs, fully reproducible, cache-invalidated by the same signature the graph already uses.

This makes the feature strictly additive and low-risk.

---

## 3. Entity vocabulary (FtM-inspired, scoped to this project)

A **small, closed, high-precision** set — false entities pollute the graph exactly like false wikilinks. Each type is a named regex + normalizer. Extraction only fires **outside** frontmatter, code fences, inline code, and existing links (reuse `crosslink.mask_non_prose`).

| Entity type | Node id | Recognizer (conservative) | Canonical form |
|---|---|---|---|
| `host` | `entity:host/<value>` | IPv4 literal (every octet ≤ 255) | the IP |
| `command` | `entity:command/<value>` | `app:<seg>:<seg>…` console namespace | the command, lowercased |
| `role` | `entity:role/<NAME>` | `ROLE_*` Symfony security roles | the role name |
| `envvar` | `entity:envvar/<NAME>` | `SCREAMING_SNAKE` whose LAST segment is an env suffix (`KEY`, `SECRET`, `TOKEN`, `DSN`, `URL`, `URI`, `PATH`, `HOST`, `PORT`, `ENDPOINT`, `REGION`, `BUCKET`, `PASSWORD`, `PWD`) | the name |
| `service` | `entity:service/<tail>` | PascalCase ending `Service`/`Repository`/`Controller`/`Handler`/`Subscriber`/`Listener`/`Factory` | the class short-name |
| `url` | `entity:url/<host+path>` | `https?://…`, `wss?://…` | scheme-stripped host+path |
| `path` | `entity:path/<rel>` | `src/…`, `templates/…`, `assets/…`, `config/…` | the relative path |

> **Slice-1 dry-run result (2026-09-05, live KB):** `host`, `command`, `role`,
> `envvar`, `url`, `path`, `service` all measured clean. The original single
> `envvar` recognizer (any `SCREAMING_SNAKE`) was too broad — it swept in
> `ROLE_*` and non-env constants (`STATUS_DISABLED`, `DEFAULT_RATES`). Fixed by
> splitting `role` out and gating `envvar` on the suffix allowlist above.
> Precision over recall: a suffixless real var (`DEPLOY_AGENT`) is deliberately
> missed rather than readmit the junk. **70 entities are cross-referenced across
> ≥ 2 articles** — the graph-density win Slice 2 will materialize.

Notes:
- **Precision over recall.** Start with `host` + `command` + `envvar` (near-zero false-positive rate), add the rest once the false-link rate is measured on the real KB.
- `path`/`service` overlap the existing `file:`/`class:` layer — for those, **prefer resolving the mention to the existing `file:`/`class:` node** rather than minting a parallel `entity:` node. Only mint an `entity:` node when no code node exists (e.g. a service named in prose but not in `src/`).
- Canonicalization is where cross-referencing happens: `65.21.4.203` and `prod (65.21.4.203)` normalize to the **same** node id, so both articles attach to one entity.

---

## 4. Plan — three slices, each shippable alone

Follow the `mhb-feature` sizing discipline: small slices, tests per slice, no slice N+1 until N is green.

### Slice 1 — the extractor module (no graph wiring yet)
- New file `scripts/entities.py`, modeled on `crosslink.py`'s structure:
  - `RECOGNIZERS: dict[str, Recognizer]` — one compiled regex + normalizer per type.
  - `extract(body: str) -> list[Entity]` — masks non-prose (reuse `crosslink.mask_non_prose`), runs recognizers, returns `[{type, canonical_id, surface, pos}]`, de-duplicated per article.
  - CLI dry-run: `uv run python scripts/entities.py` prints, per article, the entities found — **for eyeballing the false-positive rate before any wiring.**
- Tests: `tests/test_entities.py` — one positive + one masked-negative (inside code fence, inside `[src:]`) per recognizer; a canonicalization test (`65.21.4.203` two ways → one id).
- **Ships value alone:** a standalone "what named things does the KB talk about?" report, zero risk to the graph.

### Slice 2 — wire into the unified graph
- In `unified_graph.build()` Pass-2 (the `for src_id, content in article_contents.items()` loop), call `entities.extract(stripped)` and, per hit:
  - `entity_id = hit.canonical_id`; if the hit resolves to an existing `file:`/`class:` node, reuse that id instead.
  - mint `nodes[entity_id] = {"kind": "entity", "entity_type": t, "label": surface}` if absent.
  - `edges.append({"from": src_id, "to": entity_id, "kind": "mentions"})`.
- Extend the node-id-prefix contract comment (currently `article:`/`file:`/`class:`/`symbol:`/`template:`) to include `entity:`.
- Tests: build a tiny graph from two fixture articles that both mention one host → assert a single shared `entity:` node with two inbound `mentions` edges.
- **Cache:** the graph signature already hashes node ids + edge endpoints, so new entity nodes bust the cache correctly — no cache change needed. Verify this in the test.

### Slice 3 — surface it
- `salience.py` / `get_salience` pick up entity nodes **for free** (they operate on the graph) — add one test asserting a highly-mentioned entity ranks as a hub.
- New MCP tool `find_entity(query, entity_type=None)` on the code-intel server: fuzzy-match against entity labels, return the entity node + its `mentions` neighbourhood (the articles/files that reference it). Thin wrapper over `get_unified_neighbors`.
- Optional, defer: an `entity_type` filter on `search_knowledge`.
- Docs: one row each in the README "Code Intelligence MCP" table + a short "Entity Cross-Referencing" section; a line in `AGENTS.md`.

---

## 5. Risks & guards

- **False entities pollute the graph** (same failure mode as false wikilinks). Guard: closed vocabulary, conservative regexes, mandatory dry-run eyeball (Slice 1) before wiring (Slice 2). Ship only the recognizers whose measured false-positive rate is near zero.
- **Hub explosion.** A universal token (e.g. `envvar:ANTHROPIC_API_KEY` appearing everywhere) becomes a mega-hub that swamps salience — the exact `EntityManagerInterface::flush` problem already documented in the salience "Bridges" logic. Guard: salience's degree-normalization already handles this; add a per-type stop-list for tokens that appear in > N% of articles.
- **Overlap with the code layer.** Prefer existing `file:`/`class:` nodes over parallel `entity:` nodes (see §3) so we don't split one real thing across two ids.

---

## 6. What we are explicitly NOT taking from OpenAleph

- Elasticsearch / Postgres / Redis / Docker — wrong weight, wrong job. Our local ChromaDB + markdown store stays.
- Structured tabular ingest (CSV/XLS/SQL) — no use case here.
- The full FollowTheMoney schema — we borrow the *idea* (closed entity vocabulary + cross-referencing), not its ~40 types.

---

## 7. Cost & next step

Pure Python, zero LLM cost, three small slices. **Not yet ingested into the KB** — this is a tooling design note, not app research, so it lives here rather than under `knowledge/research/`. Next step: build Slice 1 and run its dry-run against the live KB to measure the false-positive rate before committing to the vocabulary.
