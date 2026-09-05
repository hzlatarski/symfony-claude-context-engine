# Decay Stage — Design Spec

**Status:** proposed (NOT built — spec only). **LLM cost:** $0 — pure Python, same class as `lint.py` / `salience.py` / `crosslink.py` / `entities.py`.
**Date:** 2026-09-05.
**Origin:** the five-stage memory-engineering framing in [@0xWast3's post](https://x.com/0xWast3/status/2084625810112032849) (Capture · Consolidate · Retrieve · Reconcile · **Decay**), reconciled against this engine's own standing gap note `concepts/memory-compiler-retention-eviction`.

---

## 1. Problem — the half-done stage

A memory store that never forgets degrades into an unsearchable landfill: stale context competes for retrieval space on equal footing with what matters today. Of the five stages, this engine already ships four with real mechanisms:

| Stage | Existing mechanism | State |
|---|---|---|
| Capture | `flush.py` → `compile.py` select what becomes an article | ✅ |
| Consolidate | `compile.py` merges; `crosslink.py` de-dups links | ✅ |
| Retrieve | slim `search_knowledge`, RRF, priority-scored `compiled-truth.md` | ✅ |
| Reconcile | contradiction quarantine, skeptical compile prompt, `CONTRADICTION:` flag | ✅ |
| **Decay** | 90-day confidence half-life **weights notes down in `compiled-truth.md`** | ⚠️ **half-done** |

The gap, stated by the engine's own KB (`memory-compiler-retention-eviction`): *"the knowledge tree grows without bound. `compile_truth.py` ranks low-confidence articles toward the bottom … but no article is ever moved or archived."* So the **soft** half of decay exists (ranking); the **hard** half (eviction from the live retrieval surface) does not. Critically, `search_knowledge` still returns faded notes on equal footing — the confidence weighting only shapes `compiled-truth.md`, not the vector search. This spec closes that.

---

## 2. Locked design decisions (from brainstorming, 2026-09-05)

1. **Staged fade: soft → hard.** A note first sinks in live search (soft); if it stays faded longer, it is archived out of the live surface (hard). Not violent deletion.
2. **Combined decay score.** Blend age + decayed-confidence + unused (retrieval feedback) + superseded — the article's "unused, unreinforced, or expired".
3. **Protection: type defaults + manual pin.** `type: decision` and `type: preference` never fade unless superseded; `pinned: true` force-keeps any note.
4. **Reversible: opt-in searchable archive + auto-revive.** Archived notes are hidden from normal search, reachable with `include_archived=true`, and climb back to live automatically when rated `useful` or newly cited/linked.

**Explicitly out of scope:** entity nodes (already self-heal — rebuilt from prose each graph build), and the byte-for-byte raw daily transcript archive (kept lossless on purpose — decay never touches it).

---

## 3. The decay score

A pure function over one article's frontmatter + retrieval-outcome record. All weights and thresholds live in `config.py` so they are tunable per project without code change.

```
# Protections are applied BEFORE scoring — a protected note scores 0 (immune).
if pinned:                                   decay_score = 0.0
elif type in {decision, preference} and not superseded:
                                             decay_score = 0.0
else:
    decay_score = clamp01(
        W_AGE        * age_factor          # time since `updated`, normalized on a half-life
      + W_CONF       * (1 - confidence)     # `confidence` is ALREADY 90-day-decayed upstream
      + W_UNUSED     * unused_factor        # time since last `useful` retrieval; + dead_end weight
      + W_SUPERSEDED * superseded_factor    # 1.0 if another note supersedes/refutes this one
    )
```

Signal sources (all already collected — no new capture):

| Factor | Source | Notes |
|---|---|---|
| `age_factor` | `updated:` frontmatter | normalized `1 - 2^(-age_days / AGE_HALFLIFE_DAYS)` |
| `confidence` | `confidence:` frontmatter | upstream confidence-decay already ran; reuse it, don't re-decay |
| `unused_factor` | `record_retrieval_outcome` log (`reflect.py`'s input) | days since last `useful`; `dead_end`/`corrected` add weight |
| `superseded_factor` | inbound `[[..]]{supersedes}` / `{refutes}` wikilinks, or contradiction-quarantine state | 1.0 when a newer note replaces this one |

**Two thresholds + dwell:**
- `SOFT_LINE` (e.g. 0.50) → note is *soft-decayed*: a heavy rank penalty in live retrieval, still returned.
- `HARD_LINE` (e.g. 0.75) **held for `HARD_DWELL_DAYS`** (e.g. 30) → note is *archived*. The dwell prevents a transient dip from evicting a note; only sustained fade archives it.

---

## 4. State (frontmatter fields)

Decay is **flag-in-place** — archived notes stay in `concepts/` so their node id, `[[wikilinks]]`, and `[src:]` graph edges never break. Only metadata changes.

```yaml
decay_score: 0.82        # last computed score (informational)
decay_stage: archived    # live | soft | archived
archived: true           # convenience boolean = (decay_stage == archived)
archived_at: 2026-09-05  # when it crossed hard+dwell
pinned: false            # manual force-keep (author-set, never auto-written)
```

`pinned` is author-owned; the pass never writes it. Everything else is pass-owned and idempotent.

---

## 5. Components

Each unit has one purpose, a clear interface, and is testable in isolation.

- **`scripts/decay.py`** (new) — the pass.
  - `score_article(meta, outcomes, graph) -> float` — the pure scoring function above. Unit-testable with synthetic inputs, no I/O.
  - `classify(score, prior_stage, dwell_days) -> stage` — maps score+dwell → `live|soft|archived`.
  - `run(knowledge_root, apply=False)` — dry-run report by default (like `entities.py`/`crosslink.py`); `--apply` writes frontmatter. Prints, per note, current stage → proposed stage and why.
- **`config.py`** (edit) — `W_AGE`, `W_CONF`, `W_UNUSED`, `W_SUPERSEDED`, `AGE_HALFLIFE_DAYS`, `SOFT_LINE`, `HARD_LINE`, `HARD_DWELL_DAYS`. One source of truth.
- **Retrieval integration** (edit `vector_store` / `hybrid_search` + `compile_truth.py`):
  - Reindex stamps `archived` + `decay_stage` into Chroma metadata.
  - Live `search_knowledge` filters out `archived` by default; `include_archived=true` includes them (tagged). Soft-decayed notes get a score penalty, not exclusion.
  - `compile_truth.py` skips `archived` (it already skips quarantined — same seam).
- **Revive** (edit the retrieval-outcome path + `crosslink.py`/`compile.py`):
  - `record_retrieval_outcome(slug, "useful")` on an archived slug → clear archive (back to `live`), stamp `revived_at`.
  - When a link/consolidation pass detects a **new inbound** `cites`/`wikilink` to an archived slug → revive it.
- **`kb_health`** (edit) — add a `decay` block: counts by stage, oldest archived, revive count. One-glance health of the stage.
- **MCP** (optional, later) — `decay_status()` read-only summary; mirrors `ingest_status` style.

---

## 6. Lifecycle

```
        (compute each pass)
new note ──► live ──► score ≥ SOFT_LINE ──► soft  (rank-penalized in live search)
                          │                    │
                          │                    ▼
                          │        score ≥ HARD_LINE, held HARD_DWELL_DAYS
                          │                    │
                          ▼                    ▼
                    score drops           archived  (hidden from live search;
                    back < SOFT_LINE        visible only via include_archived)
                          │                    │
                          ▼                    │  rated `useful`  OR  new inbound cite/link
                        live  ◄────────────────┘  (auto-revive)

Protected (pinned, or live decision/preference): pinned to `live`, never scored up.
```

---

## 7. CLI

```bash
uv run python scripts/decay.py                 # dry-run report — what WOULD change, nothing written
uv run python scripts/decay.py --explain foo   # per-factor score breakdown for one slug
uv run python scripts/decay.py --apply          # write decay_stage/archived frontmatter
uv run python scripts/decay.py restore <slug>   # manual un-archive (escape hatch)
```

Dry-run is the default, matching every other write-capable pass in this engine. You see the fade list before anything moves.

---

## 8. Proposed slices (for a later build — NOT part of this spec's delivery)

Sized per the `mhb-feature` discipline: small, each with tests, each shippable alone.

1. **Scoring + dry-run report.** `score_article`/`classify` + `decay.py` report. Pure, no writes. Ships as "what would fade?" visibility — the safe measurement pass, like entity Slice 1.
2. **State write + protections + restore.** `--apply` writes frontmatter; protections (pin + type) enforced; `restore` command. Tests: a pinned/decision note never archives; dwell prevents a one-off dip from archiving.
3. **Retrieval integration.** Reindex stamps metadata; `search_knowledge` + `compile_truth` skip archived; `include_archived` flag. Tests: archived note absent by default, present with the flag; soft note ranks last.
4. **Auto-revive + kb_health.** Revive on `useful` / new inbound link; `kb_health` decay block. Tests: an archived note rated useful returns to live; a new wikilink revives its target.

---

## 9. Risks & guards

- **Wrongly fading a live-critical note.** Guards: protections (§2.3), the `HARD_DWELL_DAYS` hold, dry-run-by-default, and full reversibility (auto-revive + manual `restore`). Nothing is deleted — only flagged.
- **Graph breakage.** Avoided by flag-in-place: archived notes keep their slug/node id, so `[[wikilinks]]` and `[src:]` edges stay valid. (Rejected alternative: physically moving files to `knowledge/archive/`, which changes slugs and breaks inbound links.)
- **Threshold mis-tuning.** All weights/lines in `config.py`; `--explain` shows the per-factor breakdown so tuning is evidence-based. Ship conservative lines first; watch the dry-run.
- **Thrash (revive ↔ archive flip-flop).** The soft stage + dwell create hysteresis; a revived note re-enters at `live`, and needs the full dwell again before it can re-archive.
- **Provenance retention** (raised by the post's top comment). `[src:]` anchors are never decayed and travel with the note into archive — the store stays auditable.

---

## 10. Open tuning questions (defer to build, decide from real dry-run data)

- Starting values for the four weights and three thresholds — set from a dry-run over the live KB, not guessed.
- Whether `corrected` retrieval outcomes should count toward `unused_factor` or trigger reconciliation instead.
- Whether a separate Chroma `archived` collection is worth it vs. a metadata filter on the live collection (default: metadata filter — simpler, one store).
