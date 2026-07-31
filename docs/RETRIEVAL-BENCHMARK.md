# Retrieval benchmark (LongMemEval-S)

Until this existed, every retrieval knob in the compiler — the RRF constant,
the pool multiplier, the tokenizer, the embedding model, chunk sizing — was
set by argument rather than by measurement. This harness turns those into
numbers you can regress against.

Adapted from the benchmark in [rohitg00/agentmemory](https://github.com/rohitg00/agentmemory)
(Apache-2.0). Only the *protocol* was taken; the code is ours, in Python,
and the dataset is upstream of that project. See
`knowledge/research/agentmemory-evaluation.md` for what else was and wasn't
worth taking.

## What it measures

Retrieval only — no answer generation. Each LongMemEval-S question ships its
own haystack of ~50 chat sessions, one or more of which are the gold
evidence. The harness builds a throwaway index over *that question's*
haystack, searches it with the question text, and scores whether gold
surfaced.

| Metric | Meaning |
|---|---|
| `recall@k` | Did **any** gold session make the top *k*? Binary, averaged. The headline. |
| `ndcg@10` | Did gold rank *high*, not merely inside the cutoff? Binary relevance. |
| `rr` | Reciprocal rank of the first gold hit (truncated to `depth` — see caveat). |

## Running it

The dataset is not vendored (278 MB). Fetch it once:

```bash
curl -L -o /c/tmp/longmemeval_s.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s
```

Then:

```bash
cd .claude/memory-compiler
unset VIRTUAL_ENV && PATH="$PATH:/c/Users/WindMaster/AppData/Roaming/Python/Python313/Scripts"

# keyword baseline — no embeddings, ~20s for all 500
uv run python scripts/eval_longmemeval.py --dataset /c/tmp/longmemeval_s.json --mode bm25

# full pipeline — embeds every session, ~15 min
uv run python scripts/eval_longmemeval.py --dataset /c/tmp/longmemeval_s.json --mode hybrid

# size a run first, and keep the per-question rankings
uv run python scripts/eval_longmemeval.py --dataset ... --mode hybrid --limit 50 --json out.json

# chunk long documents before embedding — better on every metric, ~10x slower
uv run python scripts/eval_longmemeval.py --dataset ... --mode hybrid --chunk-words 150
```

`--mode bm25` is the right default for a quick regression check after a
tokenizer or ranking change. `--mode hybrid` is the number that reflects
live search.

## Baseline — 2026-07-31, all 500 questions

Embedder `all-MiniLM-L6-v2`, RRF k=60, pool multiplier 3.

| Mode | recall@5 | recall@10 | recall@20 | ndcg@10 | rr@20 | runtime |
|---|---|---|---|---|---|---|
| bm25 | 0.9640 | 0.9760 | 0.9920 | **0.8920** | **0.9077** | 16s |
| vector | 0.9240 | 0.9680 | 0.9840 | 0.8248 | 0.8260 | 921s |
| **hybrid** | **0.9680** | **0.9860** | **0.9980** | 0.8902 | 0.8896 | 958s |

Protocol deviations, identical across all three runs: 15 collapsed duplicate
sessions, 1228 blank sessions excluded from the vector index.

Hybrid wins every recall cutoff — the result RRF is supposed to produce:
neither stream dominates and fusing them beats both. But read the last two
columns before celebrating. BM25 alone edges hybrid on **both** NDCG@10 and
rr@20, meaning when BM25 finds the gold session it ranks it *higher*; hybrid
simply finds it more often. Fusion is buying coverage, not precision, at 60×
the runtime. For a latency-sensitive path, BM25 alone is defensible.

For reference, the numbers agentmemory publishes for the same dataset are
BM25 0.862 / hybrid 0.952 / vector 0.966 at recall@5. Our BM25 is 10 points
above theirs and our hybrid beats their hybrid; our vector-only trails their
0.966, which is consistent with them chunking before embedding (see below).

### Chunking

`all-MiniLM-L6-v2` truncates at roughly 190 words, but LongMemEval sessions
run a median of 1633 — so a whole-session embedding represents only the
opening, and anything stated later is invisible to vector search.
`--chunk-words 150 --chunk-overlap 30` splits each document into overlapping
windows, embeds each, and ranks a document by its best-matching window.

Hybrid, first 150 questions, same subset both rows:

| | recall@5 | recall@10 | ndcg@10 | rr@20 | runtime |
|---|---|---|---|---|---|
| whole-session | 0.9467 | 0.9867 | 0.8512 | 0.8353 | ~285s |
| **chunked 150w/30o** | **0.9600** | **1.0000** | **0.8926** | **0.8904** | 2790s |

Better on every metric, and recall@10 is perfect. The gain is largest on the
*ranking* metrics (NDCG +4.1pp, rr +5.5pp) rather than raw recall, which is
what you would expect: chunking mostly helps documents the vector stream
already reached but ranked poorly, by letting the matching passage speak for
the document instead of its first paragraph.

The cost is roughly 10× the runtime, because each document becomes ~11
embeddings. Chunking applies to the vector index only — BM25 reads whole
documents and has no window limit, so its score is unchanged (there is a
test pinning that).

**Sanity controls** (rerun these if a number ever looks too good):

| Control | recall@5 | Expected |
|---|---|---|
| Random ranking | 0.180 | ≈ chance rate, 0.191 |
| Real ranking vs. a deliberately wrong gold id | 0.060 | Well below chance |

## Design: it must measure *us*

The harness borrows the algorithms rather than reimplementing them:

- keyword ranking via `bm25_store.TokenIndex` — the project tokenizer and
  the negative-IDF presence gate;
- vector ranking via Chroma with the same default ONNX embedder
  (`all-MiniLM-L6-v2`) the persistent store uses;
- fusion via `hybrid_search.fuse_rankings` — the same RRF constant (k=60)
  and the same oversample and first-seen dedup as live search.

Both `TokenIndex` and `fuse_rankings` were extracted from previously inlined
code specifically so there is one implementation, not two. A harness that
copies the ranking code measures the copy.

### What it does *not* measure

Be precise about the claim: this exercises the **ranking primitives**, not
the whole of `search_articles`. Specifically out of scope —

- **Article representation.** Live BM25 indexes one document per Truth zone,
  folds the article title into the indexed text, and keys on `slug::zone`.
  The benchmark indexes one flattened chat session with no title and no zones.
- **Metadata filtering.** Live search applies confidence, type, zone, and
  quarantine filters to both streams. The benchmark has no equivalent, so a
  filter regression would not show up here.
- **Chunking and ingestion.** Sessions are indexed whole; the compiler's own
  daily-log chunking is not in the path.

So a good score here means the tokenizer, the negative-IDF gate, and the RRF
fusion are sound over long conversational text. It does not certify that
live knowledge-base search is healthy — see the title-indexing trap below
for a regression that this benchmark would happily have scored 0.966 through.

## Traps this harness already fell into

Recorded because both produced confident, plausible, wrong numbers.

**1. `chromadb.EphemeralClient()` is not isolated.** It resolves through a
shared in-process system client, so two clients requesting the same
collection name get the *same* collection. The first version of
`eval_corpus.py` used a fixed name, which merged all 500 haystacks into one
growing pool — vector recall decayed as the run advanced and reported
**0.354 instead of the true 0.920**, which also dragged hybrid below BM25.
Worse than being wrong, it was wrong in a *plausible* direction: it told a
tidy story ("our embedder is weak, fusion hurts") that survived a whole
follow-up experiment before the leak surfaced. Fixed with a per-corpus UUID
collection name plus an explicit `close()`; guarded by
`TestIsolationBetweenCorpora`. If you build any other ephemeral Chroma index
in this project, do the same.

**2. `ingest.py` truncated state.json to `{}` on every run.** Its migration
mutator called `migrate_state_schema(current)`, which mutates in place and
returns *the same object*, then did `current.clear()` — emptying the
"migrated" copy too — and updated from the now-empty dict. Every ingest wiped
the ingest history, which then made the next run re-process all 497 sources
at LLM cost. It destroyed 496 source records and 83 daily records twice in
one week before being found. Guarded by
`TestIngestMigrationDoesNotWipeState`; `tests/conftest.py` additionally
redirects `STATE_FILE` for every test and fails any test that touches the
real file.

**3. The BM25 index is built from title + body, not body.**
`_iter_article_zones` deliberately folds the article title into the token
stream while the stored `text` omits it. Rebuilding the index from `text`
silently drops every title term from live search — and the entire existing
suite still passed, because every other fixture happened to repeat title
words in its body. The record now carries an explicit `index_text` field;
guarded by `TestTitleIsIndexed`.

## Dataset notes

- 500 questions, 50.2 sessions per haystack on average, sessions a median
  1633 words long.
- Those sessions are far longer than `all-MiniLM-L6-v2`'s ~190-word window,
  so the vector stream only ever sees each session's opening. Restricting
  BM25 to that same window costs it 23 points (0.953 → 0.727 recall@5 on a
  150-question subset), which bounds how much information is being discarded
  — yet vector still reaches 0.924, so truncation is a real constraint here
  rather than a crippling one.
- **Chunking recovers most of it** — see below. It is now a real option
  (`--chunk-words`), not a scratch experiment.
- 15 of the 500 questions list one session id **twice**. Every observed case
  is byte-identical content and never a gold session, so the loader collapses
  them and prints a note. The same id with *differing* content raises instead
  — that would make "did we retrieve gold" ambiguous.
- The upstream dataset is now deprecated in favour of
  [longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned).
  The original is kept here because published numbers elsewhere refer to it.

## Caveat on MRR

Results are only materialized to `depth` (the widest cutoff), so `rr` is
truncated: gold ranked below `depth` contributes 0 rather than its true
reciprocal. Recall and NDCG at their cutoffs are exact.
