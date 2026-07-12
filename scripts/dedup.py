"""Near-duplicate article detection.

The compiler prompt asks the LLM to "check the index for duplicates or
near-duplicates" before creating an article. That is an instruction, not a
mechanism, and it has been quietly failing: the knowledge base contains pairs at
cosine 1.000 (byte-identical articles under two slugs), plus CSRF and HeyGen
clusters where the same fact was written three times.

This module supplies the mechanism, in two halves:

* **Prevention** — ``similar_to_text()`` retrieves the closest existing articles
  for a source about to be compiled, so the compiler prompt can name them and
  say "update one of these instead of creating a near-duplicate". Cheap, and it
  attacks the problem before the article exists.
* **Detection** — ``find_near_duplicates()`` does an all-pairs cosine sweep over
  the article vectors already in Chroma and reports pairs above a floor. Pure
  numpy over local embeddings: no LLM, no network, no cost. Wired into lint.py
  as the ``near_duplicate`` check.

Deliberately does NOT auto-merge. Merging two articles means choosing which
facts survive, and a wrong automatic choice destroys knowledge silently — the
one failure this system cannot recover from. Detection is reported; a human (or
an explicitly-invoked LLM pass) decides.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict

import numpy as np

import vector_store as vs
from config import KNOWLEDGE_DIR

# Cosine floor for calling two articles near-duplicates.
#
# Calibrated against the live 776-vector index:
#   >=0.95 -> 3 pairs    (identical / near-identical text)
#   >=0.90 -> 10 pairs   (all genuine duplicates on inspection)
#   >=0.88 -> 12 pairs
#   >=0.85 -> 29 pairs   (starts admitting legitimately-distinct neighbours,
#                         e.g. scenarios/the-label vs concepts/label-archetype)
# 0.88 keeps precision high enough that every hit is worth a human look, which
# is the bar for a lint check that must not cry wolf.
DEFAULT_THRESHOLD = 0.88

# Above this, the two articles are effectively the same text. Reported
# separately because these need no judgement call — one of them is redundant.
IDENTICAL_THRESHOLD = 0.97


@dataclass(frozen=True)
class DuplicatePair:
    """Two articles whose Observed-zone text is near-identical."""

    slug_a: str
    slug_b: str
    similarity: float

    @property
    def identical(self) -> bool:
        return self.similarity >= IDENTICAL_THRESHOLD

    def __str__(self) -> str:
        mark = "IDENTICAL" if self.identical else "near-dup "
        return f"{mark} {self.similarity:.3f}  {self.slug_a}  <->  {self.slug_b}"


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows so a dot product is a cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Guard the zero vector: an empty/failed embedding must not divide by zero
    # and must not come out looking similar to everything.
    norms[norms == 0] = 1e-10
    return matrix / norms


def article_exists(slug: str) -> bool:
    """True when ``slug`` still has a file backing it on disk."""
    return (KNOWLEDGE_DIR / f"{slug}.md").exists()


def find_stale_vectors() -> list[str]:
    """Indexed slugs whose article file is gone — ghosts in the vector store.

    Deleting or renaming an article does not remove its embedding unless
    ``delete_article`` is called, so the index accumulates slugs that no longer
    resolve. They are worse than useless: ``search_knowledge`` happily returns
    one, ``get_article`` then fails, and the agent burns a turn on a dead end.
    They also fabricate duplicate pairs against articles that no longer exist.
    """
    collection = vs._articles_collection()
    got = collection.get(include=[])
    indexed = {i.rsplit("::", 1)[0] for i in (got.get("ids") or [])}
    return sorted(slug for slug in indexed if not article_exists(slug))


def prune_stale_vectors() -> list[str]:
    """Drop every ghost vector. Returns the slugs removed."""
    stale = find_stale_vectors()
    for slug in stale:
        vs.delete_article(slug)
    return stale


def _load_observed_vectors() -> tuple[list[str], np.ndarray]:
    """Fetch the Observed-zone article vectors from Chroma.

    Only the Observed zone: Synthesized text is the compiler's own inference and
    two articles can legitimately share it without being duplicates.

    Ghost vectors (slug indexed, file gone) are dropped here so they cannot
    manufacture duplicate pairs against articles that no longer exist.
    """
    collection = vs._articles_collection()
    got = collection.get(
        include=["embeddings", "metadatas"],
        where={"zone": {"$eq": "observed"}},
    )
    ids = got.get("ids") or []
    raw = got.get("embeddings")
    if not ids or raw is None or len(raw) == 0:
        return [], np.zeros((0, 0), dtype=np.float32)

    slugs: list[str] = []
    keep: list[int] = []
    for idx, id_ in enumerate(ids):
        slug = id_.replace("::observed", "")
        if not article_exists(slug):
            continue
        slugs.append(slug)
        keep.append(idx)

    if not slugs:
        return [], np.zeros((0, 0), dtype=np.float32)

    embeddings = np.asarray(raw, dtype=np.float32)[keep]
    return slugs, embeddings


def find_near_duplicates(
    threshold: float = DEFAULT_THRESHOLD,
) -> list[DuplicatePair]:
    """All-pairs cosine sweep over article vectors; pairs at/above ``threshold``.

    Returned highest-similarity first. Each unordered pair appears once.
    """
    slugs, embeddings = _load_observed_vectors()
    if len(slugs) < 2:
        return []

    unit = _normalize(embeddings)
    sims = unit @ unit.T

    # Threshold FIRST, then take the strict upper triangle of the boolean mask.
    # np.triu_indices(n, k=1) would materialize two int64 arrays of n²/2 entries
    # whether or not anything matches — ~1.2 GB at 10k articles, on top of the
    # similarity matrix itself. Masking keeps the extra allocation proportional
    # to the number of *hits*, which is tiny.
    hits = np.argwhere(np.triu(sims >= threshold, k=1))

    pairs = [
        DuplicatePair(
            slug_a=slugs[i],
            slug_b=slugs[j],
            # float() so the value is JSON-serializable, not np.float32
            similarity=float(sims[i, j]),
        )
        for i, j in hits
    ]
    return sorted(pairs, key=lambda p: p.similarity, reverse=True)


# The embedding model (all-MiniLM-L6-v2) truncates its input at 256 tokens —
# roughly 1,000 characters. Handing it a whole daily log as one query would
# silently embed only the first session of the day and ignore everything after
# it, so a source is chunked and the per-chunk hits are unioned.
QUERY_CHUNK_CHARS = 900
MAX_QUERY_CHUNKS = 12


def _chunk_for_query(text: str, chunk_chars: int = QUERY_CHUNK_CHARS) -> list[str]:
    """Split a source into embedding-sized chunks, preferring section breaks."""
    text = text.strip()
    if len(text) <= chunk_chars:
        return [text] if text else []

    chunks: list[str] = []
    remaining = text
    while remaining and len(chunks) < MAX_QUERY_CHUNKS:
        if len(remaining) <= chunk_chars:
            chunks.append(remaining)
            break
        head = remaining[:chunk_chars]
        # Prefer a markdown heading break, then a paragraph break, so a chunk
        # lands on a coherent topic rather than mid-sentence.
        cut = max(head.rfind("\n#"), head.rfind("\n\n"))
        if cut < chunk_chars // 2:
            cut = chunk_chars
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()

    return [c for c in chunks if c]


def similar_to_text(text: str, limit: int = 5, threshold: float = 0.55) -> list[dict]:
    """Existing articles most similar to ``text`` — the prevention half.

    Called before compiling a source so the prompt can name the articles this
    source is most likely to duplicate. The floor is intentionally much looser
    than the detection threshold: here a false positive merely shows the model
    one extra candidate to consider, while a false negative lets a duplicate
    through. Cheap to be generous.

    The source is chunked (see QUERY_CHUNK_CHARS) and the best hit per article
    across all chunks is kept, so a topic discussed late in a long daily log is
    still matched.
    """
    if not text.strip():
        return []

    best: dict[str, dict] = {}

    for chunk in _chunk_for_query(text):
        for r in vs.search_articles(query=chunk, limit=limit, zone_filter="observed"):
            # Chroma returns cosine *distance*; similarity is 1 - distance.
            distance = r.get("distance")
            similarity = 1.0 - distance if distance is not None else None
            if similarity is not None and similarity < threshold:
                continue

            meta = r.get("metadata") or {}
            slug = str(r.get("slug") or r.get("id", "")).replace("::observed", "")
            candidate = {
                "slug": slug,
                "title": meta.get("title", ""),
                "similarity": similarity,
            }

            prior = best.get(slug)
            if prior is None or _sim_key(candidate) > _sim_key(prior):
                best[slug] = candidate

    ranked = sorted(best.values(), key=_sim_key, reverse=True)
    return ranked[:limit]


def _sim_key(candidate: dict) -> float:
    """Sort key that tolerates a missing similarity (Chroma omits distances rarely)."""
    value = candidate.get("similarity")
    return value if isinstance(value, (int, float)) else -1.0


def format_preflight_block(candidates: list[dict]) -> str:
    """Render similar-article candidates for injection into a compiler prompt."""
    if not candidates:
        return ""

    lines = [
        "## Closest existing articles (possible duplicates)",
        "",
        "These existing articles are semantically closest to the source you are "
        "about to compile. Before creating a NEW article, check whether the "
        "knowledge belongs in one of these — prefer UPDATING an existing article "
        "over creating a near-duplicate. Use `Read` to inspect any that look close.",
        "",
    ]
    for c in candidates:
        sim = c.get("similarity")
        sim_text = f"{sim:.2f}" if isinstance(sim, float) else "?"
        title = f" — {c['title']}" if c.get("title") else ""
        lines.append(f"- `{c['slug']}` (similarity {sim_text}){title}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find near-duplicate knowledge articles (zero-cost, local embeddings)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Cosine floor for reporting a pair (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a human-readable report",
    )
    parser.add_argument(
        "--prune-stale",
        action="store_true",
        help="Delete vectors whose article file no longer exists, then exit",
    )
    args = parser.parse_args()

    if args.prune_stale:
        removed = prune_stale_vectors()
        if not removed:
            print("No stale vectors — every indexed slug has a file on disk.")
            return 0
        print(f"Pruned {len(removed)} stale vector(s):")
        for slug in removed:
            print(f"  - {slug}")
        return 0

    stale = find_stale_vectors()
    pairs = find_near_duplicates(threshold=args.threshold)

    if args.json:
        print(json.dumps(
            {"near_duplicates": [asdict(p) for p in pairs], "stale_vectors": stale},
            indent=2,
        ))
        return 1 if (pairs or stale) else 0

    if stale:
        # ASCII only: this prints to a Windows console under cp1252, where an
        # em-dash comes out as a replacement char.
        print(
            f"{len(stale)} stale vector(s) - indexed but no file on disk "
            "(search can return these; get_article then fails):"
        )
        for slug in stale:
            print(f"  - {slug}")
        print("  Fix: uv run python scripts/dedup.py --prune-stale")
        print()

    if not pairs:
        print(f"No near-duplicate articles at cosine >= {args.threshold}.")
        return 1 if stale else 0

    identical = [p for p in pairs if p.identical]
    print(f"Found {len(pairs)} near-duplicate pair(s) at cosine >= {args.threshold}")
    if identical:
        print(f"  {len(identical)} of them are effectively identical (>= {IDENTICAL_THRESHOLD})")
    print()
    for pair in pairs:
        print(f"  {pair}")
    print()
    print("Merge by hand: fold the weaker article's facts into the stronger one,")
    print("replace the loser with a [[wikilink]] redirect or delete it, then re-run")
    print("`uv run python scripts/reindex.py --all` to drop its stale vector.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
