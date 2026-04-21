"""Return top-k scenario matches as JSON for the training-dedup pipeline.

Shells out from the Symfony KnowledgeSearchClient (see
src/Service/TrainingDedup/KnowledgeSearchClient.php). The PHP side depends
on the output shape — keep the emitted JSON stable:

    [
      {
        "slug": "scenarios/foo/bar",
        "title": "Foo Bar",
        "snippet": "first 400 chars of the article text…",
        "distance": 0.12,
        "metadata": {...full chroma metadata dict...}
      },
      ...
    ]

The memory compiler's ChromaDB `articles` collection does NOT persist a
`source_category` metadata field (see utils.py::reindex_article_into_vector_store
— only type/confidence/quarantined/updated/pinned are stored). So we cannot
filter by category at query time. Instead we identify scenarios by their
slug prefix (`scenarios/…`) because the ingest path uses the article's
relative path under knowledge/ as the slug — every scenario file lives at
knowledge/scenarios/**/*.md and therefore has a slug starting with
"scenarios/".

We pull a wider candidate pool from Chroma (SEARCH_MULTIPLIER * top_k) and
then post-filter to the scenario slugs, trimming to top_k. If the
collection has no scenarios yet (the 93-scenario export in Task 3 hasn't
run), we emit [] and exit 0 — empty is a legitimate state, not an error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make sibling modules importable when invoked as
#   uv run python scripts/search_scenarios.py
# from the memory-compiler root.
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SCENARIO_SLUG_PREFIX = "scenarios/"

# Pull this many times top_k from Chroma before post-filtering to scenarios.
# 5x gives enough headroom that even when scenarios are a minority of the
# articles collection we still get top_k scenario hits back, without being
# so wide that query latency suffers.
SEARCH_MULTIPLIER = 5

# Max chars of article body to include in each hit's snippet. The PHP
# adjudicator only needs a few sentences for the LLM prompt — full bodies
# would bloat the subprocess stdout for no gain.
SNIPPET_CHARS = 400


def _to_snippet(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= SNIPPET_CHARS:
        return text
    return text[:SNIPPET_CHARS].rstrip() + "…"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search the memory-compiler's ChromaDB articles collection "
                    "and return the top-K scenario matches as JSON on stdout.",
    )
    parser.add_argument("--query", required=True, help="Search text")
    parser.add_argument("--top-k", type=int, default=10, help="Max results (default 10)")
    args = parser.parse_args()

    # Empty query is still valid — ChromaDB handles it and returns whatever
    # the nearest-neighbor search picks up. Don't bail here; let the store
    # decide.
    try:
        import vector_store
    except ImportError as e:
        # Fatal — the memory-compiler package is broken or missing.
        # Surface loudly so the PHP caller raises instead of pretending
        # there were simply no matches.
        import traceback
        print(
            f"search_scenarios.py: failed to import vector_store: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        return 1

    pool_size = max(args.top_k, args.top_k * SEARCH_MULTIPLIER)

    try:
        raw = vector_store.search_articles(
            query=args.query,
            limit=pool_size,
        )
    except Exception as e:
        # Real failure (ChromaDB down, fastembed model missing, collection
        # corrupt). A genuinely empty collection does NOT raise here —
        # Chroma returns [] and we fall through to post-filter as usual.
        # So any exception reaching this block must be surfaced, not
        # swallowed, or the adjudicator will mis-classify every candidate
        # as novel and import duplicates.
        import traceback
        print(
            f"search_scenarios.py: vector_store error: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        return 1

    out: list[dict] = []
    seen_slugs: set[str] = set()
    for hit in raw or []:
        slug = hit.get("slug") or ""
        if not slug.startswith(SCENARIO_SLUG_PREFIX):
            continue
        # Dedupe by slug — one article can appear twice when both its
        # observed and synthesized zones score high.
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)

        meta = hit.get("metadata") or {}
        out.append({
            "slug": slug,
            "title": meta.get("title") or slug.rsplit("/", 1)[-1],
            "snippet": _to_snippet(hit.get("text") or ""),
            "distance": float(hit.get("distance") or 0.0),
            "metadata": meta,
        })
        if len(out) >= args.top_k:
            break

    json.dump(out, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
