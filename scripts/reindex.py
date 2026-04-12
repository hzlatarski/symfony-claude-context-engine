"""Reindex articles and daily logs into the ChromaDB vector store.

Modes
-----
    --all            Force re-embed everything (ignores hash cache).
    --articles-only  Skip daily-log chunks (fast when only articles changed).
    --daily-only     Skip curated articles (fast after a flush).
    (default)        Incremental — only re-embed files whose hash changed.

State
-----
Hash cache lives in ``state.json`` under two keys that are created on first
run:

    "vector_article_hashes": {"concepts/foo": "abc123...", ...}
    "vector_daily_hashes":   {"daily/2026-04-12.md": "def456...", ...}

A file hash changes ⇒ its slug is embedded (or re-embedded) and the cache
is updated. Delete the cache entries manually if you suspect drift between
the vector store and the cache and want a targeted rebuild without ``--all``.

Chunking the daily logs requires ``scripts/chunk_daily.py`` which lands in
Task 5 of the steal-list plan. Until that file exists, the ``--daily-only``
path will raise ImportError — that's intentional, ``reindex_daily`` gates
the import locally so module-level imports still succeed.
"""
from __future__ import annotations

import argparse
import sys

from config import KNOWLEDGE_DIR
from utils import (
    embed_article_file,
    embed_daily_file,
    file_hash,
    list_raw_files,
    list_wiki_articles,
    load_contradictions,
    load_state,
    save_state,
)
from vector_store import stats as vector_stats


def reindex_articles(force: bool = False) -> tuple[int, int]:
    """Re-embed changed (or all, if force) articles via ``embed_article_file``.

    Returns ``(embedded, skipped)``. State is always persisted via
    ``try/finally`` so a mid-run exception doesn't lose the hash cache
    for articles that were successfully embedded before the failure.
    """
    state = load_state()
    vector_hashes = state.setdefault("vector_article_hashes", {})
    quarantined = load_contradictions()  # hoist outside the loop

    embedded = 0
    skipped = 0

    try:
        for article in list_wiki_articles():
            rel = str(article.relative_to(KNOWLEDGE_DIR)).replace("\\", "/")
            slug = rel.removesuffix(".md")
            h = file_hash(article)

            if not force and vector_hashes.get(slug) == h:
                skipped += 1
                continue

            embed_article_file(article, quarantined=quarantined)
            vector_hashes[slug] = h
            embedded += 1
    finally:
        save_state(state)

    return embedded, skipped


def reindex_daily(force: bool = False) -> tuple[int, int]:
    """Chunk + embed changed daily logs via ``embed_daily_file``.

    Returns ``(embedded_files, skipped_files)``. Raises
    ``ModuleNotFoundError`` (name=``"chunk_daily"``) if Task 5 hasn't
    landed yet — ``main()`` catches this specifically.
    """
    state = load_state()
    vector_hashes = state.setdefault("vector_daily_hashes", {})

    embedded = 0
    skipped = 0

    try:
        for daily in list_raw_files():
            rel = str(daily.relative_to(KNOWLEDGE_DIR)).replace("\\", "/")
            h = file_hash(daily)

            if not force and vector_hashes.get(rel) == h:
                skipped += 1
                continue

            embed_daily_file(daily)
            vector_hashes[rel] = h
            embedded += 1
    finally:
        save_state(state)

    return embedded, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Reindex knowledge into ChromaDB")
    parser.add_argument("--all", action="store_true", help="Force re-embed everything")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--articles-only", action="store_true", help="Skip daily chunks")
    group.add_argument("--daily-only", action="store_true", help="Skip articles")
    args = parser.parse_args()

    if not args.daily_only:
        embedded, skipped = reindex_articles(force=args.all)
        print(f"Articles: embedded {embedded}, skipped {skipped}")

    if not args.articles_only:
        try:
            embedded, skipped = reindex_daily(force=args.all)
            print(f"Daily chunks: embedded {embedded} files, skipped {skipped}")
        except ModuleNotFoundError as exc:
            # Tight check via exc.name — string matching would accept
            # unrelated import failures inside a working chunk_daily.
            if getattr(exc, "name", None) == "chunk_daily":
                print(
                    "Daily reindex skipped: chunk_daily module not yet available "
                    "(lands in Task 5 of the steal-list plan).",
                    file=sys.stderr,
                )
            else:
                raise

    s = vector_stats()
    print(f"\nTotal in store: {s['articles']} article zones, {s['daily_chunks']} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
