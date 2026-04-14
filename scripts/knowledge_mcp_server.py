"""Knowledge MCP Server — semantic retrieval over the compiled knowledge base.

A second MCP server (separate from the Symfony code-intel one at
``mcp_server.py``) that exposes the ChromaDB vector store and the
curated knowledge/ tree as retrieval tools for the agent:

    search_knowledge       Semantic search over curated concept articles,
                           with filters for memory type, confidence, zone,
                           and quarantine state.
    search_raw_daily       Semantic search over verbatim daily-log chunks
                           (the "drawer" layer — never summarized).
    get_article            Fetch one article's full content + frontmatter.
    list_contradictions    Return the current contradiction quarantine list.

The two servers are kept separate on purpose: code structure and
knowledge retrieval have very different durability and cache stories,
and a failure in the knowledge layer (e.g. a bad LLM compile) must
never break code-intel queries.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# Bootstrap: when Python runs this file directly (as Claude Code does via
# `python scripts/knowledge_mcp_server.py`), only the script's own
# directory is on sys.path. Pytest resolves imports via pythonpath =
# [".", "scripts"] in pyproject.toml, but direct execution doesn't.
# Manually add the memory-compiler root to sys.path so imports resolve
# regardless of how this script was invoked.
_HERE = Path(__file__).resolve().parent          # .../memory-compiler/scripts
_MEMORY_COMPILER_ROOT = _HERE.parent             # .../memory-compiler
for path in (str(_MEMORY_COMPILER_ROOT), str(_HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import bm25_store  # noqa: E402
import hybrid_search  # noqa: E402
import vector_store  # noqa: E402
from compile_truth import parse_frontmatter  # noqa: E402
from config import KNOWLEDGE_DIR, MEMORY_TYPES  # noqa: E402
from utils import load_contradictions  # noqa: E402

log = logging.getLogger("knowledge_mcp_server")

# Allowed values for the ``mode`` parameter on search_knowledge. ``hybrid``
# is the default because it matches the canonical behavior most agents
# expect; ``vector`` and ``bm25`` are exposed for A/B inspection and for
# falling back when one path is degraded.
SEARCH_MODES = {"hybrid", "vector", "bm25"}

# Token-efficient retrieval: search tools return snippets, not full bodies.
# Agents fetch full content via ``get_article`` / ``get_articles`` only for
# the slugs they actually want to read. Idea borrowed from claude-mem's
# search → get_observations split — saves ~10x context on multi-hit queries
# where most matches turn out to be irrelevant.
SNIPPET_CHARS = 220

# Metadata fields preserved on slim search hits. Everything else in the
# Chroma metadata blob (slug duplicates, large strings, internal bookkeeping)
# is dropped to keep search responses tight.
_SLIM_METADATA_KEYS = ("type", "confidence", "zone", "quarantined", "updated")


def _slim_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """Project a backend search hit down to the token-efficient shape.

    Backend searches (vector / BM25 / hybrid) return rich dicts with the full
    article or chunk body in ``text`` and the entire Chroma metadata blob.
    For MCP responses we only need enough to let the agent decide whether to
    fetch the full article — slug, a one-line title, a short snippet, score,
    and the load-bearing metadata flags. The full body is one ``get_article``
    call away.
    """
    text = hit.get("text") or ""
    md = hit.get("metadata") or {}
    snippet = text[:SNIPPET_CHARS]
    if len(text) > SNIPPET_CHARS:
        snippet = snippet.rstrip() + "…"
    slim_md = {k: md[k] for k in _SLIM_METADATA_KEYS if k in md}
    return {
        "id": hit.get("id"),
        "slug": hit.get("slug"),
        "title": md.get("title") or hit.get("slug"),
        "snippet": snippet,
        "distance": hit.get("distance"),
        "metadata": slim_md,
    }


# -----------------------------------------------------------------------------
# Tool implementations (pure functions — directly testable without MCP runtime)
# -----------------------------------------------------------------------------


def _search_knowledge_impl(
    query: str,
    limit: int = 5,
    type_filter: str | None = None,
    min_confidence: float | None = None,
    zone_filter: str | None = None,
    include_quarantined: bool = False,
    mode: str = "hybrid",
) -> list[dict[str, Any]]:
    """Search curated articles with metadata filters.

    Validates ``type_filter`` against ``config.MEMORY_TYPES``,
    ``zone_filter`` against ``{"observed", "synthesized"}``, and
    ``mode`` against ``SEARCH_MODES`` before dispatching. Unknown
    values raise ``ValueError`` so the agent sees a clear failure
    message instead of silently-empty results.
    """
    if type_filter is not None and type_filter not in MEMORY_TYPES:
        raise ValueError(
            f"type_filter must be one of {sorted(MEMORY_TYPES)}, got {type_filter!r}"
        )
    if zone_filter is not None and zone_filter not in {"observed", "synthesized"}:
        raise ValueError(
            f"zone_filter must be 'observed' or 'synthesized', got {zone_filter!r}"
        )
    if mode not in SEARCH_MODES:
        raise ValueError(
            f"mode must be one of {sorted(SEARCH_MODES)}, got {mode!r}"
        )

    backend = {
        "hybrid": hybrid_search.search_articles,
        "vector": vector_store.search_articles,
        "bm25": bm25_store.search_articles,
    }[mode]

    raw = backend(
        query=query,
        limit=limit,
        type_filter=type_filter,
        min_confidence=min_confidence,
        zone_filter=zone_filter,
        include_quarantined=include_quarantined,
    )
    return [_slim_hit(hit) for hit in raw]


def _search_raw_daily_impl(
    query: str,
    limit: int = 5,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    """Semantic search over verbatim daily-log chunks.

    Delegates to ``vector_store.search_daily``. ``date_from`` / ``date_to``
    are ISO YYYY-MM-DD strings — ``search_daily`` converts them to
    int-encoded ``date_int`` metadata for Chroma's $gte/$lte filters.
    """
    raw = vector_store.search_daily(
        query=query, limit=limit, date_from=date_from, date_to=date_to,
    )
    return [_slim_hit(hit) for hit in raw]


def _get_article_impl(slug: str) -> dict[str, Any]:
    """Fetch one article's full content + parsed frontmatter by slug.

    Resolves ``slug`` (e.g. ``concepts/stimulus-naming``) under the
    project's ``knowledge/`` directory and raises ``FileNotFoundError``
    if missing. Reads ``config.KNOWLEDGE_DIR`` at call time so tests
    can monkeypatch the path.
    """
    import config

    path = config.KNOWLEDGE_DIR / f"{slug}.md"
    if not path.exists():
        raise FileNotFoundError(f"No article at slug {slug!r}")
    content = path.read_text(encoding="utf-8")
    return {
        "slug": slug,
        "content": content,
        "frontmatter": parse_frontmatter(content),
    }


def _get_articles_impl(slugs: list[str]) -> list[dict[str, Any]]:
    """Batch-fetch multiple articles by slug in one MCP call.

    Companion to ``_get_article_impl`` — the two-tier retrieval pattern
    borrowed from claude-mem: ``search_knowledge`` returns slim hits
    (slug + snippet), the agent picks the interesting slugs, then one
    ``get_articles`` call fetches the full bodies for only those slugs.
    Missing slugs return ``{"slug": <slug>, "error": "not_found"}`` so a
    single bad slug in the batch does not abort the rest.
    """
    import config

    results: list[dict[str, Any]] = []
    for slug in slugs:
        path = config.KNOWLEDGE_DIR / f"{slug}.md"
        if not path.exists():
            results.append({"slug": slug, "error": "not_found"})
            continue
        content = path.read_text(encoding="utf-8")
        results.append({
            "slug": slug,
            "content": content,
            "frontmatter": parse_frontmatter(content),
        })
    return results


def _list_contradictions_impl() -> dict[str, Any]:
    """Return the current contradiction quarantine as a sorted list."""
    slugs = sorted(load_contradictions())
    return {"quarantined": slugs, "count": len(slugs)}


# -----------------------------------------------------------------------------
# FastMCP server bindings
# -----------------------------------------------------------------------------


def _make_server():
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("knowledge")

    @server.tool()
    def search_knowledge(
        query: str,
        limit: int = 5,
        type_filter: str | None = None,
        min_confidence: float | None = None,
        zone_filter: str | None = None,
        include_quarantined: bool = False,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """Search the curated knowledge base (hybrid BM25 + vector by default).

        Use when you need to find articles related to a natural-language
        question OR an identifier-shaped query (class name, function
        name, file path, error string). Prefer this over reading
        compiled-truth.md directly when the question is specific —
        filters narrow the result set far more efficiently than grep.

        Args:
            query: natural-language question or keywords
            limit: max results (default 5, usually enough)
            mode: 'hybrid' (default — fused BM25 + vector via RRF,
                best all-round), 'vector' (semantic only — better for
                abstract questions), or 'bm25' (literal token only —
                best for exact identifiers and file paths).
            type_filter: restrict to one of fact | event | discovery |
                preference | advice | decision
            min_confidence: only return articles with confidence >= this
                value (0.0-1.0). Use 0.7+ when you need firm answers,
                omit when exploring.
            zone_filter: 'observed' (direct source extractions — low
                hallucination risk) or 'synthesized' (compiler
                inferences — higher risk). Default: both zones.
            include_quarantined: include articles currently flagged as
                contradicted. Default False — quarantined articles are
                hidden until lint --resolve clears them.

        Returns:
            List of slim hits: ``{id, slug, title, snippet, distance, metadata}``.
            ``snippet`` is a ~220-char preview — the full body is NOT included.
            Lower ``distance`` means closer semantic match (0.0 = identical,
            1.0+ = unrelated). ``metadata`` is narrowed to
            ``{type, confidence, zone, quarantined, updated}``.

            This is deliberately token-efficient: scan the snippets to decide
            which slugs are worth reading, then call ``get_article(slug)``
            or ``get_articles([slug, ...])`` to pull the full body only for
            the hits you actually want.
        """
        return _search_knowledge_impl(
            query=query,
            limit=limit,
            type_filter=type_filter,
            min_confidence=min_confidence,
            zone_filter=zone_filter,
            include_quarantined=include_quarantined,
            mode=mode,
        )

    @server.tool()
    def search_raw_daily(
        query: str,
        limit: int = 5,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over verbatim daily-log chunks.

        Use this when ``search_knowledge`` returns weak results or when
        you need the raw unedited source material — e.g. to verify a
        compiled article against what was actually discussed in a
        session. Chunks are H2/H3-section-sized.

        Args:
            query: natural-language question or keywords
            limit: max chunks to return
            date_from: optional ISO YYYY-MM-DD lower bound (inclusive)
            date_to:   optional ISO YYYY-MM-DD upper bound (inclusive)

        Returns:
            List of slim hits: ``{id, slug, title, snippet, distance, metadata}``.
            ``snippet`` is a ~220-char preview of the chunk. Slim ``metadata``
            preserves the core flags; for full chunk text plus the richer
            ``section`` / ``date`` / ``source_file`` fields, the raw chunks
            are embedded in ``knowledge/daily/*.md`` and can be read directly.
        """
        return _search_raw_daily_impl(
            query=query, limit=limit, date_from=date_from, date_to=date_to,
        )

    @server.tool()
    def get_article(slug: str) -> dict[str, Any]:
        """Fetch one article's full content + frontmatter by slug.

        Use this after ``search_knowledge`` returns a promising match
        and you need the complete article text (Truth + Timeline) rather
        than the excerpt the vector store returns.

        Args:
            slug: article slug without the ``.md`` extension, e.g.
                ``concepts/stimulus-naming`` or ``connections/foo-and-bar``.

        Returns:
            ``{slug, content, frontmatter}``. ``content`` is the full
            markdown text; ``frontmatter`` is the parsed YAML header.

        Raises:
            FileNotFoundError: if the slug doesn't resolve to a real file.
        """
        return _get_article_impl(slug)

    @server.tool()
    def get_articles(slugs: list[str]) -> list[dict[str, Any]]:
        """Batch-fetch multiple articles by slug in one call.

        Paired with ``search_knowledge`` for the token-efficient two-step
        retrieval flow: search returns slim snippets, you pick the slugs
        worth reading in full, then ``get_articles`` pulls all of their
        bodies in one round trip. Cheaper than N sequential ``get_article``
        calls when you need 2+ articles.

        Args:
            slugs: list of article slugs without ``.md``, e.g.
                ``["concepts/stimulus-naming", "concepts/email-validation"]``.

        Returns:
            List aligned with ``slugs``. Each entry is either
            ``{slug, content, frontmatter}`` for a hit, or
            ``{slug, error: "not_found"}`` for a missing slug. A single
            bad slug does NOT abort the batch — the rest still return.
        """
        return _get_articles_impl(slugs)

    @server.tool()
    def list_contradictions() -> dict[str, Any]:
        """List all article slugs currently in the contradiction quarantine.

        Quarantined articles are excluded from ``compiled-truth.md`` and
        from ``search_knowledge`` results by default. Use this to see
        which articles need human review. Clear the quarantine with
        ``lint.py --resolve`` after review.

        Returns:
            ``{quarantined: [slug, ...], count: int}``
        """
        return _list_contradictions_impl()

    return server


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    server = _make_server()
    server.run()


if __name__ == "__main__":
    main()
