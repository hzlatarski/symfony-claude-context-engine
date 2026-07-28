"""Knowledge MCP Server — semantic retrieval over the compiled knowledge base.

A second MCP server (separate from the Symfony code-intel one at
``mcp_server.py``) that exposes the ChromaDB vector store and the
curated knowledge/ tree as retrieval tools for the agent:

    search_knowledge       Semantic search over curated concept articles,
                           with filters for memory type, confidence, zone,
                           and quarantine state.
    search_raw_daily       Semantic search over verbatim daily-log chunks
                           (the "drawer" layer — never summarized).
    get_raw_daily_chunk    Fetch one complete raw chunk by search-result ID.
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
import parent_watchdog  # noqa: E402
import vector_store  # noqa: E402
from compile_truth import parse_frontmatter  # noqa: E402
from config import KNOWLEDGE_DIR, MEMORY_TYPES  # noqa: E402
from utils import load_contradictions, resolve_article_path  # noqa: E402

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
_SLIM_METADATA_KEYS = (
    "type",
    "confidence",
    "zone",
    "quarantined",
    "updated",
    "source_file",
    "section",
    "date",
    "session_id",
)

# When include_linked is set, we also surface the project tag so the agent
# can tell at a glance which project a hit came from.
_LINKED_EXTRA_KEYS = ("project",)


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
    out: dict[str, Any] = {
        "id": hit.get("id"),
        "slug": hit.get("slug"),
        "title": md.get("title") or hit.get("slug"),
        "snippet": snippet,
        "distance": hit.get("distance"),
        "metadata": slim_md,
    }
    # Surface linked-project tagging when present so the agent can tell
    # which knowledge base a cross-project hit originated from.
    for key in _LINKED_EXTRA_KEYS:
        if key in hit:
            out[key] = hit[key]
    return out


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
    include_linked: bool = False,
) -> list[dict[str, Any]]:
    """Search curated articles with metadata filters.

    Validates ``type_filter`` against ``config.MEMORY_TYPES``,
    ``zone_filter`` against ``{"observed", "synthesized"}``, and
    ``mode`` against ``SEARCH_MODES`` before dispatching. Unknown
    values raise ``ValueError`` so the agent sees a clear failure
    message instead of silently-empty results.

    ``include_linked`` activates the cross-project fan-out path —
    incompatible with ``mode="bm25"`` (cross-process BM25 is not
    supported), so we raise ValueError if both are requested.
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
    if include_linked and mode == "bm25":
        raise ValueError(
            "include_linked is not supported with mode='bm25' — cross-project "
            "BM25 indexes are not exposed. Use mode='hybrid' or 'vector'."
        )

    if include_linked:
        import linked_search

        raw = linked_search.search_articles_linked(
            query=query,
            limit=limit,
            type_filter=type_filter,
            min_confidence=min_confidence,
            zone_filter=zone_filter,
            include_quarantined=include_quarantined,
        )
        return [_slim_hit(hit) for hit in raw]

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
    """Literal-first plus semantic search over verbatim transcript chunks."""
    exact = vector_store.search_daily_exact(
        query=query, limit=limit, date_from=date_from, date_to=date_to,
    )
    semantic = vector_store.search_daily(
        query=query, limit=limit, date_from=date_from, date_to=date_to,
    )

    combined: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in exact + semantic:
        hit_id = hit.get("id")
        if not hit_id or hit_id in seen:
            continue
        seen.add(hit_id)
        if hit in exact:
            text = hit.get("text") or ""
            position = text.find(query)
            if position >= 0:
                start = max(0, position - 80)
                hit = {**hit, "text": text[start : position + len(query) + 80]}
        combined.append(hit)
        if len(combined) >= limit:
            break
    return [_slim_hit(hit) for hit in combined]


def _get_raw_daily_chunk_impl(chunk_id: str) -> dict[str, Any]:
    """Fetch one complete verbatim chunk by its search-result ID."""
    return vector_store.get_daily_chunk(chunk_id)


def _get_article_impl(slug: str) -> dict[str, Any]:
    """Fetch one article's full content + parsed frontmatter by slug.

    Resolves ``slug`` (e.g. ``concepts/stimulus-naming``) under the
    project's ``knowledge/`` directory and raises ``FileNotFoundError``
    if missing. Reads ``config.KNOWLEDGE_DIR`` at call time so tests
    can monkeypatch the path.
    """
    import config

    normalized = slug.removesuffix(".md")
    path = resolve_article_path(
        normalized, config.KNOWLEDGE_DIR, must_exist=True,
    )
    content = path.read_text(encoding="utf-8")
    return {
        "slug": normalized,
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
        try:
            path = resolve_article_path(
                slug, config.KNOWLEDGE_DIR, must_exist=True,
            )
        except ValueError:
            results.append({"slug": slug, "error": "invalid_slug"})
            continue
        except FileNotFoundError:
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


def _record_retrieval_outcome_impl(
    slug: str,
    outcome: str,
    note: str = "",
) -> dict[str, Any]:
    """Persist a retrieval-outcome rating for one article slug.

    Validates ``outcome`` against ``retrieval_feedback.OUTCOMES`` and
    verifies the slug resolves to a real article so a typo'd slug can't
    silently accrue feedback that never influences ranking.
    """
    import config
    import retrieval_feedback

    normalized = slug.removesuffix(".md")
    try:
        resolve_article_path(normalized, config.KNOWLEDGE_DIR, must_exist=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No article at slug {normalized!r} — refusing to record feedback "
            f"for a slug that doesn't resolve to a file."
        ) from exc

    if outcome not in retrieval_feedback.OUTCOMES:
        raise ValueError(
            f"outcome must be one of {sorted(retrieval_feedback.OUTCOMES)}, got {outcome!r}"
        )

    rec = retrieval_feedback.record(normalized, outcome, note=note)
    return {
        "ok": True,
        "slug": normalized,
        "outcome": outcome,
        "totals": {
            "useful": rec.get("useful", 0),
            "dead_end": rec.get("dead_end", 0),
            "corrected": rec.get("corrected", 0),
        },
        "score": retrieval_feedback.score_for(normalized),
    }


_CODEBASE_FILE_TYPES = {"php", "js", "twig", "yaml"}


def _search_codebase_impl(
    query: str,
    limit: int = 5,
    file_type: str | None = None,
) -> list[dict]:
    """Search indexed source code chunks by semantic similarity.

    Validates file_type before dispatching to codebase_store so the
    agent gets a clear ValueError instead of silently-empty results
    when passing an unsupported extension.
    """
    if file_type is not None and file_type not in _CODEBASE_FILE_TYPES:
        raise ValueError(
            f"file_type must be one of {sorted(_CODEBASE_FILE_TYPES)}, got {file_type!r}"
        )

    import codebase_store

    raw = codebase_store.search_codebase(query=query, limit=limit, file_type=file_type)
    out = []
    for hit in raw:
        text = hit.get("text") or ""
        meta = hit.get("metadata") or {}
        preview = text[:300]
        if len(text) > 300:
            preview = preview.rstrip() + "…"
        rel = hit.get("rel_path") or meta.get("rel_path", "")
        start = meta.get("start_line")
        end = meta.get("end_line")
        path = f"{rel}:{start}-{end}" if start and end else rel
        entry: dict = {"path": path, "preview": preview}
        symbols = meta.get("symbols") or ""
        if symbols:
            entry["symbols"] = symbols
        # source_description tells the agent why this group exists and when
        # to consult it. Surfaced on every hit so the model has the cue
        # right next to the snippet — no separate lookup needed.
        sd = meta.get("source_description") or ""
        if sd:
            entry["source_description"] = sd
        out.append(entry)
    return out


def _list_sources_impl() -> list[dict]:
    """Return the full source-group catalog with descriptions and chunk counts."""
    import index_codebase
    return index_codebase.list_source_groups()


def _ingest_status_impl() -> dict[str, Any]:
    """Return the live ingest progress snapshot, or a sentinel if idle."""
    import ingest_state

    snap = ingest_state.read_status()
    if snap is None:
        return {"phase": "idle", "message": "No ingest run has started yet."}
    return snap


def _kb_health_impl() -> dict[str, Any]:
    """One-shot health snapshot of the entire knowledge base.

    Aggregates: vector store sizes, article counts by memory type,
    quarantine count, broken source-anchor count, last ingest timestamp,
    and the most recent compiled-truth mtime. Designed as a single MCP
    call so the agent (or a CI gate) can answer "is the KB healthy?"
    without running six separate scripts.
    """
    import codebase_store
    import vector_store
    from compile_truth import COMPILED_TRUTH_FILE
    from utils import load_state

    # Collection sizes — one call each, fast.
    try:
        vector_stats = vector_store.stats()
    except Exception as exc:
        vector_stats = {"error": str(exc)}
    try:
        codebase_stats = codebase_store.stats()
    except Exception as exc:
        codebase_stats = {"error": str(exc)}

    # Article counts by memory type. Walk the markdown filesystem rather
    # than hitting Chroma — frontmatter is the source of truth for `type`.
    type_counts: dict[str, int] = {t: 0 for t in MEMORY_TYPES}
    type_counts["unknown"] = 0
    total_articles = 0
    if KNOWLEDGE_DIR.exists():
        for md_path in KNOWLEDGE_DIR.glob("concepts/*.md"):
            total_articles += 1
            try:
                fm = parse_frontmatter(md_path.read_text(encoding="utf-8"))
            except Exception:
                fm = {}
            mtype = (fm or {}).get("type")
            if mtype in MEMORY_TYPES:
                type_counts[mtype] += 1
            else:
                type_counts["unknown"] += 1

    # Source-anchor health — reuse lint.py's checker so the rule lives in one place.
    broken_anchor_count = 0
    missing_anchor_count = 0
    try:
        import lint

        for issue in lint.check_source_anchors():
            check = issue.get("check")
            if check == "broken_source_anchor":
                broken_anchor_count += 1
            elif check == "missing_source_anchor":
                missing_anchor_count += 1
    except Exception as exc:
        broken_anchor_count = -1  # sentinel: linter failed
        missing_anchor_count = -1
        log.warning("lint.check_source_anchors failed in kb_health: %s", exc)

    # Quarantine count.
    contradictions = sorted(load_contradictions())

    # Retrieval-outcome feedback rollup — surfaces the most-contested
    # articles (repeatedly dead_end / corrected) as a review signal
    # alongside the automated contradiction quarantine.
    try:
        import retrieval_feedback

        agg = retrieval_feedback.aggregate()
        rated = len(agg)
        contested = [
            {"slug": r["slug"], "score": round(r["score"], 3),
             "corrected": r["corrected"], "dead_end": r["dead_end"], "useful": r["useful"]}
            for r in agg if r["score"] < 0.4
        ][:10]
        feedback_stats = {
            "rated_articles": rated,
            "contested_count": sum(1 for r in agg if r["score"] < 0.4),
            "most_contested": contested,
        }
    except Exception as exc:  # noqa: BLE001 — kb_health must never raise
        feedback_stats = {"error": str(exc)}

    # Last ingest timestamp — most recent `ingested_at` in state.json.
    last_ingest_at: str | None = None
    state = load_state()
    sources = state.get("ingested_sources") or {}
    if sources:
        ingested_times = [
            v.get("ingested_at") for v in sources.values() if v.get("ingested_at")
        ]
        if ingested_times:
            last_ingest_at = max(ingested_times)

    # Compiled-truth freshness.
    compiled_truth_mtime: float | None = None
    if COMPILED_TRUTH_FILE.exists():
        compiled_truth_mtime = COMPILED_TRUTH_FILE.stat().st_mtime

    # Live ingest status (idle / running / etc.) — one extra read, useful
    # alongside the rest because "is anything in flight?" is a common gate.
    import ingest_state

    live_status = ingest_state.read_status() or {"phase": "idle"}

    # Unified-graph stats — guarded so a build failure doesn't break the diagnostic.
    try:
        from scripts import unified_graph as _ug
        from scripts.parsers import PROJECT_ROOT
        _graph = _ug.build_for_project(PROJECT_ROOT, KNOWLEDGE_DIR)
        _article_nodes = [n for n in _graph["nodes"] if n.startswith("article:")]
        _referenced = {e["from"] for e in _graph["edges"]} | {e["to"] for e in _graph["edges"]}
        graph_stats: dict[str, Any] = {
            "articles": len(_article_nodes),
            "files": sum(1 for n in _graph["nodes"] if n.startswith("file:")),
            "edges": len(_graph["edges"]),
            "orphan_articles": sum(1 for n in _article_nodes if n not in _referenced),
        }
    except Exception as exc:  # noqa: BLE001 — kb_health must never raise
        graph_stats = {"error": str(exc)}

    return {
        "vector_store": vector_stats,
        "codebase_store": codebase_stats,
        "articles": {
            "total": total_articles,
            "by_type": type_counts,
        },
        "anchors": {
            "broken": broken_anchor_count,
            "articles_missing_anchors": missing_anchor_count,
        },
        "quarantine": {
            "count": len(contradictions),
            "slugs": contradictions,
        },
        "feedback": feedback_stats,
        "freshness": {
            "last_ingest_at": last_ingest_at,
            "compiled_truth_mtime": compiled_truth_mtime,
        },
        "ingest": {
            "phase": live_status.get("phase"),
            "current_file": live_status.get("current_file"),
            "processed": live_status.get("processed"),
            "total": live_status.get("total"),
        },
        "graph": graph_stats,
    }


def _ingest_stop_impl() -> dict[str, Any]:
    """Signal the running ingest to halt at the next file boundary."""
    import ingest_state

    if not ingest_state.read_status() or ingest_state.read_status().get("phase") not in {"starting", "running"}:
        return {
            "ok": False,
            "message": "No active ingest run — stop signal not raised.",
        }
    newly = ingest_state.request_stop()
    return {
        "ok": True,
        "newly_requested": newly,
        "message": (
            "Stop signal raised — the ingest will halt at the next file boundary."
            if newly
            else "Stop signal already pending; ingest will halt at the next boundary."
        ),
    }


# -----------------------------------------------------------------------------
# FastMCP server bindings
# -----------------------------------------------------------------------------


def _make_server():
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("knowledge-compiler")

    @server.tool()
    def search_knowledge(
        query: str,
        limit: int = 5,
        type_filter: str | None = None,
        min_confidence: float | None = None,
        zone_filter: str | None = None,
        include_quarantined: bool = False,
        mode: str = "hybrid",
        include_linked: bool = False,
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
            include_linked: also search the knowledge bases of projects
                listed in ``MEMORY_COMPILER_LINKED_PROJECTS``. Useful
                for "have we hit this gotcha before in another project?"
                queries. Hits get a ``project`` tag (``"<local>"`` or
                the linked project's directory name). Incompatible with
                ``mode='bm25'``.

        Returns:
            List of slim hits: ``{id, slug, title, snippet, distance, metadata}``.
            ``snippet`` is a ~220-char preview — the full body is NOT included.
            Lower ``distance`` means closer semantic match (0.0 = identical,
            1.0+ = unrelated). ``metadata`` is narrowed to
            ``{type, confidence, zone, quarantined, updated}``. When
            ``include_linked=True`` each hit also carries ``project``.

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
            include_linked=include_linked,
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
            includes ``section`` / ``date`` / ``source_file`` when present.
            Call ``get_raw_daily_chunk(id)`` for the complete, unclipped text.
        """
        return _search_raw_daily_impl(
            query=query, limit=limit, date_from=date_from, date_to=date_to,
        )

    @server.tool()
    def get_raw_daily_chunk(chunk_id: str) -> dict[str, Any]:
        """Fetch one complete raw daily/transcript chunk by ID.

        Use the ``id`` returned by ``search_raw_daily`` when the short snippet
        is insufficient or contains a clipped URL, identifier, or tool result.

        Returns:
            ``{id, text, metadata}``, with the full verbatim chunk text.

        Raises:
            FileNotFoundError: if the chunk ID no longer exists.
        """
        return _get_raw_daily_chunk_impl(chunk_id)

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
    def record_retrieval_outcome(
        slug: str,
        outcome: str,
        note: str = "",
    ) -> dict[str, Any]:
        """Record whether an article you retrieved actually helped.

        Call this AFTER you have used an article from ``search_knowledge`` /
        ``get_article`` and know whether it was any good. The outcome is
        recency-weighted and feeds directly into ``compiled-truth.md``
        priority scoring (via ``retrieval_feedback``) — so rating articles
        makes the knowledge base rank genuinely-useful ones higher over time
        and sink misleading ones. This is the one signal time-based
        confidence decay can't capture: whether the content was *correct and
        useful*, not merely *recent*.

        When to record which outcome:
            - ``useful``    — the article answered the question / was correct
              and load-bearing for what you did.
            - ``dead_end``  — retrieved but unhelpful (off-topic, too vague,
              didn't apply). Not wrong, just not useful here.
            - ``corrected`` — the article was WRONG or stale and you had to
              correct course. The strongest negative signal — repeatedly
              ``corrected`` articles surface in ``reflect.py`` / ``kb_health``
              for human review.

        Args:
            slug: article slug without ``.md`` (e.g. ``concepts/foo``). Must
                resolve to a real article or this raises FileNotFoundError.
            outcome: ``useful`` | ``dead_end`` | ``corrected``.
            note: optional short context ("stale — prod moved to X"), stored
                with the event and shown in reflect/kb_health.

        Returns:
            ``{ok, slug, outcome, totals, score}`` where ``score`` is the
            article's new recency-weighted feedback score in ``[0, 1]``
            (0.5 = neutral).
        """
        return _record_retrieval_outcome_impl(slug=slug, outcome=outcome, note=note)

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

    @server.tool()
    def search_codebase(
        query: str,
        limit: int = 5,
        file_type: str | None = None,
    ) -> list[dict]:
        """Search indexed source code files by semantic similarity.

        Use when you need to find code that implements a concept — e.g.
        "how does authentication work?" returns actual PHP/Twig/JS file
        paths with code excerpts, not prose articles about them.
        Complements search_knowledge (which searches compiled articles).

        Run `index_codebase.py --all` first if the index is empty.
        The index is auto-updated after every Write/Edit via the
        PostToolUse hook — no manual refresh needed in normal operation.

        Indexed: src/**/*.php, assets/controllers/**/*.js,
        templates/**/*.twig, config/**/*.yaml

        Args:
            query: natural-language question or code identifier
            limit: max results (default 5)
            file_type: restrict to 'php', 'js', 'twig', or 'yaml'

        Returns:
            List of slim hits: ``{path, preview}`` plus ``symbols`` when
            non-empty. ``path`` is ``rel_path:start-end`` (git/IDE-style
            line range, relative to the Symfony project root) — pass the
            file portion to the ``Read`` tool with ``offset=start`` and
            ``limit=end-start`` to get the full chunk canonically from
            disk (the index may lag edits; Read cannot). ``preview`` is
            a ~300-char excerpt to decide if a hit is worth expanding.
            ``symbols`` (PHP only) lists class/method names in the chunk
            and anchors identifier-shaped queries.
        """
        return _search_codebase_impl(query=query, limit=limit, file_type=file_type)

    @server.tool()
    def kb_health() -> dict[str, Any]:
        """One-shot health snapshot of the knowledge base.

        Use as a single trust check before answering critical questions
        from the KB, or wired into CI as a regression gate. Combines six
        signals that previously required six separate scripts.

        Returns:
            ``{vector_store, codebase_store, articles, anchors, quarantine, freshness, ingest}``.

            - ``vector_store`` / ``codebase_store``: Chroma collection sizes.
            - ``articles.by_type``: count per memory type plus an
              ``unknown`` bucket for articles missing a valid ``type:``.
            - ``anchors.broken``: number of ``[src:...]`` anchors pointing
              to files that don't exist (``-1`` means lint failed).
              ``anchors.articles_missing_anchors``: articles with no
              anchors at all (suggestion-level, not blocking).
            - ``quarantine.slugs``: articles flagged by contradiction lint.
            - ``freshness.last_ingest_at``: ISO timestamp of the most
              recent ``ingest.py`` file write. ``compiled_truth_mtime``:
              POSIX mtime of ``compiled-truth.md``.
            - ``ingest``: live status from the running ingest, if any.

            A healthy KB has ``anchors.broken == 0``, ``quarantine.count == 0``,
            and a recent ``last_ingest_at``.
        """
        return _kb_health_impl()

    @server.tool()
    def ingest_status() -> dict[str, Any]:
        """Live progress snapshot for the most recent / active ingest run.

        Use to check whether a long-running ``ingest.py`` is still alive
        and how far along it is — analogous to SocratiCode's
        ``codebase_status``. Reads ``knowledge/.ingest-status.json``;
        the file is rewritten atomically at every file boundary.

        Returns:
            ``{phase, current_file, processed, total, total_cost, started_at, updated_at}``.

            ``phase`` is one of ``idle`` (nothing has run), ``starting``,
            ``running``, ``finished``, ``stopped``, or ``error``. When
            ``phase == "idle"`` only the ``message`` field is populated.

            Compare ``updated_at`` against the current time — a snapshot
            older than a few minutes during ``running`` likely means the
            process crashed silently.
        """
        return _ingest_status_impl()

    @server.tool()
    def ingest_stop() -> dict[str, Any]:
        """Request a graceful halt of the running ingest at the next file boundary.

        Cooperative: ingest.py polls for the stop file between files, so
        the current Sonnet call (if any) finishes and is checkpointed
        before the run exits. Already-completed files are NOT rolled
        back — re-running ``ingest.py`` resumes from the next unprocessed
        file via the existing hash-based skip logic.

        Returns:
            ``{ok, newly_requested, message}``. ``ok`` is False if no
            active run is detected — there is nothing to stop in that
            case. ``newly_requested`` distinguishes the first stop from
            an idempotent retry.
        """
        return _ingest_stop_impl()

    @server.tool()
    def list_sources() -> list[dict]:
        """List the indexed source-code groups with descriptions and chunk counts.

        Use this **before** ``search_codebase`` when you don't yet know
        which file_type best fits your query — the descriptions explain
        *when* to consult each group ("check this before writing
        migrations to match naming conventions", etc.). After picking a
        group, narrow ``search_codebase`` with ``file_type=<type>``.

        Also useful as a freshness check: ``chunk_count: 0`` for a group
        you expected to be populated means the index is stale or the
        glob is misconfigured — re-run ``python scripts/index_codebase.py``.

        Returns:
            List of ``{file_type, patterns, description, chunk_count}``
            entries, one per source group. ``patterns`` are the globs
            (relative to the Symfony project root) that the indexer walks.
        """
        return _list_sources_impl()

    return server


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parent_watchdog.start()
    server = _make_server()
    server.run()


if __name__ == "__main__":
    main()
