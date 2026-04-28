"""FastAPI web viewer for the memory-compiler knowledge base.

A read-only dashboard over the curated articles, daily logs, tool drawer,
ChromaDB stats, and flush cost history. Server-rendered with Jinja, no
build step, no React, no static asset pipeline — just ``uv run python
scripts/viewer.py`` and browse ``http://127.0.0.1:37778``.

Design goals:

1. **Read-only.** This is an inspector, not an editor. The viewer never writes
   to ``knowledge/`` or to Chroma. Mistakes in the UI can't corrupt the store.
2. **Zero-dep frontend.** Vanilla Jinja templates + inline CSS. No JS build,
   no static mount, no CDN. Works offline, boots in ~1s.
3. **Domain-aware.** The knowledge base has memory types, confidence scores,
   zones, and quarantine state — the UI surfaces those prominently instead
   of rendering everything as generic cards. Type is tinted by color so the
   eye finds facts vs decisions vs events at a glance.
4. **Bound to localhost by default.** No auth is intentional — the viewer is
   a local dev tool. Binding to ``127.0.0.1`` makes it inaccessible to other
   machines on the LAN.

The claude-mem comparison report earlier this session flagged the absence of
a viewer as one of the few things claude-mem does better than us. This file
closes that gap, adapted to our server-rendered stack and our curated-wiki
format (claude-mem's viewer is React + SSE over a live session stream; we
don't need either because our store is markdown-on-disk).
"""
from __future__ import annotations

import json
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Bootstrap — same sys.path dance as the MCP servers so the viewer is
# runnable via ``python scripts/viewer.py`` without needing pytest's
# pythonpath to be on the loader's radar.
import sys
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for path in (str(_ROOT), str(_HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile  # noqa: E402
from pydantic import BaseModel as _BaseModel  # noqa: E402
from fastapi.responses import HTMLResponse, RedirectResponse  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402
from markdown_it import MarkdownIt  # noqa: E402

import yaml  # noqa: E402

import logging  # noqa: E402

import config  # noqa: E402
from compile_truth import strip_frontmatter  # noqa: E402
from utils import load_contradictions  # noqa: E402


logger = logging.getLogger(__name__)


# Request model for the /api/whisper/re-enhance endpoint.
# Defined at module level so Pydantic v2 can fully resolve the type at import
# time; local class definitions inside create_app() cause forward-reference
# errors when FastAPI registers the route.
class _ReEnhanceRequest(_BaseModel):
    transcript: str
    mode: str = "rewrite"
    scope_override: list[str] | None = None


# -----------------------------------------------------------------------------
# Frontmatter parser (viewer-local)
# -----------------------------------------------------------------------------
#
# The project's ``compile_truth.parse_frontmatter`` is a hand-rolled line-wise
# parser that intentionally only extracts title/type/confidence/updated/created
# /pinned and computes a flat ``source_count``. That's the right fit for the
# compiler's scoring loop but wrong for the viewer, which needs the full
# ``sources:`` list, ``quarantined``, ``zone``, and any other custom fields
# an article carries. We use PyYAML here because it's already a project dep
# (via sources.yaml loading) and because a real YAML parser is the only
# sensible way to preserve list-valued fields without reinventing the wheel.


def parse_frontmatter(content: str) -> dict[str, Any]:
    """Extract YAML frontmatter from an article as a full dict.

    Returns ``{}`` when there is no frontmatter, the delimiters are malformed,
    or YAML parsing fails — the viewer must never crash on a weird article,
    the rest of the page is still useful.
    """
    if not content.startswith("---"):
        return {}
    end = content.find("---", 3)
    if end == -1:
        return {}
    fm_text = content[3:end]
    try:
        parsed = yaml.safe_load(fm_text)
    except yaml.YAMLError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


TEMPLATES_DIR = _HERE / "viewer_templates"

# Markdown renderer used for article bodies and daily logs. ``commonmark``
# is enough — no tables, no footnotes, no strikethrough — which keeps the
# rendered output close to what Obsidian shows.
_md = MarkdownIt("commonmark")

# Wikilinks ``[[foo]]`` aren't CommonMark; pre-process them into ordinary
# markdown links pointing at our own routes before handing to the renderer.
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+?)(?:\|([^\]]+))?\]\]")


def _rewrite_wikilinks(text: str) -> str:
    """Turn ``[[concepts/foo]]`` into ``[concepts/foo](/articles/concepts/foo)``.

    Pipe alias syntax is supported: ``[[concepts/foo|display name]]`` →
    ``[display name](/articles/concepts/foo)``. Targets get URL-encoded via
    their own path segment so a slug with odd characters doesn't break.
    """
    def _sub(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        label = (match.group(2) or target).strip()
        return f"[{label}](/articles/{target})"
    return _WIKILINK_RE.sub(_sub, text)


def _render_markdown(text: str) -> str:
    """Pipeline: strip frontmatter → rewrite wikilinks → render commonmark."""
    body = strip_frontmatter(text)
    body = _rewrite_wikilinks(body)
    return _md.render(body)


# -----------------------------------------------------------------------------
# Data loaders — pure functions, directly testable
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ArticleSummary:
    slug: str            # e.g. "concepts/stimulus-naming"
    title: str
    type: str            # fact | event | discovery | preference | advice | decision | ""
    confidence: float    # 0.0 - 1.0; 0.0 when absent
    updated: str         # ISO date string; "" when absent
    quarantined: bool
    source_count: int    # number of ``sources:`` entries in frontmatter
    excerpt: str         # first ~200 chars of the body


def _iter_article_paths(knowledge_dir: Path) -> Iterable[Path]:
    """Walk all ``.md`` files under the curated subtrees.

    Only ``concepts/`` and ``connections/`` are treated as article subtrees —
    ``daily/``, ``research/``, and top-level files like ``index.md`` are
    deliberately excluded so the viewer's article list stays focused on
    compiled knowledge, not raw inputs.
    """
    for sub in ("concepts", "connections"):
        root = knowledge_dir / sub
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.md")):
            yield path


def _summarize_article(path: Path, knowledge_dir: Path) -> ArticleSummary:
    content = path.read_text(encoding="utf-8", errors="replace")
    fm = parse_frontmatter(content) or {}
    body = strip_frontmatter(content).strip()
    # Relative slug = path from knowledge/ minus the .md extension.
    rel = path.relative_to(knowledge_dir).as_posix()
    slug = rel[:-3] if rel.endswith(".md") else rel
    excerpt = body[:200].replace("\n", " ").strip()
    if len(body) > 200:
        excerpt = excerpt.rstrip() + "…"
    try:
        confidence = float(fm.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    sources = fm.get("sources") or []
    return ArticleSummary(
        slug=slug,
        title=str(fm.get("title") or slug),
        type=str(fm.get("type") or ""),
        confidence=confidence,
        updated=str(fm.get("updated") or ""),
        quarantined=bool(fm.get("quarantined") or False),
        source_count=len(sources) if isinstance(sources, list) else 0,
        excerpt=excerpt,
    )


def load_all_articles(knowledge_dir: Path) -> list[ArticleSummary]:
    """Return a summary row for every curated article under ``knowledge_dir``."""
    return [_summarize_article(p, knowledge_dir) for p in _iter_article_paths(knowledge_dir)]


def filter_articles(
    articles: list[ArticleSummary],
    *,
    type_filter: str | None = None,
    min_confidence: float | None = None,
    quarantine: str = "hide",  # "hide" | "only" | "all"
    search: str | None = None,
) -> list[ArticleSummary]:
    """Apply the article-list filters in the order the UI presents them.

    ``search`` does substring match (case-insensitive) against title, slug,
    and excerpt — intentionally trivial because the MCP layer owns real
    semantic search. The viewer's search is just a "find by eye" aid.
    """
    results = articles
    if type_filter:
        results = [a for a in results if a.type == type_filter]
    if min_confidence is not None:
        results = [a for a in results if a.confidence >= min_confidence]
    if quarantine == "hide":
        results = [a for a in results if not a.quarantined]
    elif quarantine == "only":
        results = [a for a in results if a.quarantined]
    if search:
        needle = search.lower()
        results = [
            a for a in results
            if needle in a.title.lower()
            or needle in a.slug.lower()
            or needle in a.excerpt.lower()
        ]
    return results


def load_daily_log_index(daily_dir: Path) -> list[dict[str, Any]]:
    """Return ``[{date, size_bytes, has_drawer}]`` for every daily .md file.

    ``has_drawer`` is True when the PostToolUse JSONL drawer exists for that
    date — the viewer links the two together so a user can jump from a
    compiled session summary to the raw tool events it was built from.
    """
    if not daily_dir.exists():
        return []
    rows = []
    for md_path in sorted(daily_dir.glob("*.md"), reverse=True):
        date = md_path.stem
        drawer = daily_dir / f"{date}.tools.jsonl"
        rows.append({
            "date": date,
            "size_bytes": md_path.stat().st_size,
            "has_drawer": drawer.exists(),
        })
    return rows


def load_tool_drawer(daily_dir: Path, date: str) -> list[dict[str, Any]]:
    """Parse one day's ``YYYY-MM-DD.tools.jsonl`` drawer into a list of events.

    Defensive like ``flush.load_tool_events``: missing file → empty list,
    malformed lines are skipped. Returns events in the order they were
    written (chronological within a session).
    """
    path = daily_dir / f"{date}.tools.jsonl"
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def summarize_tool_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-tool counts and ok/err totals for a drawer day."""
    counts: dict[str, int] = {}
    errors = 0
    sessions: set[str] = set()
    for event in events:
        tool = event.get("tool") or "unknown"
        counts[tool] = counts.get(tool, 0) + 1
        if not event.get("ok", True):
            errors += 1
        sid = event.get("session_id")
        if sid:
            sessions.add(str(sid))
    return {
        "total": len(events),
        "errors": errors,
        "sessions": len(sessions),
        "by_tool": sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])),
    }


def load_flush_state() -> dict[str, Any]:
    """Read the flush state file written by flush.py (costs, dedup, history)."""
    path = config.SCRIPTS_DIR / "last-flush.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_ingest_state() -> dict[str, Any]:
    """Read the compile/ingest state file (per-file hashes, timestamps)."""
    path = config.STATE_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _today_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")


def _today_flush_total(state: dict[str, Any]) -> float:
    """Sum today's flush costs from ``state["flush_costs"]`` (aligned with flush.py)."""
    today_start = datetime.now(timezone.utc).astimezone().replace(
        hour=0, minute=0, second=0, microsecond=0
    ).timestamp()
    return sum(
        entry.get("cost_usd", 0.0)
        for entry in state.get("flush_costs", [])
        if entry.get("timestamp", 0) >= today_start
    )


def _chroma_stats() -> dict[str, int]:
    """Ask vector_store + codebase_store how many items are indexed.

    Swallowed in a broad try/except because Chroma initialization can fail
    if the ONNX model isn't cached yet — the viewer should still boot and
    render the rest of the dashboard with zeros rather than 500.
    """
    try:
        import vector_store
        import codebase_store
        s = vector_store.stats()
        s.update(codebase_store.stats())
        return s
    except Exception:
        return {"articles": 0, "daily_chunks": 0, "codebase_chunks": 0}


# -----------------------------------------------------------------------------
# FastAPI app + routes
# -----------------------------------------------------------------------------


def create_app(knowledge_dir: Path | None = None) -> FastAPI:
    """Factory returning a configured FastAPI app.

    ``knowledge_dir`` defaults to ``config.KNOWLEDGE_DIR`` and is injectable
    for tests — TestClient can point at a temp dir full of fixture articles
    without touching the real knowledge store.
    """
    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        try:
            from whisper.transcribe import preload_model
            preload_model()
        except Exception as exc:  # noqa: BLE001
            logger.warning("whisper model preload failed at startup: %s", exc)
        yield

    app = FastAPI(title="memory-compiler viewer", docs_url=None, redoc_url=None, lifespan=_lifespan)
    kb = knowledge_dir or config.KNOWLEDGE_DIR

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    # Memory-type palette — matches the type-tinted card borders borrowed
    # from claude-mem's viewer-template.html. Exposed to all templates via
    # the context processor below so sidebar nav and cards pick the same color.
    type_colors = {
        "fact": "#60a5fa",
        "event": "#f59e0b",
        "discovery": "#a78bfa",
        "preference": "#f472b6",
        "advice": "#34d399",
        "decision": "#ef4444",
        "tension": "#fb923c",       # orange — unresolved, demands attention
        "hypothesis": "#22d3ee",    # cyan — provisional, awaiting evidence
    }
    templates.env.globals["type_colors"] = type_colors
    templates.env.globals["memory_types"] = sorted(config.MEMORY_TYPES)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        articles = load_all_articles(kb)
        daily = load_daily_log_index(config.DAILY_DIR)
        flush_state = load_flush_state()
        ingest_state = load_ingest_state()
        contradictions = sorted(load_contradictions())
        chroma = _chroma_stats()
        today = _today_iso()
        today_events = load_tool_drawer(config.DAILY_DIR, today)
        # Type histogram — feeds the "memory composition" card on the home page.
        type_counts: dict[str, int] = {}
        for a in articles:
            key = a.type or "(none)"
            type_counts[key] = type_counts.get(key, 0) + 1

        # Recent articles — sorted by updated date descending. Articles missing
        # an ``updated`` field sink to the bottom; they're usually stubs anyway.
        recent = sorted(
            articles, key=lambda a: (a.updated or ""), reverse=True,
        )[:12]

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "total_articles": len(articles),
                "quarantined_count": len(contradictions),
                "daily_count": len(daily),
                "type_counts": sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0])),
                "recent": recent,
                "today": today,
                "today_tool_count": len(today_events),
                "today_flush_cost": _today_flush_total(flush_state),
                "total_flushes_tracked": len(flush_state.get("flush_costs", [])),
                "chroma_articles": chroma.get("articles", 0),
                "chroma_daily_chunks": chroma.get("daily_chunks", 0),
                "chroma_codebase_chunks": chroma.get("codebase_chunks", 0),
                "ingest_files_count": len(ingest_state.get("ingested", {})),
                "contradictions": contradictions[:10],
                "total_contradictions": len(contradictions),
            },
        )

    @app.get("/articles", response_class=HTMLResponse)
    def articles_list(
        request: Request,
        type: str | None = Query(None),
        min_confidence: float | None = Query(None, ge=0.0, le=1.0),
        quarantine: str = Query("hide", pattern="^(hide|only|all)$"),
        q: str | None = Query(None),
    ) -> HTMLResponse:
        all_articles = load_all_articles(kb)
        type_param = type if type in config.MEMORY_TYPES else None
        filtered = filter_articles(
            all_articles,
            type_filter=type_param,
            min_confidence=min_confidence,
            quarantine=quarantine,
            search=q,
        )
        # Sort filtered results by confidence descending so firm knowledge
        # surfaces above tentative plans — the opposite of the home feed
        # which prioritizes recency.
        filtered = sorted(filtered, key=lambda a: a.confidence, reverse=True)
        return templates.TemplateResponse(
            request,
            "articles.html",
            {
                "articles": filtered,
                "total": len(all_articles),
                "shown": len(filtered),
                "type": type_param or "",
                "min_confidence": min_confidence,
                "quarantine": quarantine,
                "q": q or "",
            },
        )

    @app.get("/articles/{slug:path}", response_class=HTMLResponse)
    def article_detail(request: Request, slug: str) -> HTMLResponse:
        if slug.endswith(".md"):
            return RedirectResponse(url=f"/articles/{slug[:-3]}", status_code=302)
        path = kb / f"{slug}.md"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Article {slug!r} not found")
        content = path.read_text(encoding="utf-8", errors="replace")
        frontmatter = parse_frontmatter(content) or {}
        rendered_html = _render_markdown(content)
        return templates.TemplateResponse(
            request,
            "article.html",
            {
                "slug": slug,
                "frontmatter": frontmatter,
                "rendered_html": rendered_html,
                "raw_content": content,
            },
        )

    @app.get("/daily", response_class=HTMLResponse)
    def daily_list(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "daily.html",
            {"daily": load_daily_log_index(config.DAILY_DIR)},
        )

    @app.get("/daily/{date}", response_class=HTMLResponse)
    def daily_detail(request: Request, date: str) -> HTMLResponse:
        path = config.DAILY_DIR / f"{date}.md"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Daily log {date} not found")
        content = path.read_text(encoding="utf-8", errors="replace")
        rendered_html = _md.render(_rewrite_wikilinks(content))
        drawer = config.DAILY_DIR / f"{date}.tools.jsonl"
        return templates.TemplateResponse(
            request,
            "daily_detail.html",
            {
                "date": date,
                "rendered_html": rendered_html,
                "has_drawer": drawer.exists(),
            },
        )

    @app.get("/tools", response_class=HTMLResponse)
    def tools_list(request: Request) -> HTMLResponse:
        """List every available tool drawer by date, newest first."""
        rows: list[dict[str, Any]] = []
        if config.DAILY_DIR.exists():
            for path in sorted(
                config.DAILY_DIR.glob("*.tools.jsonl"), reverse=True,
            ):
                date = path.stem.removesuffix(".tools")
                events = load_tool_drawer(config.DAILY_DIR, date)
                summary = summarize_tool_events(events)
                rows.append({
                    "date": date,
                    "total": summary["total"],
                    "errors": summary["errors"],
                    "sessions": summary["sessions"],
                })
        return templates.TemplateResponse(
            request, "tools_list.html", {"rows": rows},
        )

    @app.get("/tools/{date}", response_class=HTMLResponse)
    def tools_detail(request: Request, date: str) -> HTMLResponse:
        events = load_tool_drawer(config.DAILY_DIR, date)
        if not events:
            # Missing file vs empty day are both 404 from the UI's POV —
            # drawers don't get written when the day is quiet.
            raise HTTPException(status_code=404, detail=f"No tool drawer for {date}")
        summary = summarize_tool_events(events)
        return templates.TemplateResponse(
            request,
            "tools_detail.html",
            {
                "date": date,
                "events": events,
                "summary": summary,
            },
        )

    @app.get("/contradictions", response_class=HTMLResponse)
    def contradictions_view(request: Request) -> HTMLResponse:
        slugs = sorted(load_contradictions())
        rows = []
        for slug in slugs:
            path = kb / f"{slug}.md"
            if path.exists():
                summary = _summarize_article(path, kb)
                rows.append({"slug": slug, "article": summary, "exists": True})
            else:
                rows.append({"slug": slug, "article": None, "exists": False})
        return templates.TemplateResponse(
            request, "contradictions.html", {"rows": rows},
        )

    _FT_COLORS: dict[str, str] = {
        "php": "#60a5fa",   # blue
        "js": "#f59e0b",    # amber
        "twig": "#34d399",  # emerald
        "yaml": "#a78bfa",  # purple
    }

    @app.get("/code", response_class=HTMLResponse)
    def code_search(
        request: Request,
        q: str | None = Query(None),
        file_type: str | None = Query(None),
    ) -> HTMLResponse:
        import codebase_store

        valid_ft = file_type if file_type in _FT_COLORS else None
        file_types = sorted(_FT_COLORS)

        results = None
        if q:
            raw = codebase_store.search_codebase(query=q, limit=20, file_type=valid_ft)
            results = []
            for hit in raw:
                meta = hit.get("metadata") or {}
                text = hit.get("text") or ""
                lines = text.splitlines()
                snippet_lines = lines[:60]
                snippet = "\n".join(snippet_lines)
                if len(lines) > 60:
                    snippet += "\n…"
                results.append({
                    "rel_path": hit.get("rel_path") or meta.get("rel_path", ""),
                    "file_type": meta.get("file_type", ""),
                    "start_line": meta.get("start_line"),
                    "end_line": meta.get("end_line"),
                    "symbols": meta.get("symbols", ""),
                    "snippet": snippet,
                    "distance": hit.get("distance", 0.0),
                })

        chroma_total = 0
        type_counts: list[tuple[str, int]] = []
        try:
            chroma_total = codebase_store.stats().get("codebase_chunks", 0)
            ts = codebase_store.type_stats()
            type_counts = [(ft, ts.get(ft, 0)) for ft in file_types]
        except Exception:
            pass

        return templates.TemplateResponse(
            request,
            "code.html",
            {
                "q": q or "",
                "file_type": valid_ft or "",
                "file_types": file_types,
                "results": results,
                "ft_colors": _FT_COLORS,
                "total_chunks": chroma_total,
                "type_counts": type_counts,
            },
        )

    @app.get("/stats", response_class=HTMLResponse)
    def stats(request: Request) -> HTMLResponse:
        flush_state = load_flush_state()
        ingest_state = load_ingest_state()
        chroma = _chroma_stats()
        # Recent flush records — most recent 20, newest first.
        flush_costs = sorted(
            flush_state.get("flush_costs", []),
            key=lambda e: e.get("timestamp", 0),
            reverse=True,
        )[:20]
        # Decorate with a human-readable local-time string.
        for entry in flush_costs:
            ts = entry.get("timestamp", 0)
            try:
                entry["when"] = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            except (TypeError, ValueError, OSError):
                entry["when"] = "?"
        return templates.TemplateResponse(
            request,
            "stats.html",
            {
                "chroma": chroma,
                "flush_costs": flush_costs,
                "today_flush_cost": _today_flush_total(flush_state),
                "ingest_files_count": len(ingest_state.get("ingested", {})),
                "total_flushes_tracked": len(flush_state.get("flush_costs", [])),
            },
        )

    # ── whisper voice-to-prompt endpoints ────────────────────────────
    from whisper.orchestrator import (
        enhance_from_audio,
        enhance_from_transcript,
        NoSpeechError,
    )
    from whisper.types import EnhanceResult as _WhResult
    from dataclasses import asdict as _asdict

    def _result_to_json(r: _WhResult) -> dict:
        return {
            "transcript": r.transcript,
            "enhanced_prompt": r.enhanced_prompt,
            "mode": r.mode,
            "citations": [_asdict(h) for h in r.citations],
            "intent": r.intent,
            "scope_used": r.scope_used,
            "queries_used": r.queries_used,
            "warnings": r.warnings,
            "timings_ms": r.timings_ms,
        }

    @app.get("/whisper", response_class=HTMLResponse)
    def whisper_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "whisper.html",
            {"request": request},
        )

    @app.post("/api/whisper/enhance")
    async def whisper_enhance_endpoint(
        audio: UploadFile = File(...),  # noqa: B008
        mode: str = Form(...),
        language: str = Form("auto"),
    ):
        from fastapi.concurrency import run_in_threadpool
        if mode not in ("verbatim", "rewrite", "clean"):
            raise HTTPException(status_code=422, detail=f"invalid mode: {mode!r}")
        audio_bytes = await audio.read()
        try:
            result = await run_in_threadpool(
                enhance_from_audio, audio_bytes, mode, language,
            )
        except NoSpeechError:
            raise HTTPException(status_code=422, detail="no_speech_detected")
        except Exception as exc:  # noqa: BLE001
            logger.exception("whisper enhance failed")
            raise HTTPException(status_code=503, detail=f"transcription_failed: {exc}")
        return _result_to_json(result)

    @app.post("/api/whisper/re-enhance")
    async def whisper_re_enhance_endpoint(payload: _ReEnhanceRequest = Body(...)) -> dict:  # noqa: B008
        from fastapi.concurrency import run_in_threadpool
        transcript = payload.transcript
        mode = payload.mode
        scope_override = payload.scope_override
        if not transcript.strip():
            raise HTTPException(status_code=422, detail="empty transcript")
        if mode not in ("verbatim", "rewrite", "clean"):
            raise HTTPException(status_code=422, detail=f"invalid mode: {mode!r}")
        result = await run_in_threadpool(
            enhance_from_transcript, transcript, mode, scope_override,
        )
        return _result_to_json(result)

    return app


def main() -> None:
    """Entry point for ``uv run python scripts/viewer.py``.

    Bound to ``127.0.0.1`` on purpose — the viewer has zero auth, so it must
    not be reachable from the LAN. Port 8765 used to avoid Windows Hyper-V
    dynamic port reservations that commonly block 37778.
    """
    import uvicorn
    port = int(os.environ.get("VIEWER_PORT", "9000"))
    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
