"""Live file-watcher daemon for the knowledge base + indexed codebase.

Run alongside ``viewer.py`` to keep both the article vector store and
the codebase chunk store live as you edit. SocratiCode has a similar
"watch" subprocess; the design here is the same idea adapted for the
two-store layout: changes under ``knowledge/concepts/`` and
``knowledge/daily/`` route through the article + daily reindexer, and
changes under the project's source globs route through
``index_codebase.reindex_single``.

Usage:
    uv run python scripts/watch.py            # foreground, Ctrl-C to stop
    uv run python scripts/watch.py --quiet    # suppress per-file lines

Architecture:
    1. ``watchdog.Observer`` produces fs events on a worker thread.
    2. Each event is filtered to a known kind (article / daily /
       codebase) and pushed onto a deque.
    3. A debounce thread drains the deque after ``DEBOUNCE_SECONDS`` of
       quiet, then issues the appropriate reindex call.

Why standalone (not embedded in the MCP server): the MCP servers are
short-lived per-request processes — embedding a watcher there would
either restart it on every tool call (wasteful) or leak threads across
calls (bug-prone). A separate process owns its lifecycle cleanly.
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

# Bootstrap path resolution — same trick as the MCP servers, since this
# script is launched directly via ``uv run``.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402

# Debounce window — the typical "save → autoformat → save" pattern in
# IDEs fires several events within ~500ms, so 2s is comfortably above
# the noise floor without feeling laggy when you actually want feedback.
DEBOUNCE_SECONDS = 2.0

log = logging.getLogger("watch")


def _classify(path: Path) -> str | None:
    """Categorize a changed path. Returns None to ignore (e.g. .git, lock files)."""
    suffix = path.suffix.lower()
    parts = set(path.parts)

    # Editor scratch files / chroma internals — drop fast.
    name = path.name
    if name.startswith(".") or name.endswith("~") or name.endswith(".tmp"):
        return None
    if "chroma" in parts or ".git" in parts or "__pycache__" in parts:
        return None
    if "node_modules" in parts or "vendor" in parts:
        return None

    # Knowledge base: concepts / daily routes are independent reindex paths.
    knowledge = config.KNOWLEDGE_DIR.resolve()
    try:
        rel_to_kb = path.resolve().relative_to(knowledge)
    except ValueError:
        rel_to_kb = None

    if rel_to_kb is not None:
        # Markdown only — yaml frontmatter is part of .md so no separate handler.
        if suffix != ".md":
            return None
        first = rel_to_kb.parts[0] if rel_to_kb.parts else ""
        if first == "concepts":
            return "article"
        if first == "daily":
            return "daily"
        return None  # other knowledge subtrees (research, ingest checkpoints)

    # Codebase: must be a supported extension AND inside one of the
    # walked source dirs (covered by index_codebase's exclude rules).
    from index_codebase import _supported_extensions

    if suffix in _supported_extensions():
        return "codebase"
    return None


class _DebouncedReindexer:
    """Collect events; flush them after DEBOUNCE_SECONDS of quiet."""

    def __init__(self, callback: Callable[[set[Path], dict[str, set[Path]]], None]) -> None:
        self._callback = callback
        self._pending: dict[str, set[Path]] = {
            "article": set(),
            "daily": set(),
            "codebase": set(),
        }
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def submit(self, kind: str, path: Path) -> None:
        with self._lock:
            self._pending.setdefault(kind, set()).add(path)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(DEBOUNCE_SECONDS, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self) -> None:
        with self._lock:
            snapshot = {k: set(v) for k, v in self._pending.items() if v}
            self._pending = {"article": set(), "daily": set(), "codebase": set()}
            all_paths = {p for paths in snapshot.values() for p in paths}
        if not all_paths:
            return
        try:
            self._callback(all_paths, snapshot)
        except Exception:
            # Never let a reindex failure kill the watcher — log and continue.
            log.exception("Reindex callback raised; watcher continues.")


def _process_batch(all_paths: set[Path], by_kind: dict[str, set[Path]]) -> None:
    """Issue the appropriate reindex calls for one debounced batch."""
    log.info("Reindex batch: %d files (%s)", len(all_paths), {k: len(v) for k, v in by_kind.items()})

    article_paths = by_kind.get("article", set())
    daily_paths = by_kind.get("daily", set())
    codebase_paths = by_kind.get("codebase", set())

    if article_paths:
        from reindex import reindex_articles

        embedded, skipped = reindex_articles(force=False)
        log.info("  articles: embedded=%d skipped=%d", embedded, skipped)

    if daily_paths:
        try:
            from reindex import reindex_daily

            embedded, skipped = reindex_daily(force=False)
            log.info("  daily: embedded=%d skipped=%d", embedded, skipped)
        except ModuleNotFoundError as exc:
            if getattr(exc, "name", None) != "chunk_daily":
                raise
            log.warning("  daily: chunk_daily module unavailable; skipping")

    if codebase_paths:
        from index_codebase import reindex_single

        for p in codebase_paths:
            try:
                n = reindex_single(str(p))
                log.info("  codebase: %s → %d chunks", p, n)
            except Exception:
                log.exception("  codebase: failed to reindex %s", p)


def _watch_paths() -> list[Path]:
    """Resolve the directories the observer should listen to.

    Watching the project root is overkill (and on Windows, the recursive
    watch on the whole tree generates a lot of noise from ``vendor/`` and
    ``var/cache/``). Instead, watch only the knowledge dir and the
    top-level source dirs that ``index_codebase.SOURCE_PATTERNS`` walks.
    """
    paths: list[Path] = []
    if config.KNOWLEDGE_DIR.exists():
        paths.append(config.KNOWLEDGE_DIR)
    project_root = config.PROJECT_ROOT
    for sub in ("src", "assets", "templates", "config"):
        d = project_root / sub
        if d.exists():
            paths.append(d)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Watch knowledge + codebase paths and reindex on change.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Log at WARNING instead of INFO — suppress per-file lines.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    debouncer = _DebouncedReindexer(_process_batch)

    class Handler(FileSystemEventHandler):
        def on_any_event(self, event):
            if event.is_directory:
                return
            # `closed`/`opened` events fire on macOS but carry no payload
            # we care about; rely on `created`/`modified`/`moved`.
            if event.event_type not in {"created", "modified", "moved"}:
                return
            target = event.dest_path if event.event_type == "moved" else event.src_path
            kind = _classify(Path(target))
            if kind is None:
                return
            debouncer.submit(kind, Path(target))

    observer = Observer()
    handler = Handler()
    paths = _watch_paths()
    if not paths:
        log.error("No watch paths exist — nothing to do.")
        return 1
    for p in paths:
        observer.schedule(handler, str(p), recursive=True)
        log.info("Watching %s", p)
    observer.start()
    log.info("Watcher up. Ctrl-C to stop.")

    try:
        while observer.is_alive():
            observer.join(timeout=1)
    except KeyboardInterrupt:
        log.info("Stop requested — shutting down observer.")
    finally:
        observer.stop()
        observer.join()

    return 0


if __name__ == "__main__":
    sys.exit(main())
