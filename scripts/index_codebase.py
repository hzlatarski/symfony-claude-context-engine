"""Index source code files into the ChromaDB codebase collection.

Usage:
    python scripts/index_codebase.py            # incremental (changed files only)
    python scripts/index_codebase.py --all      # force-reindex everything (force sync)
    python scripts/index_codebase.py --file /abs/path/to/file.php

Source patterns (relative to config.PROJECT_ROOT):
    src/**/*.php
    assets/controllers/**/*.js
    templates/**/*.twig
    config/**/*.yaml

Chunking: PHP and JS use tree-sitter AST chunking — one chunk per
class/method/function so retrieval lands on whole units instead of
mid-method line slices. Twig and YAML (and PHP/JS files where AST
parsing fails) fall back to 150-line windows with 30-line overlap.
PHP additionally extracts class/interface/trait/function names into
the 'symbols' metadata field for more precise BM25-style matching.

State: state.json["codebase_hashes"] = {rel_path: sha256[:16]}
Only re-chunks files whose hash changed (unless --all).
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402
from codebase_store import delete_chunks_for_file, upsert_chunk  # noqa: E402
from utils import load_state, save_state  # noqa: E402

CHUNK_SIZE = 150
CHUNK_OVERLAP = 30

# SOURCE_PATTERNS drives both the codebase walk and the per-chunk
# "description" metadata. Descriptions are LLM-facing — they answer
# "when should the agent reach for this source group?". Surfaced via
# `search_codebase` results and the `list_sources` MCP tool.
#
# Format: (file_type, [glob_patterns], description). Globs are evaluated
# from PROJECT_ROOT.
SOURCE_PATTERNS: list[tuple[str, list[str], str]] = [
    (
        "php",
        ["src/**/*.php"],
        "Symfony application code — controllers, services, entities, "
        "repositories, commands. Check before writing new business logic "
        "to match existing service-layer and DI conventions.",
    ),
    (
        "js",
        ["assets/controllers/**/*.js"],
        "Stimulus controllers — JS event handlers bound to Twig templates. "
        "Check before adding interactive UI to find an existing controller "
        "you can extend instead of duplicating behaviour.",
    ),
    (
        "twig",
        ["templates/**/*.twig"],
        "Twig templates — layouts, partials, page bodies. Check before "
        "adding new pages to find the right base layout and existing "
        "partials to reuse.",
    ),
    (
        "yaml",
        ["config/**/*.yaml"],
        "Symfony configuration — services, routes, security, framework "
        "bundles. Check before adding new services or routes.",
    ),
]


def _expand_extra_patterns() -> list[tuple[str, list[str], str]]:
    """Materialise SOURCE_PATTERNS entries for MEMORY_COMPILER_EXTRA_EXTENSIONS.

    Each extra extension (e.g. ``.neon``) becomes its own file_type whose
    globs are scoped to the same top-level dirs the built-in patterns
    walk (``src/``, ``assets/``, ``templates/``, ``config/``). Restricting
    to those known dirs avoids walking ``vendor/`` / ``node_modules/`` for
    one-off custom extensions. The description is generic — users who
    want a precise description should edit SOURCE_PATTERNS directly.
    """
    if not config.EXTRA_EXTENSIONS:
        return []
    scoped_dirs = ("src", "assets", "templates", "config")
    out: list[tuple[str, list[str], str]] = []
    for ext in config.EXTRA_EXTENSIONS:
        ftype = ext.lstrip(".")
        globs = [f"{d}/**/*{ext}" for d in scoped_dirs]
        out.append((
            ftype,
            globs,
            f"Custom extension {ext!r} declared via "
            f"MEMORY_COMPILER_EXTRA_EXTENSIONS — content treated as plaintext.",
        ))
    return out


_EXCLUDE_DIRS = {"vendor", "var", "node_modules", "public"}

# Matches PHP class/interface/trait/enum declarations and method definitions.
_PHP_SYMBOL_RE = re.compile(
    r"^\s*(?:(?:abstract|final|readonly)\s+)*(?:class|interface|trait|enum)\s+(\w+)"
    r"|^\s*(?:public|protected|private|static|abstract|final)[\s\w]*\bfunction\s+(\w+)",
    re.MULTILINE,
)


def _is_excluded(path: Path) -> bool:
    """Return True if any path component is in the exclusion set."""
    return bool(set(path.parts) & _EXCLUDE_DIRS)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _extract_symbols(text: str, file_type: str) -> str:
    """Extract PHP class/method names as a comma-joined string for metadata."""
    if file_type != "php":
        return ""
    names = [m.group(1) or m.group(2) for m in _PHP_SYMBOL_RE.finditer(text)]
    return ",".join(n for n in names if n)


def chunk_file(text: str, file_type: str = "") -> list[tuple[int, int, str]]:
    """Split text into (start_line, end_line, chunk_text) tuples.

    For PHP and JS, attempts tree-sitter AST chunking — chunks fall on
    class / method / function boundaries instead of arbitrary line offsets.
    Falls through to 150-line windows with 30-line overlap when the file
    type is unsupported, when tree-sitter isn't installed, or when the AST
    walk produces no top-level declarations (e.g. PHP config files that
    are bare ``return [...];`` arrays).

    Line numbers are 1-based. Returns an empty list for empty input.
    The ``file_type`` argument is optional for backward compatibility.
    """
    if file_type:
        from ast_chunker import chunk_ast, is_supported
        if is_supported(file_type):
            ast_chunks = chunk_ast(text, file_type)
            if ast_chunks is not None:
                return ast_chunks
    return _chunk_lines(text)


def _chunk_lines(text: str) -> list[tuple[int, int, str]]:
    """Naive line-window chunker — fallback when AST chunking can't apply."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    chunks: list[tuple[int, int, str]] = []
    i = 0
    while i < len(lines):
        end = min(i + CHUNK_SIZE, len(lines))
        chunks.append((i + 1, end, "".join(lines[i:end])))
        if end == len(lines):
            break
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def index_file(path: Path) -> int:
    """Chunk and upsert one file. Deletes old chunks first.

    Returns number of chunks written. Returns 0 for empty files.
    """
    try:
        rel = str(path.relative_to(config.PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        rel = path.name

    file_type = path.suffix.lstrip(".")

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0

    if not text.strip():
        return 0

    symbols = _extract_symbols(text, file_type)
    delete_chunks_for_file(rel)

    description = _description_for_file_type(file_type)

    chunks = chunk_file(text, file_type)
    for idx, (start, end, chunk_text) in enumerate(chunks):
        chunk_id = f"{rel}::{idx}"
        metadata: dict = {
            "file_type": file_type,
            "start_line": start,
            "end_line": end,
        }
        if symbols:
            metadata["symbols"] = symbols
        if description:
            # source_description tells the agent *when* to consult this file
            # group. SocratiCode's context-artifact pattern: "check this
            # before writing migrations" beats undirected file reading.
            metadata["source_description"] = description
        upsert_chunk(chunk_id, rel, chunk_text, metadata)

    return len(chunks)


def _all_source_groups() -> list[tuple[str, list[str], str]]:
    """SOURCE_PATTERNS plus EXTRA_EXTENSIONS, deduped on file_type."""
    seen: set[str] = set()
    out: list[tuple[str, list[str], str]] = []
    for entry in (*SOURCE_PATTERNS, *_expand_extra_patterns()):
        ftype = entry[0]
        if ftype in seen:
            continue
        seen.add(ftype)
        out.append(entry)
    return out


def _description_for_file_type(file_type: str) -> str:
    for ftype, _globs, desc in _all_source_groups():
        if ftype == file_type:
            return desc
    return ""


def _supported_extensions() -> set[str]:
    """Set of dot-prefixed extensions accepted by reindex_single."""
    base = {".php", ".js", ".twig", ".yaml"}
    base.update(config.EXTRA_EXTENSIONS)
    return base


def list_source_files() -> list[tuple[str, Path]]:
    """Return all indexed source files as (file_type, Path) pairs."""
    results: list[tuple[str, Path]] = []
    for file_type, patterns, _desc in _all_source_groups():
        for pattern in patterns:
            for match in config.PROJECT_ROOT.glob(pattern):
                if match.is_file() and not _is_excluded(match):
                    results.append((file_type, match))
    return results


def list_source_groups() -> list[dict]:
    """Return the full source-group catalog for the list_sources MCP tool.

    Each entry: ``{file_type, patterns, description, chunk_count}``. The
    chunk count is read live from the codebase Chroma collection so the
    LLM can tell at a glance which groups are populated and which are
    empty (suggesting an out-of-date index).
    """
    from codebase_store import _codebase_collection  # local import — reuse client

    coll = _codebase_collection()
    out: list[dict] = []
    for ftype, patterns, desc in _all_source_groups():
        try:
            res = coll.get(where={"file_type": {"$eq": ftype}}, include=[])
            count = len(res.get("ids") or [])
        except Exception:
            count = 0
        out.append({
            "file_type": ftype,
            "patterns": list(patterns),
            "description": desc,
            "chunk_count": count,
        })
    return out


def reindex_all(force: bool = False, progress_callback=None) -> tuple[int, int]:
    """Re-index changed (or all, if force) source files.

    Returns (indexed_count, skipped_count). Always persists state via
    try/finally so a mid-run error doesn't lose already-indexed hashes.

    If ``progress_callback`` is supplied, it is invoked once per file
    *scanned* (whether indexed or skipped) as ``cb(scanned, total, rel)``.
    The default per-file print is suppressed in that case so a caller
    rendering its own progress bar isn't fighting the script for stdout.
    """
    state = load_state()
    hashes: dict[str, str] = state.setdefault("codebase_hashes", {})

    files = list_source_files()
    total = len(files)
    indexed = 0
    skipped = 0
    try:
        for scanned, (_ftype, path) in enumerate(files, 1):
            rel = str(path.relative_to(config.PROJECT_ROOT)).replace("\\", "/")
            h = _file_hash(path)
            if not force and hashes.get(rel) == h:
                skipped += 1
                if progress_callback:
                    progress_callback(scanned, total, rel)
                continue
            n = index_file(path)
            hashes[rel] = h
            indexed += 1
            if progress_callback:
                progress_callback(scanned, total, rel)
            else:
                print(f"  [{indexed}] {n} chunks  {rel}")
    finally:
        save_state(state)

    return indexed, skipped


def reindex_single(path_str: str) -> int:
    """Re-index one file unconditionally. Updates hash cache.

    Returns chunk count. Returns 0 for unsupported extensions or missing files.
    Used by the Claude Code PostToolUse hook after Write/Edit.
    """
    path = Path(path_str).resolve()
    if not path.exists():
        print(f"  skip (not found): {path_str}", file=sys.stderr)
        return 0

    if path.suffix.lower() not in _supported_extensions():
        return 0

    state = load_state()
    hashes: dict[str, str] = state.setdefault("codebase_hashes", {})
    try:
        n = index_file(path)
        try:
            rel = str(path.relative_to(config.PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            rel = path.name
        if n > 0:
            hashes[rel] = _file_hash(path)
            print(f"  indexed {n} chunks: {rel}")
    finally:
        save_state(state)

    return n


def main() -> int:
    parser = argparse.ArgumentParser(description="Index source code into ChromaDB")
    parser.add_argument(
        "--all", action="store_true",
        help="Force re-index all files (force sync — equivalent to Auggie's refresh button)",
    )
    parser.add_argument(
        "--file", metavar="PATH",
        help="Index a single file (used by the PostToolUse auto-sync hook)",
    )
    parser.add_argument(
        "--progress", action="store_true",
        help=(
            "Emit machine-readable progress lines on stdout instead of the "
            "default per-file human output. Each line is "
            "'PROGRESS\\t<scanned>\\t<total>\\t<rel_path>'. Used by install.py "
            "to render a progress bar without parsing free-form output."
        ),
    )
    args = parser.parse_args()

    if args.file:
        n = reindex_single(args.file)
        print(f"Done: {n} chunks written")
        return 0

    if args.progress:
        def _cb(scanned: int, total: int, rel: str) -> None:
            print(f"PROGRESS\t{scanned}\t{total}\t{rel}", flush=True)
        indexed, skipped = reindex_all(force=args.all, progress_callback=_cb)
    else:
        print("Scanning source files…")
        indexed, skipped = reindex_all(force=args.all)
    print(f"\nDone: indexed {indexed} files, skipped {skipped} unchanged")

    from codebase_store import stats
    s = stats()
    print(f"Total codebase chunks in store: {s['codebase_chunks']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
