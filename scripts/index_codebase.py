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

Chunking: 150-line windows with 30-line overlap.
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

SOURCE_PATTERNS: list[tuple[str, list[str]]] = [
    ("php",  ["src/**/*.php"]),
    ("js",   ["assets/controllers/**/*.js"]),
    ("twig", ["templates/**/*.twig"]),
    ("yaml", ["config/**/*.yaml"]),
]

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


def chunk_file(text: str) -> list[tuple[int, int, str]]:
    """Split text into (start_line, end_line, chunk_text) tuples.

    Uses 150-line windows with 30-line overlap. Line numbers are 1-based.
    Returns an empty list for empty input.
    """
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

    chunks = chunk_file(text)
    for idx, (start, end, chunk_text) in enumerate(chunks):
        chunk_id = f"{rel}::{idx}"
        metadata: dict = {
            "file_type": file_type,
            "start_line": start,
            "end_line": end,
        }
        if symbols:
            metadata["symbols"] = symbols
        upsert_chunk(chunk_id, rel, chunk_text, metadata)

    return len(chunks)


def list_source_files() -> list[tuple[str, Path]]:
    """Return all indexed source files as (file_type, Path) pairs."""
    results: list[tuple[str, Path]] = []
    for file_type, patterns in SOURCE_PATTERNS:
        for pattern in patterns:
            for match in config.PROJECT_ROOT.glob(pattern):
                if match.is_file() and not _is_excluded(match):
                    results.append((file_type, match))
    return results


def reindex_all(force: bool = False) -> tuple[int, int]:
    """Re-index changed (or all, if force) source files.

    Returns (indexed_count, skipped_count). Always persists state via
    try/finally so a mid-run error doesn't lose already-indexed hashes.
    """
    state = load_state()
    hashes: dict[str, str] = state.setdefault("codebase_hashes", {})

    indexed = 0
    skipped = 0
    try:
        for _ftype, path in list_source_files():
            rel = str(path.relative_to(config.PROJECT_ROOT)).replace("\\", "/")
            h = _file_hash(path)
            if not force and hashes.get(rel) == h:
                skipped += 1
                continue
            n = index_file(path)
            hashes[rel] = h
            indexed += 1
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

    if path.suffix.lower() not in {".php", ".js", ".twig", ".yaml"}:
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
    args = parser.parse_args()

    if args.file:
        n = reindex_single(args.file)
        print(f"Done: {n} chunks written")
        return 0

    print("Scanning source files…")
    indexed, skipped = reindex_all(force=args.all)
    print(f"\nDone: indexed {indexed} files, skipped {skipped} unchanged")

    from codebase_store import stats
    s = stats()
    print(f"Total codebase chunks in store: {s['codebase_chunks']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
