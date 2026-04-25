"""AST-based code chunking for PHP and JavaScript via tree-sitter.

Splits source at semantic boundaries (classes, methods, top-level functions)
instead of naive line windows. The motivation: line-window chunks routinely
cut mid-method, so when retrieval lands on a chunk the LLM sees half a
method's tail and half another method's head. Method-aligned chunks keep
each unit whole, which both improves embedding quality and means a single
``get_codebase`` hit yields a complete, callable-shaped excerpt.

Public surface:
    is_supported(file_type) -> bool
    chunk_ast(text, file_type) -> list[(start, end, text)] | None

A ``None`` return signals "fall back to line-window chunker" — callers
should always have that fallback wired. Reasons we return ``None``:
    - file_type isn't ``"php"`` or ``"js"``
    - the tree-sitter import or parser construction fails
    - the parsed tree contains no top-level declarations (e.g. a config
      file written as ``<?php return [...]``)

Chunk sizing rules:
    - declarations <= MAX_CHUNK_LINES: one chunk per declaration
    - declarations > MAX_CHUNK_LINES: split into header + per-method chunks;
      monster methods get window-split inside themselves so we never emit
      a single chunk longer than the line-window fallback would
    - prelude (use/import/namespace) and inter-declaration content collected
      into their own small chunks
    - tiny adjacent chunks (< MIN_CHUNK_LINES) merge with their neighbour
"""
from __future__ import annotations

from typing import Any

# Sized to keep AST chunks comparable to the line-window fallback. The
# window chunker uses CHUNK_SIZE=150; we allow a single declaration up to
# ~2.5x that before splitting — most real classes fit, but a 600-line god
# class triggers the per-method split that's the whole point of this module.
MAX_CHUNK_LINES = 400
MIN_CHUNK_LINES = 5

# Window-split parameters used when a single method exceeds MAX_CHUNK_LINES.
# Match the line-window chunker so callers don't see different sizing
# behaviour depending on which path produced the chunk.
_WINDOW_SIZE = 150
_WINDOW_OVERLAP = 30

_PHP_TOP_NODES = frozenset({
    "class_declaration",
    "interface_declaration",
    "trait_declaration",
    "enum_declaration",
    "function_definition",
})
_PHP_METHOD_NODES = frozenset({"method_declaration"})

_JS_TOP_NODES = frozenset({
    "class_declaration",
    "function_declaration",
    "lexical_declaration",
    "export_statement",
})
_JS_METHOD_NODES = frozenset({"method_definition"})

_PARSER_CACHE: dict[str, Any] = {}


def is_supported(file_type: str) -> bool:
    return file_type in {"php", "js"}


def _get_parser(file_type: str):
    """Return a cached tree-sitter Parser, or None if the deps are missing.

    Lazy import so a missing wheel doesn't break callers that fall back to
    line windows. The Parser instance is reused across calls — tree-sitter
    parsers are stateless between parses, only the Tree they emit holds state.
    """
    if file_type in _PARSER_CACHE:
        return _PARSER_CACHE[file_type]

    try:
        from tree_sitter import Language, Parser
    except ImportError:
        _PARSER_CACHE[file_type] = None
        return None

    try:
        if file_type == "php":
            import tree_sitter_php
            language = Language(tree_sitter_php.language_php())
        elif file_type == "js":
            import tree_sitter_javascript
            language = Language(tree_sitter_javascript.language())
        else:
            _PARSER_CACHE[file_type] = None
            return None
    except (ImportError, AttributeError, ValueError):
        _PARSER_CACHE[file_type] = None
        return None

    parser = Parser(language)
    _PARSER_CACHE[file_type] = parser
    return parser


def _node_span(node) -> tuple[int, int]:
    """Return 1-based (start_line, end_line) inclusive."""
    return node.start_point[0] + 1, node.end_point[0] + 1


def _slice(lines: list[str], start: int, end: int) -> str:
    """Slice ``lines`` (0-indexed, with newlines) by 1-based inclusive range."""
    return "".join(lines[start - 1:end])


def _find_methods(node, method_types: frozenset[str]) -> list:
    """Collect every descendant whose type is in ``method_types``.

    Walks the full subtree because methods sit inside ``declaration_list``
    (PHP) or ``class_body`` (JS) — not as direct children of the class node.
    Order is preserved: tree-sitter children are returned in source order.
    """
    out = []
    for child in node.children:
        if child.type in method_types:
            out.append(child)
        else:
            out.extend(_find_methods(child, method_types))
    return out


def _window_split(
    lines: list[str],
    start: int,
    end: int,
) -> list[tuple[int, int, str]]:
    """Window-split a 1-based inclusive range. Used for over-large methods."""
    region = lines[start - 1:end]
    if not region:
        return []
    chunks: list[tuple[int, int, str]] = []
    offset = start - 1
    i = 0
    while i < len(region):
        sub_end = min(i + _WINDOW_SIZE, len(region))
        chunks.append((
            offset + i + 1,
            offset + sub_end,
            "".join(region[i:sub_end]),
        ))
        if sub_end == len(region):
            break
        i += _WINDOW_SIZE - _WINDOW_OVERLAP
    return chunks


def _emit_declaration(
    node,
    lines: list[str],
    method_types: frozenset[str],
) -> list[tuple[int, int, str]]:
    """Chunk a single top-level declaration node.

    Small declarations come back as one chunk. Large ones get split into
    a header (everything up to the first method), one chunk per method,
    and a trailing footer if the declaration has content after the last
    method (rare but valid — closing braces, interface defaults, etc.).
    """
    start, end = _node_span(node)
    if (end - start + 1) <= MAX_CHUNK_LINES:
        return [(start, end, _slice(lines, start, end))]

    methods = _find_methods(node, method_types)
    if not methods:
        return _window_split(lines, start, end)

    chunks: list[tuple[int, int, str]] = []

    first_method_start = methods[0].start_point[0] + 1
    if first_method_start > start:
        chunks.append((
            start,
            first_method_start - 1,
            _slice(lines, start, first_method_start - 1),
        ))

    for method in methods:
        m_start, m_end = _node_span(method)
        if (m_end - m_start + 1) > MAX_CHUNK_LINES:
            chunks.extend(_window_split(lines, m_start, m_end))
        else:
            chunks.append((m_start, m_end, _slice(lines, m_start, m_end)))

    last_method_end = methods[-1].end_point[0] + 1
    if end > last_method_end:
        footer = _slice(lines, last_method_end + 1, end)
        if footer.strip():
            chunks.append((last_method_end + 1, end, footer))

    return chunks


def _merge_tiny(
    chunks: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    """Merge any chunk shorter than MIN_CHUNK_LINES into its neighbour.

    Walks left-to-right: a tiny chunk merges into the previous chunk if
    one exists, otherwise into the next. Single-pass, so a run of three
    tiny chunks collapses cleanly. Preserves source order and line ranges.
    """
    if not chunks:
        return chunks

    merged: list[tuple[int, int, str]] = []
    for chunk in chunks:
        size = chunk[1] - chunk[0] + 1
        if size < MIN_CHUNK_LINES and merged:
            prev = merged[-1]
            merged[-1] = (prev[0], chunk[1], prev[2] + chunk[2])
        else:
            merged.append(chunk)

    if len(merged) >= 2 and (merged[0][1] - merged[0][0] + 1) < MIN_CHUNK_LINES:
        first, second = merged[0], merged[1]
        merged[:2] = [(first[0], second[1], first[2] + second[2])]

    return merged


def chunk_ast(
    text: str,
    file_type: str,
) -> list[tuple[int, int, str]] | None:
    """Return semantic chunks for ``text``, or None to signal fallback."""
    if not is_supported(file_type):
        return None
    parser = _get_parser(file_type)
    if parser is None:
        return None

    try:
        tree = parser.parse(text.encode("utf-8"))
    except (ValueError, RuntimeError):
        return None

    lines = text.splitlines(keepends=True)
    if not lines:
        return []

    top_types = _PHP_TOP_NODES if file_type == "php" else _JS_TOP_NODES
    method_types = _PHP_METHOD_NODES if file_type == "php" else _JS_METHOD_NODES

    raw: list[tuple[int, int, str]] = []
    cursor = 0  # 1-based: highest line covered so far
    found_decl = False

    for child in tree.root_node.children:
        if child.type not in top_types:
            continue
        found_decl = True
        start, end = _node_span(child)

        if start > cursor + 1:
            prelude_text = _slice(lines, cursor + 1, start - 1)
            if prelude_text.strip():
                raw.append((cursor + 1, start - 1, prelude_text))

        raw.extend(_emit_declaration(child, lines, method_types))
        cursor = end

    if not found_decl:
        # No class/function/interface anywhere — let the line-window
        # fallback handle it. Bare config-style PHP and IIFE-style JS land
        # here and are better served by uniform window slicing than by a
        # single all-content chunk.
        return None

    total = len(lines)
    if cursor < total:
        trailing = _slice(lines, cursor + 1, total)
        if trailing.strip():
            raw.append((cursor + 1, total, trailing))

    return _merge_tiny(raw)
