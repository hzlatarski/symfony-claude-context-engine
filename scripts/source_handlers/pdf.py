"""PDF source handler — extracts page text so PDFs can be ingested like markdown.

The pipeline was markdown-only, so any PDF in a project (compliance docs,
customer curricula, specs delivered as PDF, papers) was invisible to the
knowledge base. This closes that gap without adding a second retrieval
stack: a PDF becomes text, and everything downstream — dedup pre-flight,
the ingest prompt, ``[src:]`` anchors, checkpointing — works unchanged.

**No hard dependency.** The compiler is MIT; PyMuPDF is AGPL-or-commercial.
Forcing it on every install would impose a licence nobody asked for, and
would grow the footprint for the majority who never ingest a PDF. So the
extractor is a preference chain over libraries you opt into:

1. ``pymupdf4llm`` — best output: real markdown with headings and tables.
   Install ``pymupdf4llm`` (pulls PyMuPDF, **AGPL-3.0** unless you hold a
   commercial licence).
2. ``pypdf`` — plain text, no structure, but **BSD** and pure Python.

Neither present → a ``RuntimeError`` naming both, rather than a silent
empty ingest.

Two guards exist because both failure modes are silent and expensive:

* **Character budget.** ``ingest.py`` inlines ``doc.content`` into a prompt
  that already carries AGENTS.md, the wiki index and compiled truth. An
  unbounded 300-page PDF blows the context window and fails opaquely.
  Content is truncated at a page boundary with a visible notice; override
  the budget with ``MEMORY_COMPILER_PDF_MAX_CHARS``.
* **Empty extraction.** A scanned, image-only PDF yields no text at all.
  That must raise, not ingest an empty document — an empty ingest looks
  like a successful one.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import SourceDocument, register

# Roughly 30k tokens of source. Leaves headroom in the ingest prompt for
# AGENTS.md (~22 KB), the compact wiki index, compiled truth and the dedup
# pre-flight block.
DEFAULT_MAX_CHARS = 120_000


def _max_chars() -> int:
    """Character budget, overridable via ``MEMORY_COMPILER_PDF_MAX_CHARS``."""
    raw = os.environ.get("MEMORY_COMPILER_PDF_MAX_CHARS", "").strip()
    if not raw:
        return DEFAULT_MAX_CHARS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_CHARS
    return value if value > 0 else DEFAULT_MAX_CHARS


def _pages_via_pymupdf4llm(path: Path) -> list[str]:
    import pymupdf4llm

    chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    return [(chunk.get("text") or "") for chunk in chunks]


def _pages_via_pypdf(path: Path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]


# Ordered best-first. Each entry is (extractor name, module to import, fn).
_EXTRACTORS = (
    ("pymupdf4llm", "pymupdf4llm", _pages_via_pymupdf4llm),
    ("pypdf", "pypdf", _pages_via_pypdf),
)


def _select_extractor():
    """Return ``(name, fn)`` for the best available backend.

    Imports are attempted here rather than at module import time so that a
    project with no PDF sources never pays for — or fails on — a library it
    does not have. ``source_handlers`` is imported on every ingest.
    """
    import importlib.util

    for name, module, fn in _EXTRACTORS:
        if importlib.util.find_spec(module) is not None:
            return name, fn

    raise RuntimeError(
        "PDF ingestion needs an extractor and none is installed. Choose one:\n"
        "  uv add pymupdf4llm   # best output (markdown, tables) — AGPL-3.0\n"
        "  uv add pypdf         # plain text only — BSD, pure Python"
    )


def _assemble(pages: list[str], budget: int) -> tuple[str, int, bool]:
    """Join page texts under ``budget`` chars.

    Returns ``(content, pages_used, truncated)``. Truncation lands on a page
    boundary so no page is ever half-fed to the compiler — except when page
    one alone exceeds the budget, where a hard cut is the only alternative
    to extracting nothing.
    """
    parts: list[str] = []
    used = 0
    pages_used = 0

    for number, text in enumerate(pages, start=1):
        text = (text or "").strip()
        if not text:
            continue
        block = f"[page {number}]\n\n{text}"
        if used + len(block) > budget:
            if not parts:
                # One oversized page: hard-cut rather than return nothing.
                return block[:budget], 1, True
            break
        parts.append(block)
        used += len(block)
        pages_used += 1

    truncated = pages_used < sum(1 for p in pages if (p or "").strip())
    return "\n\n".join(parts), pages_used, truncated


def extract(path: Path) -> SourceDocument:
    """Extract text from a PDF into an ingestible ``SourceDocument``.

    Raises:
        RuntimeError: no extractor library installed.
        ValueError: the PDF yielded no extractable text (typically a scanned
            document that needs OCR).
    """
    extractor_name, extractor = _select_extractor()

    try:
        pages = extractor(path)
    except Exception as exc:  # corrupt / encrypted / unsupported PDF
        raise ValueError(f"Could not read PDF {path.name} via {extractor_name}: {exc}") from exc

    content, pages_used, truncated = _assemble(pages, _max_chars())

    if not content.strip():
        raise ValueError(
            f"No extractable text in {path.name} ({len(pages)} page(s) scanned "
            f"via {extractor_name}). It is most likely a scanned/image-only PDF — "
            f"run OCR over it first, or exclude it from sources.yaml."
        )

    if truncated:
        content += (
            f"\n\n[TRUNCATED] Only the first {pages_used} of {len(pages)} pages fit "
            f"the {_max_chars()}-character ingest budget. Raise "
            f"MEMORY_COMPILER_PDF_MAX_CHARS or split the document to ingest the rest."
        )

    return SourceDocument(
        content=content,
        path=path,
        frontmatter={
            "source_format": "pdf",
            "extractor": extractor_name,
            "pages": len(pages),
            "pages_extracted": pages_used,
            "truncated": truncated,
        },
        mtime=path.stat().st_mtime,
    )


register("pdf", extract)
