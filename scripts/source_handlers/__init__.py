"""Pluggable source handler registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class SourceDocument:
    """Extracted content from a source file, ready for LLM ingestion."""

    content: str
    path: Path
    frontmatter: dict = field(default_factory=dict)
    mtime: float = 0.0


ExtractFn = Callable[[Path], SourceDocument]

_HANDLERS: dict[str, ExtractFn] = {}


def register(type_name: str, fn: ExtractFn) -> None:
    _HANDLERS[type_name] = fn


def get_handler(type_name: str) -> ExtractFn:
    if type_name not in _HANDLERS:
        available = ", ".join(sorted(_HANDLERS)) or "(none)"
        raise KeyError(
            f"No handler registered for source type '{type_name}'. "
            f"Available: {available}"
        )
    return _HANDLERS[type_name]


def available_types() -> list[str]:
    return sorted(_HANDLERS)


# Auto-register built-in handlers on import.
#
# These are RELATIVE imports on purpose. The compiler is importable under two
# regimes — ``source_handlers`` (ingest.py, with scripts/ on sys.path) and
# ``scripts.source_handlers`` (tests, MCP server). An absolute self-import
# binds the submodule to whichever spelling is written here, so the *other*
# spelling initializes a second, permanently empty ``_HANDLERS`` dict and
# every get_handler() call against it raises "no handler registered".
# ``from . import`` resolves against __package__, so both spellings register.
#
# ``pdf`` registers unconditionally — its extractor libraries are optional and
# resolved lazily inside extract(), so importing it costs nothing and a project
# without them still gets "pdf" in available_types() plus an actionable error
# at use time, rather than a confusing "no handler for type 'pdf'".
from . import markdown as _md  # noqa: E402, F401
from . import pdf as _pdf  # noqa: E402, F401
