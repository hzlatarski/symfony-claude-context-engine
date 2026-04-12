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


# Auto-register built-in handlers on import
from source_handlers import markdown as _md  # noqa: E402, F401
