"""Dataclasses shared across whisper modules.

Kept in a dedicated module to avoid circular imports: retrieve.py produces
Hit, enhance.py consumes Hit and produces EnhanceResult, the FastAPI
endpoints in viewer.py consume EnhanceResult.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Hit:
    """A single retrieval hit from one of the three channels.

    `source` is the retrieval channel (article/code/daily). `category` is
    a sub-classification from sources.yaml for article hits only
    (implementation-plans, design-specs, project-docs, governance,
    captured-memory, research), and is None for code/daily hits.
    """

    id: str
    source: Literal["article", "code", "daily"]
    category: str | None
    path: str
    title: str
    snippet: str
    full_body: str | None
    score: float
    symbols: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhanceResult:
    """Output of the enhance pipeline — the thing the HTTP endpoint returns."""

    transcript: str
    enhanced_prompt: str
    mode: Literal["verbatim", "rewrite", "clean"]
    citations: list[Hit]
    intent: str
    scope_used: list[str]
    queries_used: list[str]
    warnings: list[str]
    timings_ms: dict[str, int]
