"""Guard tests for whisper prompt constants.

These are not behavioral tests — they assert the prompts contain the
instruction substrings that downstream code and JSON-mode schemas
depend on. If a prompt is refactored and drops one of these anchors,
the tests catch it before runtime.
"""
from __future__ import annotations

from whisper.prompts import (
    CLEAN_SYSTEM_PROMPT,
    QUERY_EXPANSION_SYSTEM_PROMPT,
    REWRITE_SYSTEM_PROMPT,
)


def test_query_expansion_prompt_contains_json_and_schema_anchors():
    p = QUERY_EXPANSION_SYSTEM_PROMPT
    assert "JSON" in p
    assert "queries" in p
    assert "intent" in p
    assert "scope" in p
    assert "articles" in p
    assert "code" in p
    assert "daily" in p


def test_rewrite_prompt_contains_grounding_clauses():
    p = REWRITE_SYSTEM_PROMPT
    assert "ONLY" in p or "only" in p
    assert "never invent" in p.lower() or "do not invent" in p.lower()
    assert "src:" in p  # anchor format we extract in post-process


def test_clean_prompt_is_minimal_and_preserves_meaning():
    p = CLEAN_SYSTEM_PROMPT
    assert "preserve" in p.lower()
    assert "filler" in p.lower() or "grammar" in p.lower()
    assert len(p) < 500  # stays lightweight
