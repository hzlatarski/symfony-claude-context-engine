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
    """Load-bearing invariants for the grounded-rewrite guarantee.

    Each assertion ties to a concrete downstream behavior:
    - 'never invent' is the phrasing that stops Sonnet from fabricating paths
    - 'verbatim' / 'exact path' is what makes post-process anchor verification work
    - '[src:' is the anchor format the regex in enhance.verify_anchors greps for
    - the mandatory-syntax phrase locks down the anchor convention against variants
    """
    p = REWRITE_SYSTEM_PROMPT
    lower = p.lower()

    # Anchor format must be declared AND marked mandatory
    assert "[src:" in p
    assert "must be wrapped" in lower or "exact syntax is required" in lower

    # Grounding clause
    assert "never invent" in lower
    assert "verbatim" in lower or "exact path" in lower

    # Missing-context escape hatch so Sonnet doesn't fabricate under pressure
    assert "no specific" in lower or "missing context" in lower or "describe it abstractly" in lower

    # Scope preservation — rule 4
    assert "not add goals" in lower or "not expanding scope" in lower


def test_clean_prompt_is_minimal_and_preserves_meaning():
    p = CLEAN_SYSTEM_PROMPT
    assert "preserve" in p.lower()
    assert "filler" in p.lower() or "grammar" in p.lower()
    assert len(p) < 500  # stays lightweight
