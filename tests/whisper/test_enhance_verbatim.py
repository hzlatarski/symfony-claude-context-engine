"""Unit tests for whisper.enhance.enhance_verbatim.

Tests the verbatim enhancement mode which appends a formatted context block
to the transcript without any LLM call.
"""
from __future__ import annotations

import pytest

from whisper.types import Hit, EnhanceResult


def _make_hit(
    id_: str,
    source: str = "article",
    path: str = "concepts/foo",
    title: str = "Foo",
    snippet: str = "snippet text",
) -> Hit:
    """Helper to construct a Hit with sensible defaults."""
    return Hit(
        id=id_,
        source=source,
        category="captured-memory" if source == "article" else None,
        path=path,
        title=title,
        snippet=snippet,
        full_body=None,
        score=0.85,
        symbols=[],
        metadata={},
    )


def test_enhance_verbatim_no_hits_returns_transcript_only():
    """With no hits, enhanced_prompt is transcript unchanged."""
    from whisper.enhance import enhance_verbatim

    result = enhance_verbatim(
        transcript="hello world",
        hits=[],
    )

    assert result.enhanced_prompt == "hello world"
    assert result.transcript == "hello world"


def test_enhance_verbatim_with_one_hit_appends_context_block():
    """With one hit, enhanced_prompt is transcript + context block."""
    from whisper.enhance import enhance_verbatim

    hit = _make_hit(id_="c1", path="docs/foo", title="Foo Title", snippet="snippet content")
    result = enhance_verbatim(
        transcript="hello world",
        hits=[hit],
    )

    assert result.transcript == "hello world"
    assert "hello world" in result.enhanced_prompt
    assert "[c1]" in result.enhanced_prompt
    assert "docs/foo" in result.enhanced_prompt
    assert "Foo Title" in result.enhanced_prompt
    assert "snippet content" in result.enhanced_prompt


def test_enhance_verbatim_formats_context_block_correctly():
    """Context block format: [id] path — Title\\n> snippet."""
    from whisper.enhance import enhance_verbatim

    hit = _make_hit(
        id_="c1",
        path="concepts/authentication",
        title="Auth Strategies",
        snippet="multi-factor authentication is recommended",
    )
    result = enhance_verbatim(
        transcript="how to secure auth",
        hits=[hit],
    )

    assert "## Retrieved Context" in result.enhanced_prompt
    assert "[c1] concepts/authentication — Auth Strategies" in result.enhanced_prompt
    assert "> multi-factor authentication is recommended" in result.enhanced_prompt


def test_enhance_verbatim_multiple_hits_each_get_context_line():
    """With multiple hits, each gets its own [id] path — Title line."""
    from whisper.enhance import enhance_verbatim

    hits = [
        _make_hit(id_="c1", path="path/one", title="Title One", snippet="snippet 1"),
        _make_hit(id_="c2", path="path/two", title="Title Two", snippet="snippet 2"),
        _make_hit(id_="c3", path="path/three", title="Title Three", snippet="snippet 3"),
    ]
    result = enhance_verbatim(
        transcript="original transcript",
        hits=hits,
    )

    assert "[c1]" in result.enhanced_prompt
    assert "[c2]" in result.enhanced_prompt
    assert "[c3]" in result.enhanced_prompt
    assert "Title One" in result.enhanced_prompt
    assert "Title Two" in result.enhanced_prompt
    assert "Title Three" in result.enhanced_prompt


def test_enhance_verbatim_mode_is_verbatim():
    """Result mode is always 'verbatim'."""
    from whisper.enhance import enhance_verbatim

    hits = [_make_hit(id_="c1")]
    result = enhance_verbatim(transcript="hi", hits=hits)

    assert result.mode == "verbatim"


def test_enhance_verbatim_citations_equals_input_hits():
    """Result citations list is exactly the hits passed in."""
    from whisper.enhance import enhance_verbatim

    hits = [
        _make_hit(id_="c1", path="a"),
        _make_hit(id_="c2", path="b"),
    ]
    result = enhance_verbatim(
        transcript="text",
        hits=hits,
    )

    assert result.citations == hits
    assert len(result.citations) == 2


def test_enhance_verbatim_default_intent_is_generic():
    """Default intent parameter is 'generic'."""
    from whisper.enhance import enhance_verbatim

    result = enhance_verbatim(transcript="hi", hits=[])

    assert result.intent == "generic"


def test_enhance_verbatim_accepts_custom_intent():
    """Can pass custom intent."""
    from whisper.enhance import enhance_verbatim

    result = enhance_verbatim(
        transcript="hi",
        hits=[],
        intent="audit",
    )

    assert result.intent == "audit"


def test_enhance_verbatim_default_scope_used_is_none():
    """Default scope_used is None, returned as empty list."""
    from whisper.enhance import enhance_verbatim

    result = enhance_verbatim(transcript="hi", hits=[])

    assert result.scope_used == []


def test_enhance_verbatim_accepts_scope_used():
    """Can pass scope_used list."""
    from whisper.enhance import enhance_verbatim

    result = enhance_verbatim(
        transcript="hi",
        hits=[],
        scope_used=["articles", "code"],
    )

    assert result.scope_used == ["articles", "code"]


def test_enhance_verbatim_default_queries_used_is_none():
    """Default queries_used is None, returned as empty list."""
    from whisper.enhance import enhance_verbatim

    result = enhance_verbatim(transcript="hi", hits=[])

    assert result.queries_used == []


def test_enhance_verbatim_accepts_queries_used():
    """Can pass queries_used list."""
    from whisper.enhance import enhance_verbatim

    result = enhance_verbatim(
        transcript="hi",
        hits=[],
        queries_used=["q1", "q2"],
    )

    assert result.queries_used == ["q1", "q2"]


def test_enhance_verbatim_warnings_is_empty_list():
    """Warnings list is always empty for verbatim mode."""
    from whisper.enhance import enhance_verbatim

    result = enhance_verbatim(transcript="hi", hits=[])

    assert result.warnings == []


def test_enhance_verbatim_timings_includes_enhance_ms():
    """timings_ms dict must include 'enhance_ms' key."""
    from whisper.enhance import enhance_verbatim

    result = enhance_verbatim(transcript="hi", hits=[])

    assert "enhance_ms" in result.timings_ms
    assert isinstance(result.timings_ms["enhance_ms"], int)


def test_enhance_verbatim_timings_enhance_ms_is_positive():
    """enhance_ms should be a positive integer."""
    from whisper.enhance import enhance_verbatim

    result = enhance_verbatim(transcript="hi", hits=[])

    assert result.timings_ms["enhance_ms"] >= 0


def test_enhance_verbatim_timings_is_dict():
    """timings_ms is a dict."""
    from whisper.enhance import enhance_verbatim

    result = enhance_verbatim(transcript="hi", hits=[])

    assert isinstance(result.timings_ms, dict)


def test_enhance_verbatim_returns_enhance_result():
    """Return type is EnhanceResult."""
    from whisper.enhance import enhance_verbatim

    result = enhance_verbatim(transcript="hi", hits=[])

    assert isinstance(result, EnhanceResult)


def test_enhance_verbatim_preserves_transcript_exactly():
    """Transcript in result matches input exactly."""
    from whisper.enhance import enhance_verbatim

    transcript_input = "this is my exact transcript with\nmultiple lines"
    result = enhance_verbatim(transcript=transcript_input, hits=[])

    assert result.transcript == transcript_input


def test_enhance_verbatim_enhanced_prompt_starts_with_transcript():
    """enhanced_prompt begins with the transcript when hits present."""
    from whisper.enhance import enhance_verbatim

    transcript = "hello world"
    hits = [_make_hit(id_="c1")]
    result = enhance_verbatim(transcript=transcript, hits=hits)

    assert result.enhanced_prompt.startswith(transcript)


def test_enhance_verbatim_context_block_after_transcript():
    """Context block appears after the transcript."""
    from whisper.enhance import enhance_verbatim

    transcript = "hello world"
    hits = [_make_hit(id_="c1")]
    result = enhance_verbatim(transcript=transcript, hits=hits)

    # Find where the context starts
    transcript_end = len(transcript)
    enhanced_from_start = result.enhanced_prompt[:transcript_end]
    assert enhanced_from_start == transcript

    # Check that the context block follows
    rest = result.enhanced_prompt[transcript_end:]
    assert "[c1]" in rest


def test_enhance_verbatim_many_hits_with_complex_content():
    """Test with realistic multi-hit scenario including special chars."""
    from whisper.enhance import enhance_verbatim

    hits = [
        _make_hit(
            id_="c1",
            path="docs/superpowers/plans/2026-04-18-codebase-search-mcp.md",
            title="Codebase Search MCP",
            snippet="Implement semantic search using BM25 + embeddings",
        ),
        _make_hit(
            id_="c2",
            source="code",
            path="src/Service/HybridSearch.php:42-87",
            title="HybridSearch Service",
            snippet="public function search($query, $limit = 10)",
        ),
        _make_hit(
            id_="c3",
            source="daily",
            path="daily/2026-04-19",
            title="Today's Log",
            snippet="Reviewed MCP implementation & performance metrics",
        ),
    ]
    result = enhance_verbatim(
        transcript="how do we do hybrid search?",
        hits=hits,
        intent="audit",
        scope_used=["articles", "code", "daily"],
        queries_used=["hybrid search", "semantic search"],
    )

    assert result.mode == "verbatim"
    assert result.intent == "audit"
    assert result.scope_used == ["articles", "code", "daily"]
    assert result.queries_used == ["hybrid search", "semantic search"]
    assert len(result.citations) == 3
    assert all(h.id.startswith("c") for h in result.citations)
    assert "[c1]" in result.enhanced_prompt
    assert "[c2]" in result.enhanced_prompt
    assert "[c3]" in result.enhanced_prompt
