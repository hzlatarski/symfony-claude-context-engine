"""Shape tests for whisper module dataclasses."""
from __future__ import annotations

from whisper.types import EnhanceResult, Hit


def test_hit_accepts_all_required_fields():
    hit = Hit(
        id="c1",
        source="article",
        category="implementation-plans",
        path="docs/superpowers/plans/foo.md",
        title="Foo",
        snippet="snippet",
        full_body=None,
        score=0.87,
        symbols=[],
        metadata={"confidence": 0.9},
    )
    assert hit.id == "c1"
    assert hit.source == "article"
    assert hit.category == "implementation-plans"


def test_hit_code_source_has_symbols():
    hit = Hit(
        id="c2",
        source="code",
        category=None,
        path="src/Foo.php:1-42",
        title="Foo",
        snippet="class Foo",
        full_body=None,
        score=0.5,
        symbols=["Foo", "bar"],
        metadata={},
    )
    assert hit.source == "code"
    assert hit.category is None
    assert hit.symbols == ["Foo", "bar"]


def test_enhance_result_shape():
    result = EnhanceResult(
        transcript="hello",
        enhanced_prompt="HELLO",
        mode="rewrite",
        citations=[],
        intent="audit",
        scope_used=["articles"],
        queries_used=["q1"],
        warnings=[],
        timings_ms={"total": 1000},
    )
    assert result.mode == "rewrite"
    assert result.timings_ms["total"] == 1000
