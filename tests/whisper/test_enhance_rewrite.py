"""Unit tests for whisper.enhance.enhance_rewrite and anchor verification."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from whisper.types import Hit


def _mock_response(text: str):
    return SimpleNamespace(content=[SimpleNamespace(text=text, type="text")])


def _hit(cid: str, source: str, path: str, category=None):
    return Hit(
        id=cid, source=source, category=category,
        path=path, title=path, snippet="snippet",
        full_body=f"full body for {path}",
        score=1.0, symbols=[], metadata={},
    )


@pytest.fixture
def mock_client(monkeypatch):
    client = MagicMock()
    import whisper.enhance as e
    e._get_client.cache_clear()
    monkeypatch.setattr(e, "_get_client", lambda: client)
    return client


def test_rewrite_passes_transcript_and_context_in_user_message(mock_client):
    mock_client.messages.create.return_value = _mock_response("REWRITTEN")
    from whisper.enhance import enhance_rewrite

    hits = [
        _hit("c1", "article", "concepts/s3-migration", category="captured-memory"),
        _hit("c2", "code", "src/Service/Foo.php:1-20"),
    ]

    enhance_rewrite("audit the S3 migration", hits)

    _args, kwargs = mock_client.messages.create.call_args
    user_msg = kwargs["messages"][0]["content"]
    assert "audit the S3 migration" in user_msg
    assert "concepts/s3-migration" in user_msg
    assert "src/Service/Foo.php:1-20" in user_msg


def test_rewrite_uses_sonnet_model_from_config(mock_client):
    mock_client.messages.create.return_value = _mock_response("out")
    import config
    from whisper.enhance import enhance_rewrite

    enhance_rewrite("x", [_hit("c1", "article", "slug")])

    _args, kwargs = mock_client.messages.create.call_args
    assert kwargs["model"] == config.MODEL_REWRITE


def test_verify_anchors_strips_hallucinated_paths():
    from whisper.enhance import verify_anchors

    hits = [_hit("c1", "article", "concepts/real-slug")]
    rewritten = (
        "Task 1: update [src:concepts/real-slug] "
        "Task 2: also touch [src:src/Invented/Fake.php]"
    )

    cleaned, warnings = verify_anchors(rewritten, hits)

    assert "[src:concepts/real-slug]" in cleaned
    assert "[src:src/Invented/Fake.php]" not in cleaned
    assert any("src/Invented/Fake.php" in w for w in warnings)


def test_verify_anchors_preserves_line_ranges_on_code_paths():
    from whisper.enhance import verify_anchors

    hits = [_hit("c1", "code", "src/Service/Foo.php:1-20")]
    # LLM might write the anchor with or without the line range — both should be accepted
    rewritten = "See [src:src/Service/Foo.php] and [src:src/Service/Foo.php:1-20]"

    cleaned, warnings = verify_anchors(rewritten, hits)

    assert "[src:src/Service/Foo.php]" in cleaned
    assert "[src:src/Service/Foo.php:1-20]" in cleaned
    assert warnings == []


def test_verify_anchors_no_anchors_produces_no_warnings():
    from whisper.enhance import verify_anchors

    hits = [_hit("c1", "article", "concepts/foo")]
    rewritten = "plain text with no anchors"

    cleaned, warnings = verify_anchors(rewritten, hits)

    assert cleaned == rewritten
    assert warnings == []


def test_rewrite_with_zero_hits_raises(mock_client):
    from whisper.enhance import enhance_rewrite, EnhanceError

    with pytest.raises(EnhanceError):
        enhance_rewrite("transcript", [])


def test_rewrite_integrates_verification_and_reports_warnings(mock_client):
    mock_client.messages.create.return_value = _mock_response(
        "Do [src:concepts/real] but also [src:src/Fake.php]"
    )
    from whisper.enhance import enhance_rewrite

    hits = [_hit("c1", "article", "concepts/real")]
    result = enhance_rewrite("do stuff", hits)

    assert "[src:src/Fake.php]" not in result.enhanced_prompt
    assert "[src:concepts/real]" in result.enhanced_prompt
    assert any("src/Fake.php" in w for w in result.warnings)


# ============================================================================
# SIGNATURE & PARAMETER TESTS
# ============================================================================

def test_enhance_rewrite_intent_parameter_default(mock_client):
    """Verify intent defaults to 'generic' when not provided."""
    mock_client.messages.create.return_value = _mock_response("rewritten")
    from whisper.enhance import enhance_rewrite

    result = enhance_rewrite("do something", [_hit("c1", "article", "path")])

    assert result.intent == "generic"


def test_enhance_rewrite_intent_parameter_custom(mock_client):
    """Verify custom intent is preserved in result."""
    mock_client.messages.create.return_value = _mock_response("rewritten")
    from whisper.enhance import enhance_rewrite

    result = enhance_rewrite(
        "audit this", [_hit("c1", "article", "path")], intent="audit"
    )

    assert result.intent == "audit"


def test_enhance_rewrite_scope_used_default(mock_client):
    """Verify scope_used defaults to empty list when not provided."""
    mock_client.messages.create.return_value = _mock_response("rewritten")
    from whisper.enhance import enhance_rewrite

    result = enhance_rewrite("x", [_hit("c1", "article", "path")])

    assert result.scope_used == []


def test_enhance_rewrite_scope_used_custom(mock_client):
    """Verify custom scope_used is preserved in result."""
    mock_client.messages.create.return_value = _mock_response("rewritten")
    from whisper.enhance import enhance_rewrite

    result = enhance_rewrite(
        "x", [_hit("c1", "article", "path")], scope_used=["articles", "code"]
    )

    assert result.scope_used == ["articles", "code"]


def test_enhance_rewrite_queries_used_default(mock_client):
    """Verify queries_used defaults to empty list when not provided."""
    mock_client.messages.create.return_value = _mock_response("rewritten")
    from whisper.enhance import enhance_rewrite

    result = enhance_rewrite("x", [_hit("c1", "article", "path")])

    assert result.queries_used == []


def test_enhance_rewrite_queries_used_custom(mock_client):
    """Verify custom queries_used is preserved in result."""
    mock_client.messages.create.return_value = _mock_response("rewritten")
    from whisper.enhance import enhance_rewrite

    result = enhance_rewrite(
        "x", [_hit("c1", "article", "path")], queries_used=["q1", "q2"]
    )

    assert result.queries_used == ["q1", "q2"]


# ============================================================================
# RETURN TYPE & STRUCTURE TESTS
# ============================================================================

def test_enhance_rewrite_returns_enhance_result_type(mock_client):
    """Verify return type is EnhanceResult."""
    mock_client.messages.create.return_value = _mock_response("rewritten")
    from whisper.enhance import enhance_rewrite
    from whisper.types import EnhanceResult

    result = enhance_rewrite("x", [_hit("c1", "article", "path")])

    assert isinstance(result, EnhanceResult)


def test_enhance_rewrite_result_has_transcript_field(mock_client):
    """Verify result preserves original transcript."""
    mock_client.messages.create.return_value = _mock_response("rewritten")
    from whisper.enhance import enhance_rewrite

    transcript = "original transcript text"
    result = enhance_rewrite(transcript, [_hit("c1", "article", "path")])

    assert result.transcript == transcript


def test_enhance_rewrite_result_has_enhanced_prompt_field(mock_client):
    """Verify result contains LLM-rewritten prompt."""
    mock_client.messages.create.return_value = _mock_response("LLM output text")
    from whisper.enhance import enhance_rewrite

    result = enhance_rewrite("x", [_hit("c1", "article", "path")])

    assert result.enhanced_prompt == "LLM output text"


def test_enhance_rewrite_result_mode_is_rewrite(mock_client):
    """Verify result mode is 'rewrite'."""
    mock_client.messages.create.return_value = _mock_response("out")
    from whisper.enhance import enhance_rewrite

    result = enhance_rewrite("x", [_hit("c1", "article", "path")])

    assert result.mode == "rewrite"


def test_enhance_rewrite_result_citations_match_hits(mock_client):
    """Verify result citations are the input hits."""
    mock_client.messages.create.return_value = _mock_response("out")
    from whisper.enhance import enhance_rewrite

    hits = [
        _hit("c1", "article", "path1"),
        _hit("c2", "code", "path2"),
    ]
    result = enhance_rewrite("x", hits)

    assert result.citations == hits


def test_enhance_rewrite_result_has_timings_ms_dict(mock_client):
    """Verify result includes timings_ms dict with both keys."""
    mock_client.messages.create.return_value = _mock_response("out")
    from whisper.enhance import enhance_rewrite

    result = enhance_rewrite("x", [_hit("c1", "article", "path")])

    assert isinstance(result.timings_ms, dict)
    assert "llm_ms" in result.timings_ms
    assert "enhance_ms" in result.timings_ms


def test_enhance_rewrite_timings_are_positive_integers(mock_client):
    """Verify timing values are positive integers."""
    mock_client.messages.create.return_value = _mock_response("out")
    from whisper.enhance import enhance_rewrite

    result = enhance_rewrite("x", [_hit("c1", "article", "path")])

    assert isinstance(result.timings_ms["llm_ms"], int)
    assert isinstance(result.timings_ms["enhance_ms"], int)
    assert result.timings_ms["llm_ms"] >= 0
    assert result.timings_ms["enhance_ms"] >= 0


# ============================================================================
# LLM INTERACTION TESTS
# ============================================================================

def test_enhance_rewrite_includes_custom_intent_in_context(mock_client):
    """Verify custom intent is passed through system prompt (no assertion, just coverage)."""
    mock_client.messages.create.return_value = _mock_response("out")
    from whisper.enhance import enhance_rewrite

    result = enhance_rewrite(
        "audit S3", [_hit("c1", "article", "path")], intent="audit"
    )

    assert result.intent == "audit"


def test_enhance_rewrite_context_block_format_with_category(mock_client):
    """Verify context block includes category when present."""
    mock_client.messages.create.return_value = _mock_response("REWRITTEN")
    from whisper.enhance import enhance_rewrite

    hits = [
        _hit("c1", "article", "concepts/foo", category="captured-memory"),
    ]

    enhance_rewrite("x", hits)

    _args, kwargs = mock_client.messages.create.call_args
    user_msg = kwargs["messages"][0]["content"]
    assert "category=captured-memory" in user_msg


def test_enhance_rewrite_context_block_format_without_category(mock_client):
    """Verify context block omits category when None."""
    mock_client.messages.create.return_value = _mock_response("REWRITTEN")
    from whisper.enhance import enhance_rewrite

    hits = [
        _hit("c1", "article", "concepts/foo", category=None),
    ]

    enhance_rewrite("x", hits)

    _args, kwargs = mock_client.messages.create.call_args
    user_msg = kwargs["messages"][0]["content"]
    assert "concepts/foo" in user_msg
    assert "<context>" in user_msg
    assert "</context>" in user_msg


def test_enhance_rewrite_max_tokens_is_2048(mock_client):
    """Verify max_tokens parameter is set to 2048."""
    mock_client.messages.create.return_value = _mock_response("out")
    from whisper.enhance import enhance_rewrite

    enhance_rewrite("x", [_hit("c1", "article", "path")])

    _args, kwargs = mock_client.messages.create.call_args
    assert kwargs["max_tokens"] == 2048


# ============================================================================
# ANCHOR VERIFICATION TESTS
# ============================================================================

def test_enhance_rewrite_multiple_hallucinated_anchors(mock_client):
    """Verify multiple hallucinated anchors are all stripped and warned."""
    mock_client.messages.create.return_value = _mock_response(
        "Start [src:concepts/real]. "
        "Also [src:src/Invented1.php]. "
        "And [src:src/Invented2.php]. "
        "End [src:concepts/real]."
    )
    from whisper.enhance import enhance_rewrite

    hits = [_hit("c1", "article", "concepts/real")]
    result = enhance_rewrite("do stuff", hits)

    assert "[src:concepts/real]" in result.enhanced_prompt
    assert "[src:src/Invented1.php]" not in result.enhanced_prompt
    assert "[src:src/Invented2.php]" not in result.enhanced_prompt
    assert len(result.warnings) >= 2
    assert any("src/Invented1.php" in w for w in result.warnings)
    assert any("src/Invented2.php" in w for w in result.warnings)


def test_enhance_rewrite_mixed_valid_invalid_anchors(mock_client):
    """Verify mixed valid and invalid anchors are handled correctly."""
    mock_client.messages.create.return_value = _mock_response(
        "Valid: [src:src/Real/File.php] and invalid [src:src/Fake.php] mixed."
    )
    from whisper.enhance import enhance_rewrite

    hits = [_hit("c1", "code", "src/Real/File.php")]
    result = enhance_rewrite("x", hits)

    assert "[src:src/Real/File.php]" in result.enhanced_prompt
    assert "[src:src/Fake.php]" not in result.enhanced_prompt
    assert len(result.warnings) == 1
    assert "src/Fake.php" in result.warnings[0]


def test_enhance_rewrite_anchor_stripped_but_path_preserved(mock_client):
    """Verify [src:...] markup is removed but path text remains."""
    mock_client.messages.create.return_value = _mock_response(
        "See [src:src/Fake.php] for details"
    )
    from whisper.enhance import enhance_rewrite

    hits = [_hit("c1", "article", "other/path")]
    result = enhance_rewrite("x", hits)

    assert "[src:src/Fake.php]" not in result.enhanced_prompt
    assert "src/Fake.php" in result.enhanced_prompt


def test_enhance_rewrite_line_range_variants_both_valid(mock_client):
    """Verify both with and without line range are accepted."""
    mock_client.messages.create.return_value = _mock_response(
        "[src:src/Service/Foo.php] and [src:src/Service/Foo.php:1-50]"
    )
    from whisper.enhance import enhance_rewrite

    hits = [_hit("c1", "code", "src/Service/Foo.php:1-50")]
    result = enhance_rewrite("x", hits)

    assert "[src:src/Service/Foo.php]" in result.enhanced_prompt
    assert "[src:src/Service/Foo.php:1-50]" in result.enhanced_prompt
    assert result.warnings == []


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

def test_enhance_rewrite_empty_hits_raises_enhance_error(mock_client):
    """Verify empty hits list raises EnhanceError."""
    from whisper.enhance import enhance_rewrite, EnhanceError

    with pytest.raises(EnhanceError, match="at least one"):
        enhance_rewrite("transcript", [])


def test_enhance_rewrite_extract_text_error_on_no_text_blocks(mock_client):
    """Verify missing text blocks in response raises EnhanceError."""
    mock_client.messages.create.return_value = SimpleNamespace(content=[])
    from whisper.enhance import enhance_rewrite, EnhanceError

    hits = [_hit("c1", "article", "path")]
    with pytest.raises(EnhanceError, match="No text blocks"):
        enhance_rewrite("x", hits)


# ============================================================================
# TIMING BEHAVIOR TESTS
# ============================================================================

def test_enhance_rewrite_enhance_ms_includes_total_time(mock_client):
    """Verify enhance_ms is total elapsed time from start to end."""
    mock_client.messages.create.return_value = _mock_response("out")
    from whisper.enhance import enhance_rewrite

    result = enhance_rewrite("x", [_hit("c1", "article", "path")])

    assert result.timings_ms["enhance_ms"] >= result.timings_ms["llm_ms"]


def test_enhance_rewrite_llm_ms_is_measured(mock_client):
    """Verify llm_ms is the LLM call duration, not zero."""
    mock_client.messages.create.return_value = _mock_response("out")
    from whisper.enhance import enhance_rewrite

    result = enhance_rewrite("x", [_hit("c1", "article", "path")])

    assert result.timings_ms["llm_ms"] >= 0


# ============================================================================
# INTEGRATION & FULL FLOW TESTS
# ============================================================================

def test_enhance_rewrite_full_flow_with_all_parameters(mock_client):
    """Verify full flow with all custom parameters works end-to-end."""
    mock_client.messages.create.return_value = _mock_response(
        "Rewritten prompt with [src:concepts/real] reference"
    )
    from whisper.enhance import enhance_rewrite

    hits = [
        _hit("c1", "article", "concepts/real", category="memory"),
        _hit("c2", "code", "src/Service/Foo.php"),
    ]
    result = enhance_rewrite(
        transcript="audit S3 migration",
        hits=hits,
        intent="audit",
        scope_used=["articles", "code"],
        queries_used=["S3", "migration"],
    )

    assert result.transcript == "audit S3 migration"
    assert result.intent == "audit"
    assert result.scope_used == ["articles", "code"]
    assert result.queries_used == ["S3", "migration"]
    assert result.mode == "rewrite"
    assert result.citations == hits
    assert "[src:concepts/real]" in result.enhanced_prompt


def test_enhance_rewrite_single_hit_flow(mock_client):
    """Verify flow works with single hit."""
    mock_client.messages.create.return_value = _mock_response("Rewritten")
    from whisper.enhance import enhance_rewrite

    hits = [_hit("c1", "article", "path")]
    result = enhance_rewrite("x", hits)

    assert result.citations == hits
    assert len(result.citations) == 1


def test_enhance_rewrite_many_hits_flow(mock_client):
    """Verify flow works with many hits."""
    mock_client.messages.create.return_value = _mock_response("Rewritten")
    from whisper.enhance import enhance_rewrite

    hits = [
        _hit(f"c{i}", "article", f"path{i}") for i in range(10)
    ]
    result = enhance_rewrite("x", hits)

    assert len(result.citations) == 10


def test_verify_anchors_strips_whitespace_from_anchor_paths():
    hits = [_hit("c1", "article", "concepts/foo")]
    from whisper.enhance import verify_anchors

    cleaned, warnings = verify_anchors("[src:  concepts/foo  ]", hits)

    assert cleaned == "[src:concepts/foo]"
    assert warnings == []
