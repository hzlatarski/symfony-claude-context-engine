"""Unit tests for whisper.enhance.enhance_clean.

Tests the clean enhancement mode which uses the Haiku LLM to perform
grammar cleanup on the transcript without any retrieval.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from whisper.types import EnhanceResult


@pytest.fixture
def mock_anthropic_client():
    """Fixture that mocks the Anthropic client for testing."""
    with patch("anthropic.Anthropic") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        # Setup the response structure
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "cleaned transcript"
        mock_instance.messages.create.return_value = mock_response

        yield mock_instance


def test_enhance_clean_returns_enhance_result(mock_anthropic_client):
    """Return type is EnhanceResult."""
    from whisper.enhance import enhance_clean

    result = enhance_clean(transcript="hello world")

    assert isinstance(result, EnhanceResult)


def test_enhance_clean_mode_is_clean(mock_anthropic_client):
    """Result mode is always 'clean'."""
    from whisper.enhance import enhance_clean

    result = enhance_clean(transcript="hi")

    assert result.mode == "clean"


def test_enhance_clean_calls_anthropic_client(mock_anthropic_client):
    """Calls anthropic.Anthropic() and messages.create()."""
    from whisper.enhance import enhance_clean

    enhance_clean(transcript="hello world")

    mock_anthropic_client.messages.create.assert_called_once()


def test_enhance_clean_uses_correct_model(mock_anthropic_client):
    """Uses config.MODEL_CLEAN for the model parameter."""
    from whisper.enhance import enhance_clean
    from config import MODEL_CLEAN

    enhance_clean(transcript="hello world")

    call_kwargs = mock_anthropic_client.messages.create.call_args[1]
    assert call_kwargs["model"] == MODEL_CLEAN


def test_enhance_clean_passes_transcript_as_user_message(mock_anthropic_client):
    """Passes transcript as the user message content."""
    from whisper.enhance import enhance_clean

    enhance_clean(transcript="hello world")

    call_kwargs = mock_anthropic_client.messages.create.call_args[1]
    messages = call_kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "hello world"


def test_enhance_clean_uses_clean_system_prompt(mock_anthropic_client):
    """Uses CLEAN_SYSTEM_PROMPT as system prompt."""
    from whisper.enhance import enhance_clean
    from whisper.prompts import CLEAN_SYSTEM_PROMPT

    enhance_clean(transcript="hello world")

    call_kwargs = mock_anthropic_client.messages.create.call_args[1]
    assert call_kwargs["system"] == CLEAN_SYSTEM_PROMPT


def test_enhance_clean_sets_max_tokens_1024(mock_anthropic_client):
    """Sets max_tokens to 1024."""
    from whisper.enhance import enhance_clean

    enhance_clean(transcript="hello world")

    call_kwargs = mock_anthropic_client.messages.create.call_args[1]
    assert call_kwargs["max_tokens"] == 1024


def test_enhance_clean_enhanced_prompt_from_llm_response(mock_anthropic_client):
    """enhanced_prompt is the LLM response text."""
    from whisper.enhance import enhance_clean

    mock_anthropic_client.messages.create.return_value.content[0].text = "cleaned text from LLM"
    result = enhance_clean(transcript="hello world")

    assert result.enhanced_prompt == "cleaned text from LLM"


def test_enhance_clean_transcript_field_equals_input(mock_anthropic_client):
    """transcript field in result equals the input transcript."""
    from whisper.enhance import enhance_clean

    mock_anthropic_client.messages.create.return_value.content[0].text = "llm output"
    result = enhance_clean(transcript="hello world")

    assert result.transcript == "hello world"


def test_enhance_clean_citations_empty_list(mock_anthropic_client):
    """citations list is always empty."""
    from whisper.enhance import enhance_clean

    result = enhance_clean(transcript="hi")

    assert result.citations == []


def test_enhance_clean_warnings_empty_list(mock_anthropic_client):
    """warnings list is always empty."""
    from whisper.enhance import enhance_clean

    result = enhance_clean(transcript="hi")

    assert result.warnings == []


def test_enhance_clean_timings_includes_llm_ms(mock_anthropic_client):
    """timings_ms dict must include 'llm_ms' key."""
    from whisper.enhance import enhance_clean

    result = enhance_clean(transcript="hi")

    assert "llm_ms" in result.timings_ms
    assert isinstance(result.timings_ms["llm_ms"], int)


def test_enhance_clean_timings_llm_ms_positive(mock_anthropic_client):
    """llm_ms should be a positive integer."""
    from whisper.enhance import enhance_clean

    result = enhance_clean(transcript="hi")

    assert result.timings_ms["llm_ms"] >= 0


def test_enhance_clean_default_intent_is_generic(mock_anthropic_client):
    """Default intent parameter is 'generic'."""
    from whisper.enhance import enhance_clean

    result = enhance_clean(transcript="hi")

    assert result.intent == "generic"


def test_enhance_clean_accepts_custom_intent(mock_anthropic_client):
    """Can pass custom intent."""
    from whisper.enhance import enhance_clean

    result = enhance_clean(transcript="hi", intent="audit")

    assert result.intent == "audit"


def test_enhance_clean_default_scope_used_is_empty_list(mock_anthropic_client):
    """Default scope_used is None, returned as empty list."""
    from whisper.enhance import enhance_clean

    result = enhance_clean(transcript="hi")

    assert result.scope_used == []


def test_enhance_clean_accepts_scope_used(mock_anthropic_client):
    """Can pass scope_used list."""
    from whisper.enhance import enhance_clean

    result = enhance_clean(
        transcript="hi",
        scope_used=["articles", "code"],
    )

    assert result.scope_used == ["articles", "code"]


def test_enhance_clean_accepts_none_scope_used(mock_anthropic_client):
    """Passing None for scope_used becomes empty list."""
    from whisper.enhance import enhance_clean

    result = enhance_clean(transcript="hi", scope_used=None)

    assert result.scope_used == []


def test_enhance_clean_default_queries_used_is_empty_list(mock_anthropic_client):
    """Default queries_used is None, returned as empty list."""
    from whisper.enhance import enhance_clean

    result = enhance_clean(transcript="hi")

    assert result.queries_used == []


def test_enhance_clean_accepts_queries_used(mock_anthropic_client):
    """Can pass queries_used list."""
    from whisper.enhance import enhance_clean

    result = enhance_clean(
        transcript="hi",
        queries_used=["q1", "q2"],
    )

    assert result.queries_used == ["q1", "q2"]


def test_enhance_clean_accepts_none_queries_used(mock_anthropic_client):
    """Passing None for queries_used becomes empty list."""
    from whisper.enhance import enhance_clean

    result = enhance_clean(transcript="hi", queries_used=None)

    assert result.queries_used == []


def test_enhance_clean_with_all_parameters(mock_anthropic_client):
    """Test with all parameters provided."""
    from whisper.enhance import enhance_clean

    mock_anthropic_client.messages.create.return_value.content[0].text = "cleaned text"
    result = enhance_clean(
        transcript="original transcript with errors um like you know",
        intent="explain",
        scope_used=["articles"],
        queries_used=["how does X work"],
    )

    assert result.transcript == "original transcript with errors um like you know"
    assert result.enhanced_prompt == "cleaned text"
    assert result.mode == "clean"
    assert result.intent == "explain"
    assert result.scope_used == ["articles"]
    assert result.queries_used == ["how does X work"]
    assert result.citations == []
    assert result.warnings == []
    assert "llm_ms" in result.timings_ms


def test_enhance_clean_preserves_transcript_exactly(mock_anthropic_client):
    """Transcript field preserves input exactly (not LLM output)."""
    from whisper.enhance import enhance_clean

    transcript_input = "this is my exact transcript with\nmultiple lines"
    mock_anthropic_client.messages.create.return_value.content[0].text = "different text from llm"
    result = enhance_clean(transcript=transcript_input)

    assert result.transcript == transcript_input
    assert result.enhanced_prompt == "different text from llm"
