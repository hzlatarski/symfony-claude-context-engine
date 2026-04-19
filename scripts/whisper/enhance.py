"""Enhance pipeline — transform transcript + hits into enhanced prompts.

This module provides the enhance_verbatim function which performs zero-LLM
enhancement: appends a formatted context block to the transcript, and
enhance_clean which uses Haiku to grammar-clean transcripts.
"""
from __future__ import annotations

import functools
import time
import anthropic

from config import MODEL_CLEAN
from whisper.types import EnhanceResult, Hit
from whisper.prompts import CLEAN_SYSTEM_PROMPT


class EnhanceError(Exception):
    """Raised when text extraction from LLM response fails."""
    pass


@functools.lru_cache(maxsize=1)
def _get_client() -> anthropic.Anthropic:
    """Get or create a cached Anthropic client."""
    return anthropic.Anthropic()


def _extract_text(resp) -> str:
    """Extract text from response content.

    Args:
        resp: Response object with content list.

    Returns:
        The text content from all text blocks joined together.

    Raises:
        EnhanceError: If no text blocks are found in response.
    """
    texts = [block.text for block in resp.content if getattr(block, "type", None) == "text"]
    if not texts:
        raise EnhanceError("No text blocks found in LLM response")
    return "".join(texts).strip()


def enhance_verbatim(
    transcript: str,
    hits: list[Hit],
    intent: str = "generic",
    scope_used: list[str] | None = None,
    queries_used: list[str] | None = None,
) -> EnhanceResult:
    """Verbatim enhancement mode: append context block, no LLM call.

    Takes a transcript and retrieval hits and builds an enhanced prompt by
    appending a formatted context block listing each hit.

    Args:
        transcript: The original transcript text to enhance.
        hits: List of retrieval hits to include in context block.
        intent: The user's intent (default "generic").
        scope_used: List of scopes used for retrieval (default None -> []).
        queries_used: List of queries used for retrieval (default None -> []).

    Returns:
        EnhanceResult with mode="verbatim", warnings=[].
    """
    start_time_ms = int(time.perf_counter() * 1000)

    # Build the enhanced prompt
    enhanced_prompt = transcript

    # If there are hits, append a formatted context block
    if hits:
        context_parts = ["\n\n## Retrieved Context\n"]
        for hit in hits:
            context_parts.append(f"\n[{hit.id}] {hit.path} — {hit.title}\n> {hit.snippet}")
        enhanced_prompt = transcript + "".join(context_parts)

    # Handle defaults for optional list parameters
    scope_used_list = scope_used if scope_used is not None else []
    queries_used_list = queries_used if queries_used is not None else []

    # Calculate timing
    end_time_ms = int(time.perf_counter() * 1000)
    enhance_ms = end_time_ms - start_time_ms

    return EnhanceResult(
        transcript=transcript,
        enhanced_prompt=enhanced_prompt,
        mode="verbatim",
        citations=hits,
        intent=intent,
        scope_used=scope_used_list,
        queries_used=queries_used_list,
        warnings=[],
        timings_ms={"enhance_ms": enhance_ms},
    )


def enhance_clean(transcript: str) -> str:
    """Clean mode enhancement: call Haiku to grammar-clean transcript.

    Takes a transcript and uses Claude Haiku with CLEAN_SYSTEM_PROMPT
    to perform grammar cleanup, removing filler words and fixing
    punctuation while preserving meaning and voice.

    Args:
        transcript: The original transcript text to clean.

    Returns:
        Cleaned transcript string with stripped whitespace.

    Raises:
        EnhanceError: If response contains no text blocks.
    """
    if not transcript.strip():
        return transcript

    client = _get_client()
    response = client.messages.create(
        model=MODEL_CLEAN,
        max_tokens=2048,
        system=CLEAN_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": transcript}
        ],
    )

    # Extract cleaned text from response and strip whitespace
    cleaned_text = _extract_text(response)
    return cleaned_text.strip()
