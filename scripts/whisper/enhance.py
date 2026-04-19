"""Enhance pipeline — transform transcript + hits into enhanced prompts.

This module provides the enhance_verbatim function which performs zero-LLM
enhancement: appends a formatted context block to the transcript.
"""
from __future__ import annotations

import time

from whisper.types import EnhanceResult, Hit


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
