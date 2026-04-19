"""Enhance pipeline — transform transcript + hits into enhanced prompts.

This module provides the enhance_verbatim function which performs zero-LLM
enhancement: appends a formatted context block to the transcript, and
enhance_clean which uses Haiku to grammar-clean transcripts.
"""
from __future__ import annotations

import functools
import re
import time
import anthropic
from dataclasses import dataclass

from config import MODEL_CLEAN, MODEL_REWRITE
from whisper.types import EnhanceResult, Hit
from whisper.prompts import CLEAN_SYSTEM_PROMPT, REWRITE_SYSTEM_PROMPT


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


_ANCHOR_RE = re.compile(r"\[src:([^\]]+)\]")
_LINE_RANGE_SUFFIX_RE = re.compile(r":\d+-\d+$")


def _build_context_block(hits: list[Hit]) -> str:
    """Render retrieved hits as a <context> block for the rewrite prompt."""
    lines = ["<context>"]
    for h in hits:
        badge = {"article": "ARTICLE", "code": "CODE", "daily": "DAILY"}[h.source]
        cat = f" category={h.category}" if h.category else ""
        lines.append(f"[{h.id}] {badge}{cat} path={h.path}")
        lines.append(f"  title: {h.title}")
        body = (h.full_body or h.snippet).strip()
        lines.append(f"  body: {body}")
        lines.append("")
    lines.append("</context>")
    return "\n".join(lines)


def verify_anchors(rewritten: str, hits: list[Hit]) -> tuple[str, list[str]]:
    """Strip [src:...] anchors whose path isn't in the retrieved context.

    A path is considered valid if it matches a Hit's path exactly, OR if
    it matches a Hit's path after stripping a trailing :START-END line
    range suffix (so the LLM is allowed to drop the range for brevity).

    Args:
        rewritten: The rewritten text containing [src:...] anchors.
        hits: The list of valid Hit objects from retrieval.

    Returns:
        A tuple of (cleaned_text, warnings_list). Anchors with invalid
        paths are removed from the text, and a warning is added for each.
    """
    valid_paths: set[str] = set()
    for h in hits:
        valid_paths.add(h.path)
        # Also accept paths with line ranges stripped
        valid_paths.add(_LINE_RANGE_SUFFIX_RE.sub("", h.path))

    warnings: list[str] = []

    def replace(match: re.Match) -> str:
        raw = match.group(1).strip()
        if raw in valid_paths:
            return match.group(0)
        warnings.append(f"Removed unverifiable anchor: {raw}")
        return raw  # drop the [src:...] markup, keep the path text

    cleaned = _ANCHOR_RE.sub(replace, rewritten)
    return cleaned, warnings


def enhance_rewrite(
    transcript: str,
    hits: list[Hit],
    intent: str = "generic",
    scope_used: list[str] | None = None,
    queries_used: list[str] | None = None,
) -> EnhanceResult:
    """Grounded Sonnet rewrite of the transcript using retrieved context.

    Takes a transcript and retrieved context hits, sends them to Claude
    Sonnet via REWRITE_SYSTEM_PROMPT to produce a grounded, precise prompt
    for Claude Code. Verifies all [src:...] anchors against the retrieved
    context and strips any hallucinated paths.

    Args:
        transcript: The original transcript text to rewrite.
        hits: The list of retrieval hits to use as context.
        intent: The user's intent (default "generic").
        scope_used: List of scopes used for retrieval (default None -> []).
        queries_used: List of queries used for retrieval (default None -> []).

    Returns:
        EnhanceResult with mode="rewrite", citations=hits, and verified anchors.

    Raises:
        EnhanceError: when hits is empty (caller should downgrade to verbatim).
    """
    if not hits:
        raise EnhanceError("enhance_rewrite requires at least one retrieved hit")

    start_time_ms = int(time.perf_counter() * 1000)

    context_block = _build_context_block(hits)
    user_message = f"<transcript>\n{transcript}\n</transcript>\n\n{context_block}"

    client = _get_client()
    llm_start_ms = int(time.perf_counter() * 1000)
    resp = client.messages.create(
        model=MODEL_REWRITE,
        max_tokens=2048,
        system=REWRITE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    llm_end_ms = int(time.perf_counter() * 1000)
    llm_ms = llm_end_ms - llm_start_ms

    raw = _extract_text(resp)
    cleaned, anchor_warnings = verify_anchors(raw, hits)

    scope_used_list = scope_used if scope_used is not None else []
    queries_used_list = queries_used if queries_used is not None else []

    end_time_ms = int(time.perf_counter() * 1000)
    enhance_ms = end_time_ms - start_time_ms

    return EnhanceResult(
        transcript=transcript,
        enhanced_prompt=cleaned,
        mode="rewrite",
        citations=hits,
        intent=intent,
        scope_used=scope_used_list,
        queries_used=queries_used_list,
        warnings=anchor_warnings,
        timings_ms={"llm_ms": llm_ms, "enhance_ms": enhance_ms},
    )
