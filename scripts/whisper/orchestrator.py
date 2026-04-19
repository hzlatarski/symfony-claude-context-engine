"""End-to-end orchestrator for the whisper enhance pipeline.

Called by the FastAPI endpoints. Wires the four steps together,
handles mode branching (verbatim/rewrite/clean), and implements the
graceful degradations: rewrite with zero hits → verbatim, rewrite
LLM failure → verbatim, empty transcript → NoSpeechError.
"""
from __future__ import annotations

import logging
import time
from typing import Literal

from whisper.enhance import (
    EnhanceError,
    RewriteOutput,
    enhance_clean,
    enhance_rewrite as _enhance_rewrite_impl,
    enhance_verbatim as _enhance_verbatim_impl,
)
from whisper.expand_query import Expansion, ExpansionError, expand
from whisper.retrieve import retrieve
from whisper.transcribe import transcribe
from whisper.types import EnhanceResult

logger = logging.getLogger(__name__)

Mode = Literal["verbatim", "rewrite", "clean"]


class NoSpeechError(Exception):
    """Raised when Whisper returns an empty transcript (no speech detected)."""


def _ms_since(t: float) -> int:
    return int((time.monotonic() - t) * 1000)


def enhance_rewrite(transcript: str, hits: list) -> RewriteOutput:
    """Adapter: call the real enhance_rewrite and return a RewriteOutput.

    This thin wrapper lives at module level so tests can monkeypatch it
    independently of the underlying whisper.enhance implementation.
    """
    result = _enhance_rewrite_impl(transcript, hits)
    return RewriteOutput(prompt=result.enhanced_prompt, warnings=result.warnings)


def enhance_verbatim(transcript: str, hits: list) -> str:
    """Adapter: call the real enhance_verbatim and return just the prompt string.

    This thin wrapper lives at module level so tests can monkeypatch it
    independently of the underlying whisper.enhance implementation.
    """
    # Intentionally returns only the prompt string; the orchestrator builds the full EnhanceResult.
    result = _enhance_verbatim_impl(transcript, hits)
    return result.enhanced_prompt


def enhance_from_audio(audio: bytes, mode: Mode, language: str = "auto") -> EnhanceResult:
    """Full pipeline: audio → transcript → expand → retrieve → enhance."""
    t_start = time.monotonic()
    timings: dict[str, int] = {}

    t = time.monotonic()
    transcript = transcribe(audio, language=language)
    timings["transcribe"] = _ms_since(t)

    if not transcript.strip():
        raise NoSpeechError("Whisper returned empty transcript")

    return _enhance_common(
        transcript=transcript,
        mode=mode,
        scope_override=None,
        timings=timings,
        t_start=t_start,
    )


def enhance_from_transcript(
    transcript: str,
    mode: Mode,
    scope_override: list[str] | None = None,
) -> EnhanceResult:
    """Same pipeline as enhance_from_audio but with transcript already known.

    Used by the /re-enhance endpoint when the user toggles scope chips
    and wants a fresh rewrite without re-recording.
    """
    t_start = time.monotonic()
    return _enhance_common(
        transcript=transcript,
        mode=mode,
        scope_override=scope_override,
        timings={},
        t_start=t_start,
    )


def _enhance_common(
    transcript: str,
    mode: Mode,
    scope_override: list[str] | None,
    timings: dict[str, int],
    t_start: float,
) -> EnhanceResult:
    warnings: list[str] = []

    # Clean mode is the short path: no retrieval, no expansion.
    if mode == "clean":
        t = time.monotonic()
        try:
            prompt = enhance_clean(transcript)
        except EnhanceError as exc:
            logger.warning("clean enhance failed: %s", exc)
            prompt = transcript
            warnings.append(f"Clean failed; returned raw transcript: {exc}")
        timings["enhance"] = _ms_since(t)
        timings["total"] = _ms_since(t_start)
        return EnhanceResult(
            transcript=transcript,
            enhanced_prompt=prompt,
            mode="clean",
            citations=[],
            intent="generic",
            scope_used=[],
            queries_used=[],
            warnings=warnings,
            timings_ms=timings,
        )

    # verbatim + rewrite both need expand + retrieve.
    t = time.monotonic()
    try:
        expansion: Expansion = expand(transcript)
    except ExpansionError as exc:
        logger.warning("query expansion failed: %s", exc)
        # Fall back to using the raw transcript as the sole query.
        expansion = Expansion(queries=[transcript], intent="generic", scope=["articles"])
        warnings.append(f"Query expansion failed; using raw transcript as query: {exc}")
    timings["expand_query"] = _ms_since(t)

    scope = scope_override if scope_override is not None else expansion.scope

    t = time.monotonic()
    hits = retrieve(queries=expansion.queries, scope=scope)
    timings["retrieve"] = _ms_since(t)

    t = time.monotonic()
    if mode == "rewrite":
        if not hits:
            warnings.append("No project context found; returned verbatim transcript")
            prompt = enhance_verbatim(transcript, hits)
            effective_mode: Mode = "verbatim"
        else:
            try:
                rw = enhance_rewrite(transcript, hits)
                prompt = rw.prompt
                warnings.extend(rw.warnings)
                effective_mode = "rewrite"
            except EnhanceError as exc:
                logger.warning("rewrite enhance failed, downgrading: %s", exc)
                prompt = enhance_verbatim(transcript, hits)
                warnings.append(f"Rewrite failed; returned verbatim transcript: {exc}")
                effective_mode = "verbatim"
    else:  # verbatim
        prompt = enhance_verbatim(transcript, hits)
        effective_mode = "verbatim"
    timings["enhance"] = _ms_since(t)
    timings["total"] = _ms_since(t_start)

    return EnhanceResult(
        transcript=transcript,
        enhanced_prompt=prompt,
        mode=effective_mode,
        citations=hits,
        intent=expansion.intent,
        scope_used=scope,
        queries_used=expansion.queries,
        warnings=warnings,
        timings_ms=timings,
    )
