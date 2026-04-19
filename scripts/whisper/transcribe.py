"""faster-whisper wrapper with a lazy, process-local singleton model.

Why a singleton:
  The faster-whisper WhisperModel loads ~150-500MB of weights depending
  on the size tier and takes 3-5s to initialize. We want the viewer to
  pay that cost once at startup (see preload_model below), and every
  subsequent transcription to reuse the same instance.

Why sync:
  faster-whisper is CPU-bound inside C++ via CTranslate2 — async buys
  us nothing. The FastAPI endpoint runs this in a threadpool via
  run_in_threadpool so the event loop stays responsive.
"""
from __future__ import annotations

import io
import logging
import threading
from typing import Optional

import config

logger = logging.getLogger(__name__)

_MODEL = None
_LOCK = threading.Lock()


def _get_model():
    """Return the shared WhisperModel, loading it on first call."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _LOCK:
        if _MODEL is not None:  # double-checked locking
            return _MODEL
        from faster_whisper import WhisperModel
        logger.info(
            "Loading faster-whisper model size=%s device=%s (one-time cost ~3-5s)",
            config.WHISPER_MODEL_SIZE,
            config.WHISPER_DEVICE,
        )
        _MODEL = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type="int8",
        )
        return _MODEL


def preload_model() -> None:
    """Eagerly load the model. Call at viewer startup to avoid
    paying the 3-5s cold-start cost on the first user request."""
    _get_model()


def transcribe(audio: bytes, language: str = "auto") -> str:
    """Transcribe an audio clip to text.

    Args:
        audio: raw bytes in a container faster-whisper can decode
            (webm/opus from browser MediaRecorder, wav, mp3, flac).
            An empty bytes object yields an empty string without
            invoking the model.
        language: ISO code ("en", "de") or "auto" for autodetect.

    Returns:
        The transcribed text with inter-segment whitespace normalized.
        Empty string when no speech is detected.
    """
    if not audio:
        return ""

    model = _get_model()
    lang: Optional[str] = None if language == "auto" else language

    segments, _info = model.transcribe(
        io.BytesIO(audio),
        language=lang,
        vad_filter=True,  # drop silence, speeds up short clips
        beam_size=5,
    )

    parts = [seg.text.strip() for seg in segments if seg.text.strip()]
    return " ".join(parts)
