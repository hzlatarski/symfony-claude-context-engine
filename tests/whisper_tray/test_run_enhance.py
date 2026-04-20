"""Tests for _run_enhance_task — the core enhance pipeline error handling.

All tests inject a fake enhance_fn and no_speech_cls so no real LLM or audio
API is invoked. The focus is on correct pill state transitions, history
recording, inject calls, event clearing, and logging.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from whisper_tray.main import _run_enhance_task


@dataclass
class _FakeResult:
    transcript: str = "hello"
    enhanced_prompt: str = "Hello world"
    mode: str = "rewrite"


class _NoSpeech(Exception):
    """Fake NoSpeechError injected instead of the real orchestrator class."""


def _call(
    enhance_fn,
    pill=None,
    cancel: bool = False,
    mode: str = "rewrite",
    language: str = "auto",
):
    """Helper: run _run_enhance_task with fully-mocked dependencies."""
    cancel_event = threading.Event()
    start_event = threading.Event()
    stop_event = threading.Event()
    start_event.set()
    if cancel:
        cancel_event.set()

    pill_ref = [pill if pill is not None else MagicMock()]
    inject_fn = MagicMock()
    history = MagicMock()

    _run_enhance_task(
        b"audio",
        mode=mode,
        language=language,
        history=history,
        cancel_event=cancel_event,
        start_record_event=start_event,
        stop_record_event=stop_event,
        pill_ref=pill_ref,
        settings={},
        inject_fn=inject_fn,
        enhance_fn=enhance_fn,
        no_speech_cls=_NoSpeech,
    )
    return history, inject_fn, pill_ref[0], cancel_event, start_event, stop_event


# --- success path ---

def test_success_appends_history_injects_and_shows_done():
    enhance = MagicMock(return_value=_FakeResult())
    history, inject_fn, pill, *_ = _call(enhance)

    history.append.assert_called_once()
    inject_fn.assert_called_once_with("Hello world", {})
    pill.show_done.assert_called_once()


def test_success_cancelled_records_history_but_skips_inject():
    enhance = MagicMock(return_value=_FakeResult())
    history, inject_fn, pill, *_ = _call(enhance, cancel=True)

    history.append.assert_called_once()  # history always recorded
    inject_fn.assert_not_called()
    pill.show_done.assert_not_called()


# --- NoSpeechError path ---

def test_no_speech_shows_correct_pill_message():
    enhance = MagicMock(side_effect=_NoSpeech("silence"))
    history, inject_fn, pill, *_ = _call(enhance)

    history.append.assert_not_called()
    inject_fn.assert_not_called()
    pill.show_error.assert_called_once_with("No speech detected", duration_ms=1500)


def test_no_speech_logs_info(caplog):
    enhance = MagicMock(side_effect=_NoSpeech())
    with patch("whisper_tray.main.logger") as mock_logger:
        _call(enhance)
    mock_logger.info.assert_called_once()


# --- generic Exception path ---

def test_generic_exception_shows_truncated_pill_error():
    long_msg = "Haiku returned non-JSON output: extra extra long error message here"
    enhance = MagicMock(side_effect=ValueError(long_msg))
    _, _, pill, *_ = _call(enhance)

    pill.show_error.assert_called_once()
    msg, = pill.show_error.call_args[0]
    assert len(msg) <= 40
    assert msg.startswith("Error:")


def test_generic_exception_logs_full_traceback():
    enhance = MagicMock(side_effect=RuntimeError("boom"))
    with patch("whisper_tray.main.logger") as mock_logger:
        _call(enhance)
    mock_logger.exception.assert_called_once()
    assert "enhance pipeline failed" in mock_logger.exception.call_args[0][0]


# --- events always cleared (finally block) ---

def test_events_cleared_on_success():
    enhance = MagicMock(return_value=_FakeResult())
    *_, cancel_ev, start_ev, stop_ev = _call(enhance)
    assert not cancel_ev.is_set()
    assert not start_ev.is_set()
    assert not stop_ev.is_set()


def test_events_cleared_on_no_speech():
    enhance = MagicMock(side_effect=_NoSpeech())
    *_, cancel_ev, start_ev, stop_ev = _call(enhance)
    assert not cancel_ev.is_set()
    assert not start_ev.is_set()
    assert not stop_ev.is_set()


def test_events_cleared_on_error():
    enhance = MagicMock(side_effect=RuntimeError("boom"))
    *_, cancel_ev, start_ev, stop_ev = _call(enhance)
    assert not cancel_ev.is_set()
    assert not start_ev.is_set()
    assert not stop_ev.is_set()


# --- None pill guard ---

def test_no_pill_does_not_raise_on_error():
    enhance = MagicMock(side_effect=RuntimeError("boom"))
    # Should complete without AttributeError even though pill is None
    _run_enhance_task(
        b"audio",
        mode="rewrite",
        language="auto",
        history=MagicMock(),
        cancel_event=threading.Event(),
        start_record_event=threading.Event(),
        stop_record_event=threading.Event(),
        pill_ref=[None],
        settings={},
        inject_fn=MagicMock(),
        enhance_fn=enhance,
        no_speech_cls=_NoSpeech,
    )


def test_no_pill_does_not_raise_on_no_speech():
    enhance = MagicMock(side_effect=_NoSpeech())
    _run_enhance_task(
        b"audio",
        mode="rewrite",
        language="auto",
        history=MagicMock(),
        cancel_event=threading.Event(),
        start_record_event=threading.Event(),
        stop_record_event=threading.Event(),
        pill_ref=[None],
        settings={},
        inject_fn=MagicMock(),
        enhance_fn=enhance,
        no_speech_cls=_NoSpeech,
    )


def test_no_pill_does_not_raise_on_success():
    enhance = MagicMock(return_value=_FakeResult())
    _run_enhance_task(
        b"audio",
        mode="rewrite",
        language="auto",
        history=MagicMock(),
        cancel_event=threading.Event(),
        start_record_event=threading.Event(),
        stop_record_event=threading.Event(),
        pill_ref=[None],
        settings={},
        inject_fn=MagicMock(),
        enhance_fn=enhance,
        no_speech_cls=_NoSpeech,
    )
