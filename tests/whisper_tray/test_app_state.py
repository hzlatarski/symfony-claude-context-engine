from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from whisper_tray.app_state import AppState
from whisper_tray.settings import DEFAULTS


def _make_state(tmp_path: Path) -> AppState:
    settings = {**DEFAULTS}
    history = MagicMock()
    return AppState(settings=settings, history=history)


def test_app_state_initial_not_recording(tmp_path):
    state = _make_state(tmp_path)
    assert not state.start_record_event.is_set()
    assert not state.stop_record_event.is_set()
    assert not state.cancel_event.is_set()


def test_app_state_is_snoozed_false_by_default(tmp_path):
    state = _make_state(tmp_path)
    assert state.is_snoozed() is False


def test_app_state_is_snoozed_true_when_future(tmp_path):
    state = _make_state(tmp_path)
    state.snooze_until = datetime.now() + timedelta(hours=1)
    assert state.is_snoozed() is True


def test_app_state_is_snoozed_false_when_past(tmp_path):
    state = _make_state(tmp_path)
    state.snooze_until = datetime.now() - timedelta(seconds=1)
    assert state.is_snoozed() is False


def test_app_state_current_mode_from_settings(tmp_path):
    settings = {**DEFAULTS, "enhancement_mode": "clean"}
    state = AppState(settings=settings, history=MagicMock())
    assert state.current_mode == "clean"
