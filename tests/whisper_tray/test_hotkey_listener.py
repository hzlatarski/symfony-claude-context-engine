from unittest.mock import MagicMock, patch

from whisper_tray.hotkey_listener import HotkeyListener


def _make_listener(hotkey="<ctrl>+<cmd>", mode="click_toggle"):
    state = MagicMock()
    state.is_snoozed.return_value = False
    state.start_record_event = MagicMock()
    state.stop_record_event = MagicMock()
    with patch("whisper_tray.hotkey_listener.keyboard.GlobalHotKeys"):
        listener = HotkeyListener(state=state, hotkey=hotkey, hotkey_mode=mode)
    return listener, state


def test_hotkey_activate_starts_recording_when_idle():
    listener, state = _make_listener()
    listener._recording = False
    listener._on_activate()
    state.start_record_event.set.assert_called_once()
    assert listener._recording is True


def test_hotkey_activate_stops_recording_when_recording_click_toggle():
    listener, state = _make_listener(mode="click_toggle")
    listener._recording = True
    listener._on_activate()
    state.stop_record_event.set.assert_called_once()
    assert listener._recording is False


def test_hotkey_ignored_when_snoozed():
    listener, state = _make_listener()
    state.is_snoozed.return_value = True
    listener._recording = False
    listener._on_activate()
    state.start_record_event.set.assert_not_called()


def test_hold_mode_press_starts_recording():
    listener, state = _make_listener(mode="hold")
    listener._recording = False
    listener._on_press()
    state.start_record_event.set.assert_called_once()
    assert listener._recording is True


def test_hold_mode_release_stops_recording():
    listener, state = _make_listener(mode="hold")
    listener._recording = True
    listener._on_release()
    state.stop_record_event.set.assert_called_once()
    assert listener._recording is False
