from unittest.mock import patch

from whisper_tray.injector import inject
from whisper_tray.settings import DEFAULTS


def test_inject_copies_to_clipboard():
    settings = {**DEFAULTS, "auto_paste": False}
    with patch("whisper_tray.injector.pyperclip") as mock_clip:
        inject("hello world", settings)
    mock_clip.copy.assert_called_once_with("hello world")


def test_inject_pastes_when_auto_paste_true():
    settings = {**DEFAULTS, "auto_paste": True}
    with patch("whisper_tray.injector.pyperclip"), \
         patch("whisper_tray.injector.pyautogui") as mock_gui, \
         patch("whisper_tray.injector.time") as mock_time:
        inject("hello", settings)
    mock_time.sleep.assert_called_once_with(0.12)
    mock_gui.hotkey.assert_called_once_with("ctrl", "v")


def test_inject_no_paste_when_auto_paste_false():
    settings = {**DEFAULTS, "auto_paste": False}
    with patch("whisper_tray.injector.pyperclip"), \
         patch("whisper_tray.injector.pyautogui") as mock_gui:
        inject("hello", settings)
    mock_gui.hotkey.assert_not_called()


def test_inject_swallows_pyautogui_error():
    settings = {**DEFAULTS, "auto_paste": True}
    with patch("whisper_tray.injector.pyperclip"), \
         patch("whisper_tray.injector.pyautogui") as mock_gui, \
         patch("whisper_tray.injector.time"):
        mock_gui.hotkey.side_effect = Exception("elevated window")
        # Should not raise
        inject("hello", settings)


def test_inject_returns_true_on_success():
    settings = {**DEFAULTS, "auto_paste": False}
    with patch("whisper_tray.injector.pyperclip"):
        result = inject("hello", settings)
    assert result is True


def test_inject_returns_false_on_pyautogui_error():
    settings = {**DEFAULTS, "auto_paste": True}
    with patch("whisper_tray.injector.pyperclip"), \
         patch("whisper_tray.injector.pyautogui") as mock_gui, \
         patch("whisper_tray.injector.time"):
        mock_gui.hotkey.side_effect = Exception("elevated window")
        result = inject("hello", settings)
    assert result is False
