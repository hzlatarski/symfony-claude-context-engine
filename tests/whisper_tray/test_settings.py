import json
from pathlib import Path

import pytest

from whisper_tray.settings import DEFAULTS, is_first_run, load_settings, save_settings


def test_defaults_have_required_keys():
    required = {
        "hotkey", "hotkey_mode", "enhancement_mode", "mode_lock_enabled",
        "auto_paste", "microphone", "language", "whisper_server_url",
        "pill_position", "startup_with_windows",
    }
    assert required <= set(DEFAULTS.keys())


def test_load_settings_returns_defaults_when_no_file(tmp_path):
    settings_file = tmp_path / "settings.json"
    result = load_settings(settings_file)
    assert result == DEFAULTS


def test_load_settings_merges_partial_file(tmp_path):
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps({"hotkey": "<ctrl>+<alt>+w"}))
    result = load_settings(settings_file)
    assert result["hotkey"] == "<ctrl>+<alt>+w"
    assert result["auto_paste"] == DEFAULTS["auto_paste"]


def test_save_settings_writes_json(tmp_path):
    settings_file = tmp_path / "settings.json"
    data = {**DEFAULTS, "hotkey": "<ctrl>+<alt>+w"}
    save_settings(data, settings_file)
    saved = json.loads(settings_file.read_text())
    assert saved["hotkey"] == "<ctrl>+<alt>+w"


def test_is_first_run_true_when_no_file(tmp_path):
    assert is_first_run(tmp_path / "settings.json") is True


def test_is_first_run_false_when_file_exists(tmp_path):
    f = tmp_path / "settings.json"
    f.write_text("{}")
    assert is_first_run(f) is False
