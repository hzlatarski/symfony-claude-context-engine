from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SETTINGS_DIR = Path.home() / ".whisper-tray"
SETTINGS_FILE = SETTINGS_DIR / "settings.json"

DEFAULTS: dict[str, Any] = {
    "hotkey": "<ctrl>+<cmd>",
    "hotkey_mode": "click_toggle",
    "enhancement_mode": "context",
    "mode_lock_enabled": False,
    "auto_paste": True,
    "microphone": "auto",
    "language": "auto",
    "whisper_server_url": "http://127.0.0.1:9000",
    "pill_position": "bottom-center",
    "startup_with_windows": False,
}


def load_settings(path: Path = SETTINGS_FILE) -> dict[str, Any]:
    if not path.exists():
        return dict(DEFAULTS)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        settings = {**DEFAULTS, **data}
        mode = settings.get("enhancement_mode")
        if mode == "verbatim":
            settings["enhancement_mode"] = "raw"
        elif mode == "rewrite":
            settings["enhancement_mode"] = "context"
        return settings
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULTS)


def save_settings(settings: dict[str, Any], path: Path = SETTINGS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def is_first_run(path: Path = SETTINGS_FILE) -> bool:
    return not path.exists()
