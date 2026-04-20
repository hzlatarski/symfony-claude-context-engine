from __future__ import annotations

import threading
from datetime import datetime
from typing import TYPE_CHECKING, Any

from whisper_tray.history import History

if TYPE_CHECKING:
    from whisper_tray.audio_recorder import AudioRecorder


class AppState:
    def __init__(self, settings: dict[str, Any], history: History) -> None:
        self.settings = settings
        self.history = history

        self.start_record_event = threading.Event()
        self.stop_record_event = threading.Event()
        self.cancel_event = threading.Event()

        self.snooze_until: datetime | None = None
        self.current_mode: str = settings.get("enhancement_mode", "rewrite")
        self.recorder: AudioRecorder | None = None
        self.target_hwnd: int = 0  # foreground window captured at recording start

    def is_snoozed(self) -> bool:
        if self.snooze_until is None:
            return False
        return datetime.now() < self.snooze_until
