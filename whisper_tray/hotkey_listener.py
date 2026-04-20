from __future__ import annotations

import threading
from typing import Any

from pynput import keyboard


class HotkeyListener:
    def __init__(
        self,
        state: Any,
        hotkey: str = "<ctrl>+<cmd>",
        hotkey_mode: str = "click_toggle",
    ) -> None:
        self._state = state
        self._hotkey = hotkey
        self._hotkey_mode = hotkey_mode
        self._recording = False
        self._lock = threading.Lock()
        self._listener: keyboard.GlobalHotKeys | None = None

    def start(self) -> None:
        if self._hotkey_mode == "click_toggle":
            mapping = {self._hotkey: self._on_activate}
        else:  # hold
            mapping = {self._hotkey: self._on_press}
        self._listener = keyboard.GlobalHotKeys(mapping)
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def _on_activate(self) -> None:
        if self._state.is_snoozed():
            return
        with self._lock:
            if not self._recording:
                self._recording = True
                self._state.start_record_event.set()
            else:
                self._recording = False
                self._state.stop_record_event.set()

    def _on_press(self) -> None:
        if self._state.is_snoozed():
            return
        with self._lock:
            if self._recording:
                return
            self._recording = True
            self._state.start_record_event.set()

    def _on_release(self) -> None:
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            self._state.stop_record_event.set()
