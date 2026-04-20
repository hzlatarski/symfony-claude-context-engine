from __future__ import annotations

import threading
from datetime import datetime, timedelta
from typing import Any, Callable

import pystray
import sounddevice as sd
from PIL import Image, ImageDraw

LANGUAGE_OPTIONS = [
    ("Auto-detect", "auto"), ("English", "en"), ("German", "de"),
    ("Spanish", "es"), ("French", "fr"), ("Italian", "it"),
    ("Portuguese", "pt"), ("Dutch", "nl"), ("Polish", "pl"),
    ("Russian", "ru"), ("Chinese", "zh"), ("Japanese", "ja"),
]
ICON_SIZE = 64


def _make_icon_image(color: str = "#8855ff") -> Image.Image:
    img = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([4, 4, ICON_SIZE - 4, ICON_SIZE - 4], fill=color)
    mic_w, mic_h = 18, 28
    mx = (ICON_SIZE - mic_w) // 2
    my = (ICON_SIZE - mic_h) // 2 - 4
    draw.rounded_rectangle([mx, my, mx + mic_w, my + mic_h], radius=9, fill="white")
    stand_y = my + mic_h + 2
    draw.arc([mx - 6, stand_y - 12, mx + mic_w + 6, stand_y + 4], 0, 180, fill="white", width=3)
    draw.line([(ICON_SIZE // 2, stand_y + 2), (ICON_SIZE // 2, stand_y + 8)],
              fill="white", width=3)
    draw.line([(ICON_SIZE // 2 - 8, stand_y + 8), (ICON_SIZE // 2 + 8, stand_y + 8)],
              fill="white", width=3)
    return img


class TrayIcon:
    def __init__(
        self,
        app_state: Any,
        on_settings: Callable[[], None],
        on_paste_last: Callable[[], None],
        on_quit: Callable[[], None],
    ) -> None:
        self._state = app_state
        self._on_settings = on_settings
        self._on_paste_last = on_paste_last
        self._on_quit = on_quit
        self._icon: pystray.Icon | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._icon = pystray.Icon(
            "WhisperTray",
            icon=_make_icon_image(),
            title="WhisperTray",
            menu=self._build_menu(),
        )
        self._thread = threading.Thread(target=self._icon.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._icon:
            self._icon.stop()

    def set_recording(self, recording: bool) -> None:
        if self._icon:
            self._icon.icon = _make_icon_image("#cc2222" if recording else "#8855ff")
            self._icon.update_menu()

    def update_menu(self) -> None:
        if self._icon:
            self._icon.update_menu()

    def set_processing(self) -> None:
        if self._icon:
            self._icon.icon = _make_icon_image("#4488ff")

    def _build_menu(self) -> pystray.Menu:
        def snooze_1h(icon: pystray.Icon, item: pystray.MenuItem) -> None:
            self._state.snooze_until = datetime.now() + timedelta(hours=1)

        def mic_item(name: str, idx: int | str) -> pystray.MenuItem:
            def select(icon: pystray.Icon, item: pystray.MenuItem) -> None:
                self._state.settings["microphone"] = idx
                if self._state.recorder:
                    self._state.recorder._device = None if idx == "auto" else idx

            def checked(item: pystray.MenuItem) -> bool:
                return self._state.settings.get("microphone", "auto") == idx

            return pystray.MenuItem(name, select, checked=checked, radio=True)

        def lang_item(label: str, code: str) -> pystray.MenuItem:
            def select(icon: pystray.Icon, item: pystray.MenuItem) -> None:
                self._state.settings["language"] = code

            def checked(item: pystray.MenuItem) -> bool:
                return self._state.settings.get("language", "auto") == code

            return pystray.MenuItem(label, select, checked=checked, radio=True)

        def history_item(entry: Any, truncated: str) -> pystray.MenuItem:
            def copy_to_clipboard(icon: pystray.Icon, item: pystray.MenuItem) -> None:
                import pyperclip
                pyperclip.copy(entry.enhanced_prompt)

            return pystray.MenuItem(truncated, copy_to_clipboard)

        def history_submenu() -> pystray.Menu:
            entries = self._state.history.entries()
            if not entries:
                return pystray.Menu(pystray.MenuItem("(empty)", None, enabled=False))
            items = [
                history_item(e, e.enhanced_prompt[:40] + ("…" if len(e.enhanced_prompt) > 40 else ""))
                for e in reversed(entries)
            ]
            return pystray.Menu(*items)

        try:
            devices = sd.query_devices()
        except Exception:
            devices = []
        mic_items = [mic_item("Auto-detect default", "auto")] + [
            mic_item(d["name"], i)
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]

        lang_items = [lang_item(label, code) for label, code in LANGUAGE_OPTIONS]

        mode_help_items = pystray.Menu(
            pystray.MenuItem("raw — transcript only, no AI, fastest", None, enabled=False),
            pystray.MenuItem("clean — fix grammar & remove filler words", None, enabled=False),
            pystray.MenuItem("context — full prompt with KB context injected", None, enabled=False),
        )

        return pystray.Menu(
            pystray.MenuItem("🎙 WhisperTray", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("⏱ Hide for 1 hour", snooze_1h),
            pystray.MenuItem("⚙ Settings", lambda i, it: self._on_settings()),
            pystray.MenuItem("❓ Modes", mode_help_items),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("🎤 Microphone", pystray.Menu(*mic_items)),
            pystray.MenuItem("🌐 Language", pystray.Menu(*lang_items)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("📋 Transcript history", pystray.Menu(history_submenu)),
            pystray.MenuItem("📎 Paste last transcript", lambda i, it: self._on_paste_last()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("✕ Quit", lambda i, it: self._on_quit()),
        )
