from __future__ import annotations

import ctypes
import time
from typing import Any

import pyautogui
import pyperclip

pyautogui.FAILSAFE = False


def inject(text: str, settings: dict[str, Any], *, target_hwnd: int = 0) -> bool:
    pyperclip.copy(text)
    if not settings.get("auto_paste", True):
        return True
    time.sleep(0.12)
    try:
        if target_hwnd:
            ctypes.windll.user32.SetForegroundWindow(target_hwnd)
            time.sleep(0.05)
        pyautogui.hotkey("ctrl", "v")
        return True
    except Exception:
        return False
