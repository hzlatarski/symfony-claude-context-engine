from __future__ import annotations

import ctypes
import time
from typing import Any

import pyautogui
import pyperclip

pyautogui.FAILSAFE = False


def _force_foreground(hwnd: int) -> None:
    """Reliably restore foreground focus using AttachThreadInput.

    Plain SetForegroundWindow fails silently when the calling process doesn't
    hold the foreground lock (which is always the case after the pill steals it).
    Attaching our input thread to the current foreground window's thread first
    grants us permission to reassign foreground.
    """
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    cur_fg = user32.GetForegroundWindow()
    if cur_fg == hwnd:
        return
    fg_thread = user32.GetWindowThreadProcessId(cur_fg, None)
    our_thread = kernel32.GetCurrentThreadId()
    user32.AttachThreadInput(our_thread, fg_thread, True)
    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)
    user32.AttachThreadInput(our_thread, fg_thread, False)


def inject(text: str, settings: dict[str, Any], *, target_hwnd: int = 0) -> bool:
    pyperclip.copy(text)
    if not settings.get("auto_paste", True):
        return True
    time.sleep(0.12)
    try:
        if target_hwnd:
            _force_foreground(target_hwnd)
            time.sleep(0.08)
        pyautogui.hotkey("ctrl", "v")
        return True
    except Exception:
        return False
