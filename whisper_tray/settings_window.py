from __future__ import annotations

import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Callable

import sounddevice as sd
from pynput import keyboard

from whisper_tray.settings import DEFAULTS, save_settings

# Trimmed to the user's likely languages plus the two most-common Romance ones.
# Auto-detect is unreliable on short utterances (faster-whisper needs a few
# seconds of audio to sample language features), so a pinned language is the
# safer default for most users.
LANGUAGE_OPTIONS = [
    ("Auto-detect", "auto"),
    ("English", "en"),
    ("German", "de"),
    ("Bulgarian", "bg"),
    ("Spanish", "es"),
    ("French", "fr"),
]
BG = "#1e1e2e"
FG = "#e0e0e0"
ENTRY_BG = "#2a2a3e"


def _center_window(win: tk.Toplevel, parent: tk.Misc) -> None:
    win.update_idletasks()
    pw = parent.winfo_screenwidth()
    ph = parent.winfo_screenheight()
    w = win.winfo_reqwidth()
    h = win.winfo_reqheight()
    win.geometry(f"+{(pw - w) // 2}+{(ph - h) // 2}")


class SettingsWindow(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Tk,
        settings: dict[str, Any],
        on_save: Callable[[dict[str, Any]], None],
    ) -> None:
        super().__init__(parent)
        self.title("WhisperTray Settings")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.grab_set()

        self._settings = dict(settings)
        self._on_save = on_save
        self._capturing_hotkey = False
        self._hotkey_listener: keyboard.Listener | None = None
        self._pressed_keys: set[str] = set()

        self._build()
        _center_window(self, parent)

    def _build(self) -> None:
        pad = {"padx": 12, "pady": 6}
        row = 0

        # Hotkey
        tk.Label(self, text="Hotkey", bg=BG, fg=FG).grid(row=row, column=0, sticky="w", **pad)
        self._hotkey_var = tk.StringVar(value=self._settings.get("hotkey", DEFAULTS["hotkey"]))
        hotkey_entry = tk.Entry(self, textvariable=self._hotkey_var, bg=ENTRY_BG, fg=FG,
                                insertbackground=FG, width=25)
        hotkey_entry.grid(row=row, column=1, **pad)
        self._capture_btn = tk.Button(self, text="Record…", bg="#333", fg=FG,
                                      relief=tk.FLAT, command=self._start_capture)
        self._capture_btn.grid(row=row, column=2, **pad)
        row += 1

        # Hotkey mode
        tk.Label(self, text="Hotkey mode", bg=BG, fg=FG).grid(row=row, column=0, sticky="w", **pad)
        self._mode_var = tk.StringVar(value=self._settings.get("hotkey_mode", "click_toggle"))
        for val, label in [("click_toggle", "Click/toggle"), ("hold", "Hold")]:
            tk.Radiobutton(self, text=label, variable=self._mode_var, value=val,
                           bg=BG, fg=FG, selectcolor="#333",
                           activebackground=BG).grid(row=row, column=1, sticky="w", **pad)
            row += 1

        # Enhancement mode
        tk.Label(self, text="Enhancement mode", bg=BG, fg=FG).grid(row=row, column=0, sticky="w", **pad)
        self._enhance_var = tk.StringVar(
            value=self._settings.get("enhancement_mode", DEFAULTS["enhancement_mode"]))
        ttk.Combobox(self, textvariable=self._enhance_var,
                     values=["raw", "clean", "context"], state="readonly",
                     width=15).grid(row=row, column=1, **pad)
        row += 1

        # Mode descriptions
        mode_help_frame = tk.Frame(self, bg=BG)
        mode_help_frame.grid(row=row, column=0, columnspan=3, sticky="w", padx=12, pady=(0, 4))
        for mode, desc in [
            ("raw",     "Transcript only — no AI, no rephrasing. Fastest."),
            ("clean",   "Fix grammar & remove filler words. One quick AI call."),
            ("context", "Full prompt with project KB context. Best for Claude Code."),
        ]:
            line = tk.Frame(mode_help_frame, bg=BG)
            line.pack(anchor="w")
            tk.Label(line, text=f"  {mode}", fg="#9977dd", bg=BG,
                     font=("Segoe UI", 8, "bold"), width=9, anchor="w").pack(side=tk.LEFT)
            tk.Label(line, text=desc, fg="#888", bg=BG,
                     font=("Segoe UI", 8)).pack(side=tk.LEFT)
        row += 1

        # Mode lock
        self._mode_lock_var = tk.BooleanVar(value=self._settings.get("mode_lock_enabled", False))
        tk.Checkbutton(self, text="Lock mode (hide per-recording selector)", bg=BG, fg=FG,
                       variable=self._mode_lock_var, selectcolor="#333",
                       activebackground=BG).grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1

        # Auto-paste
        self._auto_paste_var = tk.BooleanVar(value=self._settings.get("auto_paste", True))
        tk.Checkbutton(self, text="Auto-paste after enhance (Ctrl+V)", bg=BG, fg=FG,
                       variable=self._auto_paste_var, selectcolor="#333",
                       activebackground=BG).grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1

        # Microphone
        tk.Label(self, text="Microphone", bg=BG, fg=FG).grid(row=row, column=0, sticky="w", **pad)
        mic_names = ["Auto-detect"] + [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
        self._mic_var = tk.StringVar(value=self._settings.get("microphone", "auto"))
        ttk.Combobox(self, textvariable=self._mic_var,
                     values=mic_names, state="readonly", width=30).grid(row=row, column=1, **pad)
        row += 1

        # Language
        tk.Label(self, text="Language", bg=BG, fg=FG).grid(row=row, column=0, sticky="w", **pad)
        lang_labels = [label for label, _ in LANGUAGE_OPTIONS]
        lang_values = {label: code for label, code in LANGUAGE_OPTIONS}
        current_lang_code = self._settings.get("language", "auto")
        current_lang_label = next(
            (l for l, c in LANGUAGE_OPTIONS if c == current_lang_code), "Auto-detect")
        self._lang_label_var = tk.StringVar(value=current_lang_label)
        self._lang_values = lang_values
        ttk.Combobox(self, textvariable=self._lang_label_var,
                     values=lang_labels, state="readonly", width=15).grid(row=row, column=1, **pad)
        row += 1

        # Startup with Windows
        self._startup_var = tk.BooleanVar(
            value=self._settings.get("startup_with_windows", False))
        tk.Checkbutton(self, text="Launch with Windows", bg=BG, fg=FG,
                       variable=self._startup_var, selectcolor="#333",
                       activebackground=BG).grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1

        # Buttons
        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=12)
        tk.Button(btn_frame, text="Save", bg="#2255cc", fg="white", relief=tk.FLAT,
                  padx=16, command=self._save).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Cancel", bg="#333", fg=FG, relief=tk.FLAT,
                  padx=16, command=self.destroy).pack(side=tk.LEFT, padx=4)

    def _save(self) -> None:
        # Normalize + validate the hotkey before touching disk so an invalid
        # combo never gets persisted — otherwise listener.start() crashes the
        # next app launch with no obvious recovery.
        hotkey = self._hotkey_var.get().strip().replace(" ", "")
        try:
            keyboard.HotKey.parse(hotkey)
        except Exception as exc:
            messagebox.showerror(
                "Invalid hotkey",
                f"Could not parse hotkey '{hotkey}':\n{exc}\n\n"
                "Use combos like <ctrl>+<cmd>, <ctrl>+<alt>+w, or <f9>.\n"
                "Special keys go in angle brackets, literal characters do not.",
                parent=self,
            )
            return
        self._settings["hotkey"] = hotkey
        self._settings["hotkey_mode"] = self._mode_var.get()
        self._settings["enhancement_mode"] = self._enhance_var.get()
        self._settings["mode_lock_enabled"] = self._mode_lock_var.get()
        self._settings["auto_paste"] = self._auto_paste_var.get()
        mic = self._mic_var.get()
        self._settings["microphone"] = "auto" if mic == "Auto-detect" else mic
        self._settings["language"] = self._lang_values.get(
            self._lang_label_var.get(), "auto")
        self._settings["startup_with_windows"] = self._startup_var.get()
        self._apply_startup(self._settings["startup_with_windows"])
        save_settings(self._settings)
        self._on_save(self._settings)
        self.destroy()

    def _apply_startup(self, enabled: bool) -> None:
        try:
            import winreg
            key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path,
                                0, winreg.KEY_SET_VALUE) as key:
                if enabled:
                    uv_path = shutil.which("uv") or sys.executable.replace("python.exe", "uv.exe")
                    script_path = str(Path(__file__).parent / "main.py")
                    winreg.SetValueEx(key, "WhisperTray", 0, winreg.REG_SZ,
                                      f'"{uv_path}" run python "{script_path}"')
                else:
                    try:
                        winreg.DeleteValue(key, "WhisperTray")
                    except FileNotFoundError:
                        pass
        except Exception:
            pass

    def _start_capture(self) -> None:
        if self._capturing_hotkey:
            return
        self._capturing_hotkey = True
        self._capture_btn.configure(text="Press keys…")
        self._pressed_keys = set()
        self._hotkey_listener = keyboard.Listener(
            on_press=self._on_key_press, on_release=self._on_key_release)
        self._hotkey_listener.daemon = True
        self._hotkey_listener.start()

    def _on_key_press(self, key: Any) -> None:
        token = self._key_token(key)
        if token:
            self._pressed_keys.add(token)
            # Modifiers (angle-bracketed) come first, then literal chars.
            combo = "+".join(
                sorted(self._pressed_keys, key=lambda k: (not k.startswith("<"), k))
            )
            self.after(0, self._hotkey_var.set, combo)

    def _on_key_release(self, key: Any) -> None:
        token = self._key_token(key)
        if token:
            self._pressed_keys.discard(token)
        if self._pressed_keys:
            return
        if self._hotkey_listener:
            self._hotkey_listener.stop()
            self._hotkey_listener = None
        self._capturing_hotkey = False
        self.after(0, self._capture_btn.configure, {"text": "Record…"})

    def _key_token(self, key: Any) -> str | None:
        """Convert a pynput key event into a pynput-compatible hotkey token.

        Rules:
            - Literal chars return as-is: 'a', '1', '/'.
            - Special keys are returned as '<name>', with left/right variants
              normalised ('ctrl_l' -> 'ctrl') so Windows doesn't bake a
              handed modifier into the hotkey string.
        """
        try:
            c = key.char
            return c if c else None
        except AttributeError:
            name = str(key).replace("Key.", "")
            for suffix in ("_l", "_r", "_gr"):
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
                    break
            return f"<{name}>" if name else None


class FirstRunWizard(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Tk,
        settings: dict[str, Any],
        on_done: Callable[[dict[str, Any]], None],
    ) -> None:
        super().__init__(parent)
        self.title("WhisperTray — Setup")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.grab_set()

        self._settings = dict(settings)
        self._on_done = on_done
        self._build()
        _center_window(self, parent)

    def _build(self) -> None:
        pad = {"padx": 16, "pady": 8}

        tk.Label(self, text="Welcome to WhisperTray", bg=BG, fg=FG,
                 font=("Segoe UI", 14, "bold")).pack(**pad)
        tk.Label(self, text="Set your global hotkey, then click Finish.",
                 bg=BG, fg=FG).pack(**pad)

        hk_frame = tk.Frame(self, bg=BG)
        hk_frame.pack(**pad)
        tk.Label(hk_frame, text="Hotkey:", bg=BG, fg=FG).pack(side=tk.LEFT, padx=(0, 8))
        self._hotkey_var = tk.StringVar(value=self._settings.get("hotkey", DEFAULTS["hotkey"]))
        tk.Entry(hk_frame, textvariable=self._hotkey_var, bg=ENTRY_BG, fg=FG,
                 insertbackground=FG, width=20).pack(side=tk.LEFT)

        tk.Label(self, text="Auto-paste result into focused window?", bg=BG, fg=FG).pack(**pad)
        self._auto_paste_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self, text="Yes, auto-paste (Ctrl+V)", bg=BG, fg=FG,
                       variable=self._auto_paste_var, selectcolor="#333",
                       activebackground=BG).pack(**pad)

        tk.Button(self, text="Finish", bg="#2255cc", fg="white", relief=tk.FLAT,
                  padx=20, pady=6, command=self._finish).pack(pady=16)

    def _finish(self) -> None:
        self._settings["hotkey"] = self._hotkey_var.get()
        self._settings["auto_paste"] = self._auto_paste_var.get()
        save_settings(self._settings)
        self._on_done(self._settings)
        self.destroy()
