from __future__ import annotations

import tkinter as tk
from typing import Callable

PILL_BG = "#1a1a2e"
TEXT_COLOR = "#e0e0e0"
CANCEL_BG = "#444"
STOP_BG = "#cc2222"
DONE_BG = "#22aa44"
WARN_BG = "#cc8800"
MODE_OPTIONS = ["raw", "clean", "context"]
MODE_HELP = {
    "raw":     "Transcript only — no AI, no rephrasing. Fastest.",
    "clean":   "Fix grammar & remove filler words (um, uh). One quick AI call.",
    "context": "Full prompt with project KB context. Best for Claude Code.",
}
DOT_CHARS = ["⠋", "⠙", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class Pill:
    def __init__(
        self,
        root: tk.Tk,
        on_cancel: Callable[[], None],
        on_stop: Callable[[], None],
        on_mode_change: Callable[[str], None],
        initial_mode: str = "rewrite",
        mode_lock: bool = False,
    ) -> None:
        self._root = root
        self._on_cancel = on_cancel
        self._on_stop = on_stop
        self._on_mode_change = on_mode_change
        self._mode_lock = mode_lock
        self._dot_idx = 0
        self._animate_job: str | None = None

        self._win = tk.Toplevel(root)
        self._win.overrideredirect(True)
        self._win.wm_attributes("-topmost", True)
        self._win.withdraw()
        self._win.configure(bg=PILL_BG)
        self._win.wm_attributes("-alpha", 0.92)

        self._frame = tk.Frame(self._win, bg=PILL_BG, padx=12, pady=8)
        self._frame.pack()

        self._dot_label = tk.Label(self._frame, text="⠋", fg="#aa44ff", bg=PILL_BG,
                                   font=("Segoe UI", 14))
        self._dot_label.grid(row=0, column=0, padx=(0, 8))

        self._status_label = tk.Label(self._frame, text="Recording…", fg=TEXT_COLOR,
                                      bg=PILL_BG, font=("Segoe UI", 11))
        self._status_label.grid(row=0, column=1, padx=(0, 12))

        if not mode_lock:
            self._mode_var = tk.StringVar(value=initial_mode)
            self._mode_frame = tk.Frame(self._frame, bg=PILL_BG)
            self._mode_frame.grid(row=0, column=2, padx=(0, 12))
            for m in MODE_OPTIONS:
                rb = tk.Radiobutton(
                    self._mode_frame, text=m, variable=self._mode_var, value=m,
                    bg=PILL_BG, fg=TEXT_COLOR, selectcolor="#333",
                    activebackground=PILL_BG, activeforeground=TEXT_COLOR,
                    font=("Segoe UI", 9),
                    command=lambda: on_mode_change(self._mode_var.get()),
                )
                rb.pack(side=tk.LEFT, padx=2)

        self._cancel_btn = tk.Button(
            self._frame, text="✕", bg=CANCEL_BG, fg=TEXT_COLOR,
            relief=tk.FLAT, font=("Segoe UI", 10), padx=6,
            command=self._on_cancel,
        )
        self._cancel_btn.grid(row=0, column=3, padx=(0, 4))

        self._stop_btn = tk.Button(
            self._frame, text="■", bg=STOP_BG, fg="white",
            relief=tk.FLAT, font=("Segoe UI", 10), padx=6,
            command=self._on_stop,
        )
        self._stop_btn.grid(row=0, column=4)

        self._help_btn = tk.Button(
            self._frame, text="?", bg=PILL_BG, fg="#888",
            relief=tk.FLAT, font=("Segoe UI", 9), padx=4,
            command=self._toggle_help,
        )
        self._help_btn.grid(row=0, column=5, padx=(6, 0))
        self._help_popup: tk.Toplevel | None = None

        self._bind_drag()

    def _center_bottom(self) -> None:
        self._win.update_idletasks()
        sw = self._root.winfo_screenwidth()
        sh = self._root.winfo_screenheight()
        w = self._win.winfo_reqwidth()
        h = self._win.winfo_reqheight()
        x = (sw - w) // 2
        y = sh - h - 80
        self._win.geometry(f"+{x}+{y}")

    def _bind_drag(self) -> None:
        self._drag_x = 0
        self._drag_y = 0

        def on_drag_start(event: tk.Event) -> None:
            self._drag_x = event.x
            self._drag_y = event.y

        def on_drag_motion(event: tk.Event) -> None:
            dx = event.x - self._drag_x
            dy = event.y - self._drag_y
            x = self._win.winfo_x() + dx
            y = self._win.winfo_y() + dy
            self._win.geometry(f"+{x}+{y}")

        self._frame.bind("<ButtonPress-1>", on_drag_start)
        self._frame.bind("<B1-Motion>", on_drag_motion)

    def _stop_animate(self) -> None:
        if self._animate_job is not None:
            self._root.after_cancel(self._animate_job)
            self._animate_job = None

    def _animate_dots(self) -> None:
        self._dot_idx = (self._dot_idx + 1) % len(DOT_CHARS)
        self._dot_label.configure(text=DOT_CHARS[self._dot_idx])
        self._animate_job = self._root.after(100, self._animate_dots)

    def show_recording(self) -> None:
        self._root.after(0, self._do_show_recording)

    def _do_show_recording(self) -> None:
        self._stop_animate()
        self._dot_label.configure(text=DOT_CHARS[0], fg="#aa44ff")
        self._status_label.configure(text="Recording…")
        self._cancel_btn.grid(row=0, column=3, padx=(0, 4))
        self._stop_btn.grid(row=0, column=4)
        self._win.deiconify()
        self._center_bottom()
        self._animate_dots()

    def show_processing(self) -> None:
        self._root.after(0, self._do_show_processing)

    def _do_show_processing(self) -> None:
        self._stop_animate()
        self._status_label.configure(text="Enhancing…")
        self._dot_label.configure(fg="#4488ff")
        self._cancel_btn.grid_remove()
        self._stop_btn.grid_remove()
        self._animate_dots()

    def show_done(self) -> None:
        self._root.after(0, self._do_show_done)

    def _do_show_done(self) -> None:
        self._stop_animate()
        self._win.configure(bg=DONE_BG)
        self._frame.configure(bg=DONE_BG)
        self._status_label.configure(text="Done ✓", bg=DONE_BG)
        self._dot_label.configure(fg=DONE_BG, bg=DONE_BG)
        self._root.after(800, self._do_hide)

    def show_error(self, message: str, duration_ms: int = 1500) -> None:
        self._root.after(0, lambda: self._do_show_error(message, duration_ms))

    def _do_show_error(self, message: str, duration_ms: int) -> None:
        self._stop_animate()
        self._win.configure(bg=WARN_BG)
        self._frame.configure(bg=WARN_BG)
        self._status_label.configure(text=message, bg=WARN_BG)
        self._dot_label.configure(fg=WARN_BG, bg=WARN_BG)
        self._root.after(duration_ms, self._do_hide)

    def hide(self) -> None:
        self._root.after(0, self._do_hide)

    def _do_hide(self) -> None:
        self._stop_animate()
        self._win.configure(bg=PILL_BG)
        self._frame.configure(bg=PILL_BG)
        self._status_label.configure(bg=PILL_BG)
        self._dot_label.configure(bg=PILL_BG)
        self._win.withdraw()

    def _toggle_help(self) -> None:
        if self._help_popup and self._help_popup.winfo_exists():
            self._help_popup.destroy()
            self._help_popup = None
            return
        popup = tk.Toplevel(self._root)
        popup.overrideredirect(True)
        popup.wm_attributes("-topmost", True)
        popup.configure(bg="#2a2a40")
        frame = tk.Frame(popup, bg="#2a2a40", padx=12, pady=10)
        frame.pack()
        tk.Label(frame, text="Modes", fg="#aaaacc", bg="#2a2a40",
                 font=("Segoe UI", 9, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        for i, (mode, desc) in enumerate(MODE_HELP.items(), start=1):
            tk.Label(frame, text=mode, fg="#cc99ff", bg="#2a2a40",
                     font=("Segoe UI", 9, "bold"), anchor="w", width=8).grid(row=i, column=0, sticky="w")
            tk.Label(frame, text=desc, fg="#cccccc", bg="#2a2a40",
                     font=("Segoe UI", 9), anchor="w").grid(row=i, column=1, sticky="w")
        popup.update_idletasks()
        bx = self._help_btn.winfo_rootx()
        by = self._help_btn.winfo_rooty()
        pw = popup.winfo_reqwidth()
        popup.geometry(f"+{bx - pw + 20}+{by - popup.winfo_reqheight() - 6}")
        popup.bind("<FocusOut>", lambda _e: popup.destroy())
        popup.focus_set()
        self._help_popup = popup

    def set_mode(self, mode: str) -> None:
        if not self._mode_lock:
            self._root.after(0, lambda: self._mode_var.set(mode))
