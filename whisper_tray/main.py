from __future__ import annotations

import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

# Make packages importable when run as: uv run python whisper_tray/main.py
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "scripts"))

import ctypes
import tkinter as tk

from whisper_tray.app_state import AppState
from whisper_tray.audio_recorder import AudioRecorder
from whisper_tray.history import History, HistoryEntry
from whisper_tray.hotkey_listener import HotkeyListener
from whisper_tray.injector import inject
from whisper_tray.pill import Pill
from whisper_tray.settings import DEFAULTS, is_first_run, load_settings
from whisper_tray.settings_window import FirstRunWizard, SettingsWindow
from whisper_tray.tray import TrayIcon

logger = logging.getLogger(__name__)


def _run_enhance_task(
    audio: bytes,
    *,
    mode: str,
    language: str,
    history: History,
    cancel_event: threading.Event,
    start_record_event: threading.Event,
    stop_record_event: threading.Event,
    pill_ref: list,
    settings: dict,
    target_hwnd: int = 0,
    inject_fn: Callable | None = None,
    enhance_fn: Callable | None = None,
    no_speech_cls: type[Exception] | None = None,
    on_history_updated: Callable | None = None,
) -> None:
    """Run the enhance pipeline in a background thread.

    Accepts injectable enhance_fn, inject_fn, and no_speech_cls so unit tests
    can exercise all branches without hitting real LLM/audio APIs.
    """
    if inject_fn is None:
        inject_fn = inject
    if enhance_fn is None or no_speech_cls is None:
        from whisper.orchestrator import enhance_from_audio, NoSpeechError  # type: ignore[import]
        if enhance_fn is None:
            enhance_fn = enhance_from_audio
        if no_speech_cls is None:
            no_speech_cls = NoSpeechError
    assert enhance_fn is not None
    assert no_speech_cls is not None

    _mode_map = {"raw": "verbatim", "context": "rewrite"}
    orchestrator_mode = _mode_map.get(mode, mode)
    try:
        result = enhance_fn(audio=audio, mode=orchestrator_mode, language=language)
        entry = HistoryEntry(
            transcript=result.transcript,
            enhanced_prompt=result.enhanced_prompt,
            mode=result.mode,
        )
        history.append(entry)
        if on_history_updated:
            on_history_updated()
        if not cancel_event.is_set():
            if pill_ref[0]:
                pill_ref[0].show_done()
            inject_fn(result.enhanced_prompt, settings, target_hwnd=target_hwnd)
    except no_speech_cls:
        logger.info("No speech detected in audio")
        if pill_ref[0]:
            pill_ref[0].show_error("No speech detected", duration_ms=1500)
    except Exception as exc:
        import traceback
        traceback.print_exc()  # always visible in terminal regardless of logging setup
        logger.exception("enhance pipeline failed")
        if pill_ref[0]:
            pill_ref[0].show_error(f"Error: {exc}"[:40], duration_ms=2000)
    finally:
        start_record_event.clear()
        stop_record_event.clear()
        cancel_event.clear()


def main() -> None:
    # File logging — full tracebacks go to ~/.whisper_tray.log
    _log_path = Path.home() / ".whisper_tray.log"
    _fh = logging.FileHandler(str(_log_path), encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _root_logger = logging.getLogger("whisper_tray")
    _root_logger.setLevel(logging.DEBUG)
    if not _root_logger.handlers:
        _root_logger.addHandler(_fh)
        _root_logger.addHandler(logging.StreamHandler())

    root = tk.Tk()
    root.withdraw()  # hide root window — only the pill and dialogs are shown

    settings = load_settings()
    history = History()
    state = AppState(settings=settings, history=history)

    recorder = AudioRecorder(device=settings.get("microphone", "auto"))
    state.recorder = recorder

    executor = ThreadPoolExecutor(max_workers=1)
    nonlocal_pill: list[Pill | None] = [None]  # mutable cell so closures can reassign
    _active_recording = [False]  # one-shot guard: prevents re-firing every 50 ms

    def on_cancel() -> None:
        state.cancel_event.set()
        state.stop_record_event.set()
        recorder.cancel()
        if nonlocal_pill[0]:
            nonlocal_pill[0].hide()
        state.cancel_event.clear()
        state.stop_record_event.clear()
        state.start_record_event.clear()

    def on_stop() -> None:
        state.stop_record_event.set()

    def on_mode_change(mode: str) -> None:
        state.current_mode = mode

    def on_settings() -> None:
        def apply_new_settings(new_settings: dict) -> None:
            state.settings.update(new_settings)
            listener.stop()
            listener._hotkey = new_settings.get("hotkey", state.settings["hotkey"])
            listener._hotkey_mode = new_settings.get("hotkey_mode", state.settings["hotkey_mode"])
            listener.start()
            mic = new_settings.get("microphone", "auto")
            recorder._device = None if mic == "auto" else mic

        root.after(0, lambda: SettingsWindow(root, state.settings, apply_new_settings))

    def on_paste_last() -> None:
        last = history.last()
        if last:
            inject(last.enhanced_prompt, state.settings)

    def on_quit() -> None:
        listener.stop()
        tray.stop()
        executor.shutdown(wait=False)
        root.quit()

    tray = TrayIcon(
        app_state=state,
        on_settings=on_settings,
        on_paste_last=on_paste_last,
        on_quit=on_quit,
    )

    listener = HotkeyListener(
        state=state,
        hotkey=settings.get("hotkey", DEFAULTS["hotkey"]),
        hotkey_mode=settings.get("hotkey_mode", DEFAULTS["hotkey_mode"]),
    )

    def _create_pill() -> Pill:
        return Pill(
            root=root,
            on_cancel=on_cancel,
            on_stop=on_stop,
            on_mode_change=on_mode_change,
            initial_mode=state.current_mode,
            mode_lock=state.settings.get("mode_lock_enabled", False),
        )

    def _run_enhance(audio: bytes) -> None:
        _run_enhance_task(
            audio,
            mode=state.current_mode,
            language=state.settings.get("language", "auto"),
            history=history,
            cancel_event=state.cancel_event,
            start_record_event=state.start_record_event,
            stop_record_event=state.stop_record_event,
            pill_ref=nonlocal_pill,
            settings=state.settings,
            target_hwnd=state.target_hwnd,
            on_history_updated=tray.update_menu,
        )

    def poll() -> None:
        if state.start_record_event.is_set() and not state.stop_record_event.is_set():
            if not _active_recording[0]:
                _active_recording[0] = True
                try:
                    state.target_hwnd = ctypes.windll.user32.GetForegroundWindow()
                except Exception:
                    state.target_hwnd = 0
                if nonlocal_pill[0] is None:
                    nonlocal_pill[0] = _create_pill()
                tray.set_recording(True)
                nonlocal_pill[0].show_recording()
                recorder.start()

        if state.stop_record_event.is_set() and state.start_record_event.is_set():
            _active_recording[0] = False
            audio = recorder.stop()
            tray.set_recording(False)
            tray.set_processing()
            if nonlocal_pill[0]:
                nonlocal_pill[0].show_processing()
            state.stop_record_event.clear()
            state.start_record_event.clear()
            if not state.cancel_event.is_set() and audio:
                executor.submit(_run_enhance, audio)

        root.after(50, poll)

    def start_app() -> None:
        tray.start()
        listener.start()
        poll()
        if is_first_run():
            def on_wizard_done(new_settings: dict) -> None:
                state.settings.update(new_settings)
                listener.stop()
                listener._hotkey = new_settings["hotkey"]
                listener.start()
            FirstRunWizard(root, state.settings, on_wizard_done)

    root.after(0, start_app)
    root.mainloop()


if __name__ == "__main__":
    main()
