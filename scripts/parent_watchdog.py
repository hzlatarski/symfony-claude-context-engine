"""Parent-process watchdog: exit when the Claude Code parent dies.

Without this, MCP servers stay alive forever when their VS Code / Claude Code
parent crashes, is killed, or simply closes without cleanly shutting down its
stdio pipes. On Windows in particular, stdin EOF doesn't reliably propagate
through the ``uv → python`` wrapper chain, so the FastMCP runtime never sees a
shutdown signal. Over weeks of normal use, dozens of orphaned servers
accumulate, each holding a ChromaDB SQLite handle and ~64 MB RAM. New sessions
then contend with the orphans for those locks and the next MCP tool call
stalls indefinitely.

The watchdog records the "owner" PID at startup and polls every few seconds.
When the owner exits we call ``os._exit(0)`` to terminate unconditionally,
bypassing FastMCP's shutdown path which may itself be wedged.

Owner resolution:
    Our immediate parent is usually ``uv.exe`` (because servers launch via
    ``uv run --directory ... python scripts/mcp_server.py``), and uv's death
    is not a useful signal — uv stays alive as long as we do. Walk up the
    process tree past any ``uv``/``python`` ancestors to find the real
    spawner (Claude Code / VS Code). If psutil is unavailable, fall back
    to the immediate parent — better than nothing.
"""
from __future__ import annotations

import os
import sys
import threading
import time

_POLL_SECONDS = 5.0
_WRAPPER_NAMES = frozenset({
    "uv.exe", "uv",
    "python.exe", "python", "python3", "python3.exe",
    "pythonw.exe", "pythonw",
})


def _resolve_owner_pid() -> int:
    """Walk up the process tree past uv/python wrappers to the real spawner.

    Returns 0 when we can't make a confident determination — callers treat
    that as "don't watch" rather than guessing.
    """
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return os.getppid()

    try:
        for ancestor in psutil.Process(os.getpid()).parents():
            name = (ancestor.name() or "").lower()
            if name in _WRAPPER_NAMES:
                continue
            return ancestor.pid
        return os.getppid()
    except (psutil.Error, OSError):
        return os.getppid()


def _is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil  # type: ignore[import-not-found]
        return psutil.pid_exists(pid)
    except ImportError:
        pass

    if sys.platform == "win32":
        return _is_alive_windows(pid)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _is_alive_windows(pid: int) -> bool:
    """Stdlib-only Windows check via OpenProcess + GetExitCodeProcess."""
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259

    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _watch(pid: int) -> None:
    while True:
        time.sleep(_POLL_SECONDS)
        if not _is_alive(pid):
            # Parent gone — terminate hard. Avoid FastMCP's shutdown path
            # because it's the thing that's been failing to detect EOF.
            os._exit(0)


_started = False


def start() -> None:
    """Spawn the watchdog daemon thread. Idempotent — safe to call twice."""
    global _started
    if _started:
        return
    pid = _resolve_owner_pid()
    if pid <= 0:
        return
    threading.Thread(target=_watch, args=(pid,), daemon=True, name="parent-watchdog").start()
    _started = True
