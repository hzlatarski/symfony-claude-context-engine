"""Lightweight update-check probe for the memory compiler.

Fetches ``https://raw.githubusercontent.com/hzlatarski/symfony-claude-context-engine/main/VERSION``,
compares against the local ``VERSION`` file, and prints one of:

    JUST_UPGRADED <old> <new>     — marker found from a recent upgrade
    UPGRADE_AVAILABLE <old> <new> — remote VERSION newer than local
    (no output)                   — up to date / snoozed / disabled / cached / offline

Designed to be parsed by the SessionStart hook. Single-line output keeps
the parser trivial; silence on any failure mode means the hook can call
this on every session start without ever blocking the agent.

State directory: ``~/.memory-compiler/``
    last-update-check    — touch file (mtime is the cache key, 6h TTL)
    just-upgraded-from   — written by upgrade.py with the previous version
    update-snoozed       — `<version> <level> <epoch>` for escalating backoff
    config.json          — `{auto_upgrade, update_check}` flags

Env overrides (testing + advanced):
    MEMORY_COMPILER_REMOTE_VERSION_URL  — alternate raw VERSION URL
    MEMORY_COMPILER_STATE_DIR           — alternate state dir
    MEMORY_COMPILER_UPDATE_CHECK        — '0' to disable
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Bootstrap path resolution — same trick as the MCP servers.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Default cache TTL — 6h. Long enough that we don't spam GitHub on every
# session start, short enough that a freshly-published release is picked
# up the same day. ``--force`` busts the cache.
CACHE_TTL_SECONDS = 6 * 3600

# Default request timeout — keep tight so a slow GitHub doesn't stall the
# SessionStart hook. The hook's outer timeout (~15s) is a backstop, but
# we want to be a good citizen here.
REQUEST_TIMEOUT_SECONDS = 5.0

DEFAULT_REMOTE_URL = (
    "https://raw.githubusercontent.com/hzlatarski/"
    "symfony-claude-context-engine/main/VERSION"
)


def _state_dir() -> Path:
    override = os.environ.get("MEMORY_COMPILER_STATE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".memory-compiler"


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _read_config() -> dict:
    cfg_file = _state_dir() / "config.json"
    if not cfg_file.exists():
        return {}
    try:
        return json.loads(cfg_file.read_text(encoding="utf-8")) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _update_check_enabled() -> bool:
    """Two layers: env var (highest priority) then config.json."""
    env = os.environ.get("MEMORY_COMPILER_UPDATE_CHECK", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    cfg = _read_config()
    val = cfg.get("update_check")
    if val is False:
        return False
    return True


def _parse_version(value: str) -> tuple[int, ...] | None:
    """Parse a dotted-int version into a comparable tuple. None on failure."""
    if not value:
        return None
    parts = value.strip().split(".")
    out: list[int] = []
    for part in parts:
        if not part.isdigit():
            return None
        out.append(int(part))
    return tuple(out) if out else None


def _is_newer(remote: str, local: str) -> bool:
    rv = _parse_version(remote)
    lv = _parse_version(local)
    if rv is None or lv is None:
        # Fall back to string inequality if either side isn't dotted-int —
        # still surfaces an upgrade prompt rather than swallowing it.
        return remote != local
    # Pad shorter tuple with zeros so 0.1 and 0.1.0 compare equal.
    n = max(len(rv), len(lv))
    rv = rv + (0,) * (n - len(rv))
    lv = lv + (0,) * (n - len(lv))
    return rv > lv


def _check_snooze(remote_version: str, snooze_file: Path) -> bool:
    """Return True if currently snoozed for this remote version.

    File format: ``<version> <level> <epoch>``. Level durations:
    1=24h, 2=48h, 3+=7d. A new remote version resets the snooze
    automatically (caller writes the new version on ``Not now``).
    """
    if not snooze_file.exists():
        return False
    try:
        raw = snooze_file.read_text(encoding="utf-8").strip().split()
    except OSError:
        return False
    if len(raw) < 3:
        return False
    snoozed_ver, level_s, epoch_s = raw[0], raw[1], raw[2]
    if snoozed_ver != remote_version:
        return False
    try:
        level = int(level_s)
        epoch = int(epoch_s)
    except ValueError:
        return False
    durations = {1: 24 * 3600, 2: 48 * 3600}
    duration = durations.get(level, 7 * 24 * 3600)
    return (time.time() - epoch) < duration


def _fetch_remote_version(url: str) -> str | None:
    """Fetch the remote VERSION file. Returns None on any failure (offline,
    404, malformed) — silence is the right default at session start.

    ``file://`` URLs are supported as a convenience for offline mirrors,
    air-gapped environments, and the smoke-test suite. Everything else
    is delegated to httpx.
    """
    if url.startswith("file://"):
        try:
            from urllib.parse import urlparse
            from urllib.request import url2pathname

            path = url2pathname(urlparse(url).path)
            text = Path(path).read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return None
        return text if (text and len(text) <= 64) else None

    try:
        import httpx
    except ImportError:
        return None
    try:
        resp = httpx.get(url, timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    text = resp.text.strip()
    # Sanity-cap: a real VERSION file is <30 chars. Anything larger is
    # almost certainly an error page slipping through.
    if not text or len(text) > 64:
        return None
    return text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe the remote VERSION and emit one update-status line.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass the 6h cache and any active snooze.",
    )
    parser.add_argument(
        "--print-current",
        action="store_true",
        help="Print the local VERSION instead of running a check (used by upgrade.py).",
    )
    args = parser.parse_args()

    local_version = _read_text(_ROOT / "VERSION")

    if args.print_current:
        print(local_version or "unknown")
        return 0

    if not _update_check_enabled():
        return 0

    state = _state_dir()
    state.mkdir(parents=True, exist_ok=True)
    cache_file = state / "last-update-check"
    snooze_file = state / "update-snoozed"
    marker_file = state / "just-upgraded-from"

    # JUST_UPGRADED takes priority over a fresh check — the agent should
    # surface "what's new" before considering further upgrades.
    if marker_file.exists() and local_version:
        prev = _read_text(marker_file)
        if prev and prev != local_version:
            print(f"JUST_UPGRADED {prev} {local_version}")
            # Marker is consumed by the skill's "show what's new" step;
            # we leave it for the skill to delete after rendering.
            return 0
        # Stale marker (same version) — clean up so it doesn't linger.
        try:
            marker_file.unlink()
        except FileNotFoundError:
            pass

    # Cache check — run at most once per CACHE_TTL_SECONDS unless --force.
    if not args.force and cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            return 0

    if args.force:
        try:
            snooze_file.unlink()
        except FileNotFoundError:
            pass

    if not local_version:
        # No local VERSION — first-time install hasn't happened yet.
        # Touch the cache so we don't hammer GitHub but stay silent.
        cache_file.touch()
        return 0

    remote_url = os.environ.get("MEMORY_COMPILER_REMOTE_VERSION_URL", DEFAULT_REMOTE_URL)
    remote_version = _fetch_remote_version(remote_url)
    cache_file.touch()  # always touch — failure shouldn't trigger a retry storm

    if remote_version is None:
        return 0  # offline or remote unavailable — silent

    if not _is_newer(remote_version, local_version):
        return 0  # up to date

    if _check_snooze(remote_version, snooze_file):
        return 0  # snoozed

    print(f"UPGRADE_AVAILABLE {local_version} {remote_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
