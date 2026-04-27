"""Pull the latest memory-compiler from the remote and reinstall.

Performs a non-destructive upgrade:

    1. Stash any uncommitted local changes (so a tinkering user doesn't
       lose them — they get a hint to ``git stash pop`` afterwards).
    2. ``git fetch origin`` + ``git reset --hard origin/main`` to advance
       the working tree to the remote tip.
    3. Run any per-version migration scripts under ``scripts/migrations/``
       whose version is between the old and new VERSION.
    4. Run ``uv sync`` to pick up new Python deps.
    5. Write ``~/.memory-compiler/just-upgraded-from`` with the prior
       version. The next session-start probe surfaces this as
       ``JUST_UPGRADED`` so the agent renders the "what's new" block.
    6. Clear the cached update check so the next probe re-runs.

All output goes to stdout in human-readable form. The exit code reflects
success (0) or failure (non-zero); the caller (the skill) reads the
exit code to decide whether to render "what's new" or to fail loudly.

Usage:
    uv run python scripts/upgrade.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from check_update import _state_dir  # reuse the same path resolution


VERSION_FILE = _ROOT / "VERSION"
MIGRATIONS_DIR = _ROOT / "scripts" / "migrations"


def _read_version(file: Path) -> str:
    try:
        return file.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


def _run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> tuple[int, str, str]:
    """Run a subprocess, returning (rc, stdout, stderr)."""
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(
            f"Command failed (rc={proc.returncode}): {' '.join(cmd)}"
        )
    return proc.returncode, proc.stdout, proc.stderr


def _ensure_clean_or_stash() -> str | None:
    """Stash uncommitted changes so the upgrade can reset --hard cleanly.

    Returns the stash message if anything was stashed, None otherwise.
    """
    rc, out, _ = _run(["git", "status", "--porcelain"], cwd=_ROOT)
    if not out.strip():
        return None
    stash_msg = "memory-compiler-upgrade: pre-upgrade stash"
    _run(["git", "stash", "push", "-u", "-m", stash_msg], cwd=_ROOT)
    return stash_msg


def _versions_between(old: str, new: str) -> list[Path]:
    """Return migration scripts whose version is in (old, new].

    Filenames are ``v<X>.<Y>.<Z>.sh`` or ``v<X>.<Y>.<Z>.py``. Sorted by
    natural version order (split-on-dot, integer compare).
    """
    if not MIGRATIONS_DIR.exists():
        return []

    def _parse(name: str) -> tuple[int, ...] | None:
        if not name.startswith("v"):
            return None
        body = name[1:].rsplit(".", 1)[0]
        parts = body.split(".")
        try:
            return tuple(int(p) for p in parts)
        except ValueError:
            return None

    old_t = _parse(f"v{old}.x") or ()
    new_t = _parse(f"v{new}.x") or ()
    out: list[tuple[tuple[int, ...], Path]] = []
    for entry in MIGRATIONS_DIR.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix not in (".sh", ".py"):
            continue
        v = _parse(entry.name)
        if v is None:
            continue
        # Pad to common length for comparison
        n = max(len(v), len(old_t), len(new_t))
        vp = v + (0,) * (n - len(v))
        op = old_t + (0,) * (n - len(old_t))
        np = new_t + (0,) * (n - len(new_t))
        if op < vp <= np:
            out.append((vp, entry))
    out.sort(key=lambda kv: kv[0])
    return [p for _, p in out]


def _run_migrations(old: str, new: str) -> None:
    scripts = _versions_between(old, new)
    if not scripts:
        return
    print(f"Running {len(scripts)} migration(s) for {old} → {new}…")
    for script in scripts:
        print(f"  • {script.name}")
        if script.suffix == ".sh":
            rc, _, _ = _run(["bash", str(script)], cwd=_ROOT, check=False)
        else:
            rc, _, _ = _run([sys.executable, str(script)], cwd=_ROOT, check=False)
        if rc != 0:
            print(f"    Warning: migration {script.name} exited {rc} (non-fatal)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upgrade the memory compiler to the latest origin/main.",
    )
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Skip the post-fetch `uv sync` step. Use only if you already ran it.",
    )
    parser.add_argument(
        "--remote",
        default="origin",
        help="Git remote to pull from (default: origin).",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch to reset to (default: main).",
    )
    args = parser.parse_args()

    if not (_ROOT / ".git").exists():
        sys.stderr.write(
            "Error: memory-compiler is not a git checkout — `git clone` install required for self-upgrade.\n"
        )
        return 2

    old_version = _read_version(VERSION_FILE)
    print(f"Memory-compiler upgrade starting (current version: {old_version})")

    stash_msg = _ensure_clean_or_stash()
    if stash_msg:
        print(f"  Stashed local changes as: {stash_msg!r}")

    print(f"  Fetching {args.remote}/{args.branch}…")
    _run(["git", "fetch", args.remote, args.branch], cwd=_ROOT)

    print(f"  Resetting working tree to {args.remote}/{args.branch}…")
    _run(["git", "reset", "--hard", f"{args.remote}/{args.branch}"], cwd=_ROOT)

    new_version = _read_version(VERSION_FILE)
    print(f"  Pulled version: {new_version}")

    if not args.no_deps:
        # uv sync picks up new pyproject.toml deps. We don't pass --frozen
        # because uv.lock might be newer post-reset.
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)  # let uv pick its own venv
        proc = subprocess.run(
            ["uv", "sync"],
            cwd=str(_ROOT),
            env=env,
            text=True,
        )
        if proc.returncode != 0:
            sys.stderr.write(
                "uv sync exited non-zero — dependencies may be stale. "
                "Run `uv sync` manually inside .claude/memory-compiler/ to recover.\n"
            )

    _run_migrations(old_version, new_version)

    # State markers — JUST_UPGRADED on next probe + clear cache & snooze.
    state = _state_dir()
    state.mkdir(parents=True, exist_ok=True)
    (state / "just-upgraded-from").write_text(old_version, encoding="utf-8")
    for fname in ("last-update-check", "update-snoozed"):
        try:
            (state / fname).unlink()
        except FileNotFoundError:
            pass

    print(f"\nUpgrade complete: {old_version} → {new_version}")
    if stash_msg:
        print(
            f"  Note: your pre-upgrade local changes were stashed. "
            f"Run `git -C \"{_ROOT}\" stash pop` to restore them."
        )
    print("  Restart Claude Code so the MCP servers pick up the new code.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
