"""Cut a memory-compiler release: bump VERSION, update CHANGELOG, commit, tag, push.

The self-update loop is driven entirely by the ``VERSION`` file on the remote
``main`` branch (``check_update.py`` compares the remote raw VERSION against the
local one). If VERSION never moves, sibling projects never see an upgrade — the
mechanism is correct but starved. This helper makes bumping the version a single
deliberate action so it stops being forgotten.

What it does, in order:

    1. Refuse unless the working tree is clean and we're on the release branch.
    2. Compute the new version from the bump level (major/minor/patch) or --set.
    3. Gather commit subjects since the last release as CHANGELOG seed bullets.
    4. Write VERSION and prepend a ``## [x.y.z] — <date>`` section to CHANGELOG.
    5. ``git commit`` the two files as ``chore(release): x.y.z``.
    6. ``git tag v<x.y.z>`` (annotated) unless --no-tag.
    7. ``git push`` the branch (and tag) unless --no-push.

Usage:
    uv run python scripts/release.py minor
    uv run python scripts/release.py patch --no-push        # local-only, review then push
    uv run python scripts/release.py --set 1.0.0
    uv run python scripts/release.py minor --dry-run        # preview, touch nothing

After pushing, the next session in any sibling project (CarPro, Sentinel AI, …)
will see the new VERSION and surface the ``/memory-compiler-upgrade`` prompt.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

VERSION_FILE = _ROOT / "VERSION"
CHANGELOG_FILE = _ROOT / "CHANGELOG.md"

# Em dash to match the existing CHANGELOG header style: "## [0.1.0] — 2026-04-27".
_HEADER_DASH = "—"
# Commit subjects we never want as changelog bullets.
_SKIP_PREFIXES = ("chore(release):", "Merge ", "merge ")


# -----------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_release.py)
# -----------------------------------------------------------------------------


def bump_version(current: str, level: str) -> str:
    """Return the next dotted version. ``level`` ∈ {major, minor, patch}.

    Normalises to three segments (x.y.z). Raises ValueError on a non-numeric
    current version or an unknown level.
    """
    parts = current.strip().split(".")
    try:
        nums = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Cannot bump non-numeric version: {current!r}") from exc
    while len(nums) < 3:
        nums.append(0)
    major, minor, patch = nums[0], nums[1], nums[2]
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    if level == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"Unknown bump level: {level!r}")


def is_valid_version(value: str) -> bool:
    parts = value.strip().split(".")
    return len(parts) >= 1 and all(p.isdigit() for p in parts)


def build_changelog_section(version: str, today: str, bullets: list[str]) -> str:
    """Render a Keep-a-Changelog section. Falls back to a stub when there are
    no seed bullets, so the section is always editable-but-valid."""
    lines = [f"## [{version}] {_HEADER_DASH} {today}", ""]
    if bullets:
        lines.append("### Changed")
        lines.append("")
        lines.extend(f"- {b}" for b in bullets)
    else:
        lines.append("_Describe this release. (No commits were found since the "
                     "last release to seed bullets.)_")
    lines.append("")
    return "\n".join(lines)


def insert_changelog_section(changelog: str, section: str) -> str:
    """Insert ``section`` immediately before the first existing ``## [`` entry.

    If no prior entry exists, append after the file's preamble. The returned
    text always has the new section above all older ones (reverse-chronological).
    """
    marker = "\n## ["
    idx = changelog.find(marker)
    if idx == -1:
        # No prior versioned section — append at end with a separating blank line.
        sep = "" if changelog.endswith("\n\n") else ("\n" if changelog.endswith("\n") else "\n\n")
        return changelog + sep + section.rstrip() + "\n"
    head = changelog[: idx + 1]   # keep the leading newline that precedes "## ["
    tail = changelog[idx + 1:]
    return head + section.rstrip() + "\n\n" + tail


# -----------------------------------------------------------------------------
# Git plumbing
# -----------------------------------------------------------------------------


def _run(cmd: list[str], *, check: bool = True) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd, cwd=str(_ROOT), capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    if check and proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"Command failed (rc={proc.returncode}): {' '.join(cmd)}")
    return proc.returncode, proc.stdout, proc.stderr


def _current_branch() -> str:
    _, out, _ = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return out.strip()


def _working_tree_dirty() -> bool:
    _, out, _ = _run(["git", "status", "--porcelain"])
    return bool(out.strip())


def _last_release_ref() -> str | None:
    """Best-effort start point for the changelog range: the most recent tag,
    else the last commit that touched VERSION. None ⇒ use full history."""
    rc, out, _ = _run(["git", "describe", "--tags", "--abbrev=0"], check=False)
    if rc == 0 and out.strip():
        return out.strip()
    rc, out, _ = _run(["git", "log", "-1", "--format=%H", "--", "VERSION"], check=False)
    if rc == 0 and out.strip():
        return out.strip()
    return None


def _seed_bullets(since_ref: str | None) -> list[str]:
    rng = f"{since_ref}..HEAD" if since_ref else "HEAD"
    rc, out, _ = _run(["git", "log", rng, "--no-merges", "--pretty=%s"], check=False)
    if rc != 0:
        return []
    bullets: list[str] = []
    for line in out.splitlines():
        subj = line.strip()
        if not subj or subj.startswith(_SKIP_PREFIXES):
            continue
        bullets.append(subj)
    return bullets


def main() -> int:
    parser = argparse.ArgumentParser(description="Cut a memory-compiler release.")
    parser.add_argument("level", nargs="?", choices=["major", "minor", "patch"],
                        help="Semver bump level. Omit when using --set.")
    parser.add_argument("--set", dest="explicit", metavar="X.Y.Z",
                        help="Set an explicit version instead of bumping a level.")
    parser.add_argument("--no-tag", action="store_true", help="Do not create a git tag.")
    parser.add_argument("--no-push", action="store_true",
                        help="Commit (and tag) locally but do not push.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without writing or running git.")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="main")
    args = parser.parse_args()

    if not args.level and not args.explicit:
        parser.error("provide a bump level (major|minor|patch) or --set X.Y.Z")
    if args.level and args.explicit:
        parser.error("use either a bump level or --set, not both")

    if not (_ROOT / ".git").exists():
        sys.stderr.write("Error: not a git checkout — cannot cut a release here.\n")
        return 2

    current = VERSION_FILE.read_text(encoding="utf-8").strip()
    if args.explicit:
        if not is_valid_version(args.explicit):
            parser.error(f"--set value is not a dotted-int version: {args.explicit!r}")
        new_version = args.explicit.strip()
    else:
        new_version = bump_version(current, args.level)

    today = date.today().isoformat()
    since = _last_release_ref()
    bullets = _seed_bullets(since)
    section = build_changelog_section(new_version, today, bullets)

    print(f"Release: {current} -> {new_version}  (branch {args.branch}, {len(bullets)} seed bullet(s))")
    print("\n--- CHANGELOG section to be inserted ---")
    print(section)
    print("----------------------------------------")

    if args.dry_run:
        print("Dry run — no files written, no git commands run.")
        return 0

    # Safety gates — only after dry-run preview so users can always inspect first.
    branch = _current_branch()
    if branch != args.branch:
        sys.stderr.write(
            f"Error: on branch {branch!r}, expected {args.branch!r}. "
            f"Checkout {args.branch} or pass --branch.\n"
        )
        return 3
    if _working_tree_dirty():
        sys.stderr.write(
            "Error: working tree is dirty. Commit or stash everything except the "
            "release bump first — this helper only commits VERSION + CHANGELOG.\n"
        )
        return 4

    VERSION_FILE.write_text(new_version + "\n", encoding="utf-8")
    CHANGELOG_FILE.write_text(
        insert_changelog_section(CHANGELOG_FILE.read_text(encoding="utf-8"), section),
        encoding="utf-8",
    )

    _run(["git", "add", "VERSION", "CHANGELOG.md"])
    _run(["git", "commit", "-m", f"chore(release): {new_version}"])
    print(f"  Committed chore(release): {new_version}")

    tag = f"v{new_version}"
    if not args.no_tag:
        _run(["git", "tag", "-a", tag, "-m", f"memory-compiler {tag}"])
        print(f"  Tagged {tag}")

    if args.no_push:
        push_hint = f"git push {args.remote} {args.branch}" + ("" if args.no_tag else f" && git push {args.remote} {tag}")
        print(f"\nLocal release ready. Push when satisfied:\n  {push_hint}")
        return 0

    _run(["git", "push", args.remote, args.branch])
    print(f"  Pushed {args.branch} to {args.remote}")
    if not args.no_tag:
        _run(["git", "push", args.remote, tag])
        print(f"  Pushed tag {tag}")

    print(
        f"\nReleased {new_version}. Sibling projects will see UPGRADE_AVAILABLE on "
        f"their next session (or run `check_update.py --force` to verify now)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
