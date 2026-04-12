"""
Check Claude Code plugins for available updates.

Reads ~/.claude/plugins/installed_plugins.json and the local marketplace clone
at ~/.claude/plugins/marketplaces/<marketplace>/.claude-plugin/marketplace.json,
then queries the GitHub API for each plugin's upstream commit SHA and compares.

No auth required — uses the unauthenticated GitHub API (60 req/hour limit, fine
for ~20 plugins). Prints a color-free table that's safe to pipe.

Usage:
    python check-plugin-updates.py

Exit codes:
    0 — everything up to date
    1 — at least one plugin has an update available
    2 — error reading local files or unexpected format
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PLUGINS_DIR = Path.home() / ".claude" / "plugins"
INSTALLED = PLUGINS_DIR / "installed_plugins.json"
MARKETPLACES_DIR = PLUGINS_DIR / "marketplaces"


def gh_api(path: str) -> dict | None:
    """Fetch a GitHub API path. Returns parsed JSON or None on failure."""
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        headers={"User-Agent": "claude-plugin-update-checker", "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"  ! HTTP {e.code} fetching {path}", file=sys.stderr)
    except Exception as e:
        print(f"  ! {type(e).__name__} fetching {path}: {e}", file=sys.stderr)
    return None


def latest_sha(owner_repo: str, ref: str = "main") -> str | None:
    """Get the latest commit SHA for a GitHub repo. Falls back to master if main 404s."""
    data = gh_api(f"/repos/{owner_repo}/commits/{ref}")
    if data is None and ref == "main":
        data = gh_api(f"/repos/{owner_repo}/commits/master")
    return data.get("sha") if isinstance(data, dict) else None


def parse_github_url(url: str) -> str | None:
    """Extract owner/repo from a GitHub URL or shorthand. Returns None if not GitHub."""
    if not url:
        return None
    if "github.com" in url:
        return url.split("github.com/", 1)[1].removesuffix(".git").rstrip("/")
    if url.count("/") == 1 and not url.startswith(("http", "git@")):
        return url  # bare owner/repo shorthand
    return None


def load_marketplace_manifest(marketplace_id: str) -> tuple[dict | None, float | None]:
    """Load a local marketplace manifest. Returns ({plugin-name: entry}, mtime)."""
    manifest_file = MARKETPLACES_DIR / marketplace_id / ".claude-plugin" / "marketplace.json"
    if not manifest_file.is_file():
        return None, None
    with manifest_file.open() as f:
        m = json.load(f)
    return {p["name"]: p for p in m.get("plugins", [])}, manifest_file.stat().st_mtime


def parse_iso_timestamp(ts: str | None) -> float | None:
    """Parse an ISO 8601 timestamp to Unix epoch seconds. Returns None if invalid."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None


def main() -> int:
    if not INSTALLED.is_file():
        print(f"ERROR: {INSTALLED} not found", file=sys.stderr)
        return 2

    with INSTALLED.open() as f:
        installed = json.load(f).get("plugins", {})

    # Cache: marketplace_id -> {plugin name: marketplace entry}
    marketplace_cache: dict[str, dict] = {}
    # Cache: marketplace_id -> mtime of the manifest file
    marketplace_mtime_cache: dict[str, float | None] = {}
    # Cache: marketplace_id -> latest commit SHA (for inline plugins)
    marketplace_head_cache: dict[str, str | None] = {}
    # Cache: owner/repo -> latest SHA
    repo_head_cache: dict[str, str | None] = {}

    rows: list[tuple[str, str, str, str, str]] = []
    updates_available = 0

    for plugin_id in sorted(installed):
        entries = installed[plugin_id]
        if not entries:
            continue
        entry = entries[0]
        name, _, marketplace_id = plugin_id.rpartition("@")
        current_sha = (entry.get("gitCommitSha") or "").lower()
        installed_at = parse_iso_timestamp(entry.get("lastUpdated") or entry.get("installedAt"))

        # Load marketplace manifest on first hit
        if marketplace_id not in marketplace_cache:
            mf, mtime = load_marketplace_manifest(marketplace_id)
            marketplace_cache[marketplace_id] = mf or {}
            marketplace_mtime_cache[marketplace_id] = mtime

        mp_entry = marketplace_cache[marketplace_id].get(name)
        if mp_entry is None:
            rows.append((name, current_sha[:12] or "-", "?", "not-in-marketplace", "?"))
            continue

        source = mp_entry.get("source", "")
        latest: str | None
        origin: str
        status: str | None = None

        if isinstance(source, str):
            # Inline path — plugin lives inside the marketplace repo itself.
            # Its effective "commit" is the marketplace repo's latest commit.
            if marketplace_id not in marketplace_head_cache:
                km = PLUGINS_DIR / "known_marketplaces.json"
                mp_repo = None
                if km.is_file():
                    with km.open() as f:
                        kmdata = json.load(f)
                    src = kmdata.get(marketplace_id, {}).get("source", {})
                    mp_repo = parse_github_url(src.get("url") or src.get("repo") or "")
                marketplace_head_cache[marketplace_id] = latest_sha(mp_repo) if mp_repo else None
            latest = marketplace_head_cache[marketplace_id]
            origin = f"inline ({marketplace_id})"

            # Inline plugins installed via CLI don't record a gitCommitSha.
            # Treat as up-to-date if the install happened after the marketplace manifest was last refreshed.
            if not current_sha:
                mp_mtime = marketplace_mtime_cache.get(marketplace_id)
                if installed_at and mp_mtime and installed_at >= mp_mtime - 60:
                    status = "UP TO DATE (no sha)"

        elif isinstance(source, dict):
            # Respect a pinned SHA in the marketplace entry — it's the authoritative target.
            pinned = source.get("sha")
            if pinned:
                latest = pinned
                origin = f"{parse_github_url(source.get('url') or source.get('repo') or '') or '?'} [pinned]"
            else:
                url = source.get("url") or source.get("repo") or ""
                repo = parse_github_url(url)
                if repo is None:
                    rows.append((name, current_sha[:12], "?", f"non-github: {url[:40]}", "?"))
                    continue
                if repo not in repo_head_cache:
                    repo_head_cache[repo] = latest_sha(repo)
                latest = repo_head_cache[repo]
                origin = repo

        else:
            rows.append((name, current_sha[:12] or "-", "?", "unknown-source", "?"))
            continue

        if status is None:
            if latest is None:
                status = "fetch-failed"
            elif current_sha and latest.lower().startswith(current_sha[:12]):
                status = "UP TO DATE"
            elif not current_sha:
                # No recorded SHA and we couldn't prove freshness via timestamp — unknown
                status = "UNKNOWN (no sha)"
            else:
                status = "UPDATE AVAILABLE"
                updates_available += 1

        rows.append((name, current_sha[:12] or "-", latest[:12] if latest else "?", origin, status))

    # Render
    col_name = max(len(r[0]) for r in rows) if rows else 6
    col_origin = min(max(len(r[3]) for r in rows) if rows else 6, 55)
    header = f"{'PLUGIN':<{col_name}}  {'CURRENT':<12}  {'LATEST':<12}  {'STATUS':<18}  SOURCE"
    print(header)
    print("-" * len(header))
    for name, cur, lat, origin, status in rows:
        print(f"{name:<{col_name}}  {cur:<12}  {lat:<12}  {status:<18}  {origin[:col_origin]}")

    print()
    total = len(rows)
    up_to_date = sum(1 for r in rows if r[4].startswith("UP TO DATE"))
    print(f"Summary: {up_to_date}/{total} up to date, {updates_available} updates available")

    return 1 if updates_available else 0


if __name__ == "__main__":
    sys.exit(main())
