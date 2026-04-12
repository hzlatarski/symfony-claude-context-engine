"""Git intelligence: hotspots, co-change, decision mining, ownership.

Runs git log once, parses output, computes temporal-decay-weighted hotspot
scores, detects co-change partners, mines commit messages for architectural
decisions. Cached to knowledge/git-intel.json and invalidated on HEAD change.

Inspired by Repowise's algorithms but adapted for any git repo — no
language-specific AST required.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Analysis window — how far back we look
SINCE = "6 months ago"

# Temporal decay half-life in days
DECAY_TAU = 180.0

# Only count a file as co-change partner if they appeared together >= this many times
MIN_COCHANGE_COUNT = 2

# Automated commit patterns to skip from decision mining
_SKIP_COMMIT_RE = re.compile(r"^(Merge|Bump|chore|ci|style|build|release)\b", re.IGNORECASE)

# Decision classification patterns (in priority order)
_DECISION_PATTERNS = [
    ("extraction",  re.compile(r"\bextract", re.IGNORECASE)),
    ("migration",   re.compile(r"\bmigrat", re.IGNORECASE)),
    ("replacement", re.compile(r"\breplac", re.IGNORECASE)),
    ("removal",     re.compile(r"\b(remov|deprecat|delet)", re.IGNORECASE)),
    ("refactor",    re.compile(r"\brefactor|\brewrit|\brestructur", re.IGNORECASE)),
    ("introduction",re.compile(r"\bintroduc|\badd\b.*\b(module|service|system)", re.IGNORECASE)),
]


def _classify_commit(message: str) -> str | None:
    """Return a decision type if the commit looks like a decision, else None."""
    if _SKIP_COMMIT_RE.match(message):
        return None
    for label, rx in _DECISION_PATTERNS:
        if rx.search(message):
            return label
    return None


def _git_log_numstat(project_root: Path) -> str:
    """Run git log --numstat and return raw stdout."""
    return subprocess.check_output(
        [
            "git",
            "log",
            "--numstat",
            f"--since={SINCE}",
            "--format=__COMMIT__%H|%ae|%aI|%s",
            "--no-merges",
        ],
        cwd=str(project_root),
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _git_head(project_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(project_root),
        text=True,
    ).strip()[:12]


def _parse_git_log(raw: str) -> tuple[dict, list]:
    """Parse --numstat output.

    Returns:
        per_file: {file_path: {'commits': [(sha, date, age_days, msg, author), ...]}}
        commits:  [(sha, date, age_days, msg, author, [file1, file2, ...])]
    """
    now = datetime.now(timezone.utc)
    per_file: dict[str, dict] = defaultdict(lambda: {"commits": []})
    commits: list = []

    current_sha = None
    current_author = None
    current_date = None
    current_age = 0.0
    current_msg = ""
    current_files: list[str] = []

    def flush_commit():
        if current_sha and current_files:
            commits.append((current_sha, current_date, current_age, current_msg, current_author, list(current_files)))

    for line in raw.splitlines():
        if line.startswith("__COMMIT__"):
            flush_commit()
            header = line[len("__COMMIT__"):]
            parts = header.split("|", 3)
            if len(parts) != 4:
                continue
            current_sha, current_author, date_str, current_msg = parts
            try:
                current_date = datetime.fromisoformat(date_str)
                current_age = (now - current_date).total_seconds() / 86400.0
            except ValueError:
                current_date = now
                current_age = 0.0
            current_files = []
        elif line.strip() and current_sha:
            # numstat line: added\tdeleted\tpath
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                added = int(parts[0]) if parts[0] != "-" else 0
                deleted = int(parts[1]) if parts[1] != "-" else 0
            except ValueError:
                added = deleted = 0
            path = parts[2]
            current_files.append(path)
            per_file[path]["commits"].append(
                (current_sha, current_date, current_age, current_msg, current_author, added, deleted)
            )

    flush_commit()
    return dict(per_file), commits


def parse(project_root: Path) -> dict:
    """Run full git intelligence pass. Expensive — ~10-15 seconds."""
    head = _git_head(project_root)
    raw = _git_log_numstat(project_root)
    per_file, commits = _parse_git_log(raw)

    # Compute hotspot score per file: sum of exp(-age/tau) across commits
    hotspot_list: list[dict] = []
    for path, data in per_file.items():
        commits_total = len(data["commits"])
        score = 0.0
        added_90d = 0
        deleted_90d = 0
        commits_30d = 0
        commits_90d = 0
        authors: dict[str, int] = defaultdict(int)

        for sha, date, age, msg, author, added, deleted in data["commits"]:
            score += math.exp(-age / DECAY_TAU)
            if age <= 30:
                commits_30d += 1
            if age <= 90:
                commits_90d += 1
                added_90d += added
                deleted_90d += deleted
            authors[author] += 1

        primary_owner = max(authors.items(), key=lambda kv: kv[1])[0] if authors else ""
        # Derive a display name from email local part
        primary_owner_name = primary_owner.split("@")[0] if "@" in primary_owner else primary_owner

        hotspot_list.append({
            "file": path,
            "score": round(score, 4),
            "commits_total": commits_total,
            "commits_30d": commits_30d,
            "commits_90d": commits_90d,
            "lines_added_90d": added_90d,
            "lines_deleted_90d": deleted_90d,
            "primary_owner": primary_owner_name,
            "bus_factor": len(authors),
            "co_change_partners": [],  # filled below
        })

    hotspot_list.sort(key=lambda h: h["score"], reverse=True)

    # Co-change detection: for each commit, every pair of files is a co-change
    pair_counts: dict[tuple[str, str], float] = defaultdict(float)
    for sha, date, age, msg, author, files in commits:
        # Skip commits that touch too many files (likely cleanups / refactors)
        if len(files) > 15:
            continue
        weight = math.exp(-age / DECAY_TAU)
        for i, a in enumerate(files):
            for b in files[i + 1:]:
                key = (a, b) if a < b else (b, a)
                pair_counts[key] += weight

    # Attach co-change partners to each file's hotspot entry (top 5)
    file_to_partners: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (a, b), weight in pair_counts.items():
        if weight < MIN_COCHANGE_COUNT:
            continue
        file_to_partners[a].append((b, weight))
        file_to_partners[b].append((a, weight))

    file_to_hotspot = {h["file"]: h for h in hotspot_list}
    for path, partners in file_to_partners.items():
        if path not in file_to_hotspot:
            continue
        partners.sort(key=lambda kv: kv[1], reverse=True)
        file_to_hotspot[path]["co_change_partners"] = [
            {"file": p, "score": round(s, 2)} for p, s in partners[:5]
        ]

    # Mine decisions from commit messages
    decisions: list[dict] = []
    seen_shas: set[str] = set()
    for sha, date, age, msg, author, files in commits:
        if sha in seen_shas:
            continue
        kind = _classify_commit(msg)
        if not kind:
            continue
        seen_shas.add(sha)
        decisions.append({
            "type": kind,
            "commit": sha[:12],
            "date": date.isoformat() if date else "",
            "message": msg,
            "files": files[:20],  # cap for payload size
        })

    # Stats
    stats = {
        "total_files_tracked": len(per_file),
        "hotspot_count": len(hotspot_list),
        "decision_count": len(decisions),
        "analysis_window_days": 180,
    }

    return {
        "head": head,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hotspots": hotspot_list,
        "decisions": decisions,
        "stats": stats,
    }


def load_or_parse(project_root: Path, cache_file: Path | None = None) -> dict:
    """Return cached intel if HEAD matches, else rebuild and write cache."""
    if cache_file is None:
        cache_file = project_root / "knowledge" / "git-intel.json"

    head = _git_head(project_root)

    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("head") == head:
                return cached
        except (json.JSONDecodeError, OSError):
            pass

    result = parse(project_root)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def summary(project_root: Path) -> str:
    """Fast summary — read the cache file if it exists, don't rebuild."""
    cache_file = project_root / "knowledge" / "git-intel.json"
    if not cache_file.exists():
        return "git intel not yet built — run get_hotspots() to generate"
    try:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "git intel cache unreadable"

    top_hotspots = cached.get("hotspots", [])[:3]
    if not top_hotspots:
        return "no hotspots"

    parts = []
    for h in top_hotspots:
        short = h["file"].split("/")[-1]
        parts.append(f"{short} ({h['commits_total']}c)")
    return "Hotspots: " + ", ".join(parts)
