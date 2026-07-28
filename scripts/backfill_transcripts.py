"""Archive and index historical Claude Code transcripts for this project."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from config import PROJECT_ROOT
from transcript import archive_transcript, embed_transcript_file


def _encoded_project_name(project_root: Path) -> str:
    encoded = str(project_root).replace(":", "-").replace("\\", "-").replace("/", "-")
    return encoded[:1].lower() + encoded[1:]


def default_source_dir() -> Path:
    projects_dir = Path.home() / ".claude" / "projects"
    expected = _encoded_project_name(PROJECT_ROOT).lower()
    if projects_dir.exists():
        for candidate in projects_dir.iterdir():
            if candidate.is_dir() and candidate.name.lower() == expected:
                return candidate
    raise FileNotFoundError(
        f"Claude transcript directory not found for {PROJECT_ROOT} under {projects_dir}"
    )


def backfill(
    source_dir: Path,
    session_ids: set[str] | None = None,
) -> dict[str, Any]:
    transcripts = sorted(source_dir.glob("*.jsonl"))
    if session_ids is not None:
        by_id = {path.stem: path for path in transcripts}
        missing = session_ids - by_id.keys()
        if missing:
            raise FileNotFoundError(
                f"Transcript session(s) not found: {', '.join(sorted(missing))}"
            )
        transcripts = [by_id[session_id] for session_id in sorted(session_ids)]

    chunks = 0
    sessions = 0
    failures: list[dict[str, str]] = []
    for source in transcripts:
        try:
            archive = archive_transcript(source, source.stem)
            chunks += embed_transcript_file(archive)
        except Exception as exc:
            failures.append({"session_id": source.stem, "error": str(exc)})
            print(f"{source.stem}: failed: {exc}", file=sys.stderr, flush=True)
            continue
        sessions += 1
        print(f"{source.stem}: archived and indexed", flush=True)
    return {
        "sessions": sessions,
        "chunks": chunks,
        "failed": len(failures),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill byte-for-byte Claude transcripts into raw retrieval"
    )
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument(
        "--session",
        action="append",
        dest="sessions",
        help="Backfill one session ID; repeat for multiple. Omit to backfill all.",
    )
    args = parser.parse_args()

    result = backfill(
        args.source_dir or default_source_dir(),
        set(args.sessions) if args.sessions else None,
    )
    print(f"Backfilled {result['sessions']} session(s), {result['chunks']} chunks")
    if result["failed"]:
        print(f"Failed to backfill {result['failed']} session(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
