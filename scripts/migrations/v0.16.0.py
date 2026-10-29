"""0.16.0 migration: explain the daily-index rebuild.

0.16.0 indexes transcripts as plain message text instead of raw JSONL lines
(~9% of the old size). Existing stores keep the old chunks until rebuilt.
The rebuild is not run here: on a large project it takes tens of minutes,
too long to block an upgrade, and it needs the MCP servers stopped.
Until it runs, search still works on the old chunks.
"""
from __future__ import annotations

import sys

NEXT_STEPS = """
  Next, inside .claude/memory-compiler (with VIRTUAL_ENV unset):
    1. Close this project's Claude Code sessions (they hold the MCP servers).
    2. uv run python scripts/reset_daily_index.py   # drops daily index, shrinks the file
    3. uv run python scripts/reindex.py --daily-only
"""


def main() -> int:
    print(NEXT_STEPS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
