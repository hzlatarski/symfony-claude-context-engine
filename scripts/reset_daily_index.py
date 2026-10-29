"""Drop the daily vector collections and shrink chroma.sqlite3.

Run after upgrading to 0.16.0, then ``reindex.py --daily-only`` rebuilds the
daily index with the plain-text transcript indexer. Before 0.16.0 every raw
JSONL line was indexed; on eintollesfest that was 673M chars (7.7 GB of
store) of which 62M were message text. Deleting the chunks alone does not
give the space back, so this also VACUUMs the file.

Drops ``daily_chunks`` (abandoned since 0.14.0) and the active daily
collection. Articles and codebase collections are left untouched, and the
archived transcripts on disk are never touched.

    uv run python scripts/reset_daily_index.py
    uv run python scripts/reindex.py --daily-only

Stop this project's Claude Code sessions / MCP servers first: VACUUM needs
the file to itself, and a running server keeps the dropped collection's id.
Also make sure no reindex.py / flush.py / compile.py of this project is
running: a reindex that is mid-run saves its file hashes after this script
cleared them, and the rebuild then skips files whose chunks were deleted.
"""
from __future__ import annotations

import sqlite3
import sys

import chroma_client
import config
from utils import update_state

LEGACY_DAILY = "daily_chunks"


def main() -> int:
    db = config.CHROMA_DB_DIR / "chroma.sqlite3"
    if not db.exists():
        print("no chroma store; nothing to reset")
        return 0
    before = db.stat().st_size

    for name in (LEGACY_DAILY, config.CHROMA_COLLECTION_DAILY):
        with chroma_client.write_session(name):
            client = chroma_client.get_client()
            if name in [c.name for c in client.list_collections()]:
                client.delete_collection(name)
                print(f"deleted {name}")

    update_state(lambda state: state.pop(config.DAILY_HASHES_STATE_KEY, None))
    print(f"cleared {config.DAILY_HASHES_STATE_KEY}")

    chroma_client.reset()
    con = sqlite3.connect(str(db), timeout=120)
    try:
        con.execute("VACUUM")
    finally:
        con.close()
    print(f"chroma.sqlite3: {before / 1e9:.2f} GB -> {db.stat().st_size / 1e9:.2f} GB")
    print("next: uv run python scripts/reindex.py --daily-only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
