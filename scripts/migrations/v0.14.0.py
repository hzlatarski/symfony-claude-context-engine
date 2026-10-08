"""0.14.0 migration: fetch the EmbeddingGemma 2 model, then explain the rebuild.

Run by scripts/upgrade.py with the project venv after ``uv sync``.

It only downloads the model (~170 MB, once per machine; the cache is shared
by every project). It does not rebuild the indexes: an articles rebuild takes
~35 min of CPU per large project, too long to block an upgrade. Until the
rebuild runs, search_knowledge answers from BM25 alone and daily search is
empty -- nothing breaks, it is just less good.

Never fails the upgrade: an offline machine gets the fix-up command instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

NEXT_STEPS = """
  Next, inside .claude/memory-compiler (with VIRTUAL_ENV unset):
    1. Restart Claude Code sessions / MCP servers of this project (old code
       still running can damage the rebuilt index).
    2. uv run python scripts/reindex.py --articles-only   # Gemma articles index
    3. uv run python scripts/ingest_scenarios.py          # only if you use scenarios
    4. uv run python scripts/reindex.py --daily-only      # builds daily_chunks_v2
    5. uv run python scripts/vector_health.py             # must print ok
  Rollback for articles: set MC_EMBEDDER=minilm.
"""


def main() -> int:
    try:
        import embedder

        directory = embedder.ensure_model(download=True)
        print(f"  EmbeddingGemma 2 model ready in {directory}")
    except Exception as exc:  # offline, disk full, HF outage
        print(f"  WARNING: model download failed ({type(exc).__name__}: {exc}).")
        print("  Retry with: uv run python scripts/embedder.py --download")
    print(NEXT_STEPS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
