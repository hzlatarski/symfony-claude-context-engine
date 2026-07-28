"""Regression tests for pruning removed vector-index sources."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))


def test_article_reindex_deletes_missing_slugs(tmp_path, monkeypatch) -> None:
    import reindex
    import utils
    import vector_store

    monkeypatch.setattr(utils, "STATE_FILE", tmp_path / "state.json")
    utils.save_state({
        "vector_article_hashes": {
            "concepts/deleted": "old",
            "concepts/also-deleted": "old",
        },
    })
    monkeypatch.setattr(reindex, "list_wiki_articles", lambda: [])
    monkeypatch.setattr(reindex, "load_contradictions", lambda: set())
    deleted: list[str] = []
    monkeypatch.setattr(vector_store, "delete_article", deleted.append)

    assert reindex.reindex_articles() == (0, 0)

    assert deleted == ["concepts/also-deleted", "concepts/deleted"]
    assert utils.load_state()["vector_article_hashes"] == {}


def test_daily_reindex_deletes_missing_sources(tmp_path, monkeypatch) -> None:
    import reindex
    import utils
    import vector_store

    monkeypatch.setattr(utils, "STATE_FILE", tmp_path / "state.json")
    utils.save_state({
        "vector_daily_hashes": {
            "daily/deleted.md": "old",
            "daily/transcripts/deleted.jsonl": "old",
        },
    })
    monkeypatch.setattr(reindex, "list_raw_files", lambda: [])
    monkeypatch.setattr(reindex, "list_transcript_files", lambda: [])
    deleted: list[str] = []
    monkeypatch.setattr(vector_store, "delete_chunks_for_daily", deleted.append)

    assert reindex.reindex_daily() == (0, 0)

    assert deleted == [
        "daily/deleted.md",
        "daily/transcripts/deleted.jsonl",
    ]
    assert utils.load_state()["vector_daily_hashes"] == {}
