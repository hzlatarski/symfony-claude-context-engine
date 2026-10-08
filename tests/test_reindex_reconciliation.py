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

    monkeypatch.setenv("MC_EMBEDDER", "minilm")
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
        "vector_daily_hashes_v2": {
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
    assert utils.load_state()["vector_daily_hashes_v2"] == {}


def _articles(tmp_path, names):
    concepts = tmp_path / "concepts"
    concepts.mkdir(exist_ok=True)
    paths = []
    for name in names:
        path = concepts / f"{name}.md"
        path.write_text(f"# {name}\n", encoding="utf-8")
        paths.append(path)
    return paths


def _setup(tmp_path, monkeypatch, names):
    import reindex
    import utils

    monkeypatch.setattr(utils, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(reindex, "KNOWLEDGE_DIR", tmp_path)
    paths = _articles(tmp_path, names)
    monkeypatch.setattr(reindex, "list_wiki_articles", lambda: list(paths))
    monkeypatch.setattr(reindex, "load_contradictions", lambda: set())
    return paths


def test_gemma_reindex_uses_its_own_hash_key(tmp_path, monkeypatch) -> None:
    """Switching embedders must embed every article into the new collection."""
    import reindex
    import utils

    paths = _setup(tmp_path, monkeypatch, ["a", "b"])
    legacy = {f"concepts/{p.stem}": utils.file_hash(p) for p in paths}
    utils.save_state({"vector_article_hashes": dict(legacy)})
    embedded: list[str] = []
    monkeypatch.setattr(reindex, "embed_article_file", lambda p, quarantined: embedded.append(p.stem))
    monkeypatch.setenv("MC_EMBEDDER", "gemma2-q4")

    assert reindex.reindex_articles() == (2, 0)
    assert embedded == ["a", "b"]
    state = utils.load_state()
    assert state["vector_article_hashes_gemma2_q4"] == legacy
    assert state["vector_article_hashes"] == legacy  # rollback cache untouched

    assert reindex.reindex_articles() == (0, 2)  # second run is incremental


def test_interrupted_gemma_rebuild_resumes(tmp_path, monkeypatch) -> None:
    import pytest
    import reindex
    import utils

    _setup(tmp_path, monkeypatch, ["a", "b", "c"])
    utils.save_state({})
    monkeypatch.setenv("MC_EMBEDDER", "gemma2-q4")
    calls: list[str] = []

    def flaky(path, quarantined):
        calls.append(path.stem)
        if path.stem == "c" and calls.count("c") == 1:
            raise RuntimeError("killed mid-run")

    monkeypatch.setattr(reindex, "embed_article_file", flaky)
    with pytest.raises(RuntimeError, match="killed mid-run"):
        reindex.reindex_articles()
    assert set(utils.load_state()["vector_article_hashes_gemma2_q4"]) == {"concepts/a", "concepts/b"}

    assert reindex.reindex_articles() == (1, 2)
    assert calls == ["a", "b", "c", "c"]


def test_hash_cache_is_saved_during_the_run_not_only_at_the_end(tmp_path, monkeypatch) -> None:
    """A killed process never reaches ``finally``. Progress must already be on disk."""
    import reindex
    import utils

    _setup(tmp_path, monkeypatch, ["a", "b", "c"])
    utils.save_state({})
    monkeypatch.setenv("MC_EMBEDDER", "gemma2-q4")
    monkeypatch.setattr(reindex, "PERSIST_EVERY", 2)
    on_disk_when_c_starts: list[set[str]] = []

    def embed(path, quarantined):
        if path.stem == "c":
            saved = utils.load_state().get("vector_article_hashes_gemma2_q4", {})
            on_disk_when_c_starts.append(set(saved))

    monkeypatch.setattr(reindex, "embed_article_file", embed)
    reindex.reindex_articles()
    assert on_disk_when_c_starts == [{"concepts/a", "concepts/b"}]
