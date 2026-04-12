"""Pytest suite for the ChromaDB wrapper."""
from __future__ import annotations

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Isolated Chroma store rooted at tmp_path for each test.

    The memory-compiler's scripts directory is on pythonpath, so every
    script in the codebase uses `from config import ...` / `import config`
    — not `from scripts import config`. Importing `scripts.config` in the
    test creates a SECOND module object and the monkeypatch would never
    reach production code. Import `config` and `vector_store` directly
    so both sides of the monkeypatch see the same module.
    """
    import config
    import vector_store

    monkeypatch.setattr(config, "CHROMA_DB_DIR", tmp_path / "chroma")
    # Clear ChromaDB's process-global SharedSystemClient registry so
    # the next PersistentClient actually builds a fresh backend at
    # tmp_path/chroma instead of reusing the cached one.
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient._identifier_to_system = {}
    except (ImportError, AttributeError):
        pass
    vector_store._client = None
    return vector_store


class TestVectorStoreBasics:
    def test_upsert_and_search_article(self, store):
        store.upsert_article(
            slug="concepts/stimulus-naming",
            title="Stimulus Naming Convention",
            zone="observed",
            text="Stimulus controllers use kebab-case identifiers and underscore filenames.",
            metadata={
                "type": "fact",
                "confidence": 0.9,
                "quarantined": False,
                "updated": "2026-04-12",
            },
        )
        results = store.search_articles("how should I name my stimulus controller?", limit=3)
        assert len(results) >= 1
        assert results[0]["slug"] == "concepts/stimulus-naming"
        assert results[0]["metadata"]["type"] == "fact"

    def test_metadata_filter_type(self, store):
        store.upsert_article(
            slug="concepts/a", title="A", zone="observed",
            text="fact about stimulus naming",
            metadata={"type": "fact", "confidence": 0.9, "quarantined": False, "updated": "2026-04-12"},
        )
        store.upsert_article(
            slug="concepts/b", title="B", zone="observed",
            text="advice about stimulus naming",
            metadata={"type": "advice", "confidence": 0.9, "quarantined": False, "updated": "2026-04-12"},
        )
        results = store.search_articles("stimulus naming", limit=10, type_filter="advice")
        slugs = [r["slug"] for r in results]
        assert "concepts/b" in slugs
        assert "concepts/a" not in slugs

    def test_quarantine_filter_excludes_by_default(self, store):
        store.upsert_article(
            slug="concepts/good", title="Good", zone="observed",
            text="valid article",
            metadata={"type": "fact", "confidence": 0.9, "quarantined": False, "updated": "2026-04-12"},
        )
        store.upsert_article(
            slug="concepts/bad", title="Bad", zone="observed",
            text="contradicted article",
            metadata={"type": "fact", "confidence": 0.9, "quarantined": True, "updated": "2026-04-12"},
        )
        results = store.search_articles("article", limit=10)
        slugs = [r["slug"] for r in results]
        assert "concepts/good" in slugs
        assert "concepts/bad" not in slugs

    def test_confidence_floor_filter(self, store):
        store.upsert_article(
            slug="concepts/firm", title="Firm", zone="observed",
            text="firm decision",
            metadata={"type": "decision", "confidence": 0.9, "quarantined": False, "updated": "2026-04-12"},
        )
        store.upsert_article(
            slug="concepts/tentative", title="Tentative", zone="observed",
            text="tentative plan",
            metadata={"type": "decision", "confidence": 0.3, "quarantined": False, "updated": "2026-04-12"},
        )
        results = store.search_articles("decision", limit=10, min_confidence=0.5)
        slugs = [r["slug"] for r in results]
        assert "concepts/firm" in slugs
        assert "concepts/tentative" not in slugs

    def test_delete_article_removes_all_zones(self, store):
        store.upsert_article(
            slug="concepts/x", title="X", zone="observed",
            text="obs facts", metadata={"type": "fact", "confidence": 0.9, "quarantined": False, "updated": "2026-04-12"},
        )
        store.upsert_article(
            slug="concepts/x", title="X", zone="synthesized",
            text="synth inferences", metadata={"type": "fact", "confidence": 0.9, "quarantined": False, "updated": "2026-04-12"},
        )
        store.delete_article("concepts/x")
        results = store.search_articles("facts", limit=10)
        assert all(r["slug"] != "concepts/x" for r in results)

    def test_upsert_is_idempotent(self, store):
        for _ in range(3):
            store.upsert_article(
                slug="concepts/z", title="Z", zone="observed",
                text="same text",
                metadata={"type": "fact", "confidence": 0.9, "quarantined": False, "updated": "2026-04-12"},
            )
        stats = store.stats()
        assert stats["articles"] == 1

    def test_stats_reports_collection_sizes(self, store):
        store.upsert_article(
            slug="concepts/a", title="A", zone="observed",
            text="first",
            metadata={"type": "fact", "confidence": 0.9, "quarantined": False, "updated": "2026-04-12"},
        )
        store.upsert_chunk(
            chunk_id="daily/2026-04-12#intro",
            source_file="daily/2026-04-12.md",
            text="morning standup notes",
            metadata={"section": "Intro", "date": "2026-04-12"},
        )
        stats = store.stats()
        assert stats["articles"] >= 1
        assert stats["daily_chunks"] >= 1

    def test_search_on_empty_collection_returns_empty_list(self, store):
        """Regression: _flatten_results must survive empty result shapes."""
        assert store.search_articles("anything", limit=5) == []
        assert store.search_daily("anything", limit=5) == []

    def test_zone_filter(self, store):
        store.upsert_article(
            slug="concepts/stimulus-z", title="Z", zone="observed",
            text="raw observation about stimulus",
            metadata={"type": "fact", "confidence": 0.9, "quarantined": False, "updated": "2026-04-12"},
        )
        store.upsert_article(
            slug="concepts/stimulus-z", title="Z", zone="synthesized",
            text="inference drawn from raw observation about stimulus",
            metadata={"type": "fact", "confidence": 0.9, "quarantined": False, "updated": "2026-04-12"},
        )
        results = store.search_articles(
            "observation about stimulus", limit=10, zone_filter="synthesized"
        )
        assert len(results) >= 1
        assert all(r["metadata"]["zone"] == "synthesized" for r in results)

    def test_delete_chunks_for_daily_removes_only_target_file(self, store):
        store.upsert_chunk(
            chunk_id="daily/2026-04-10.md#section-a",
            source_file="daily/2026-04-10.md",
            text="old content",
            metadata={"section": "A", "date": "2026-04-10"},
        )
        store.upsert_chunk(
            chunk_id="daily/2026-04-11.md#section-b",
            source_file="daily/2026-04-11.md",
            text="new content",
            metadata={"section": "B", "date": "2026-04-11"},
        )
        store.delete_chunks_for_daily("daily/2026-04-10.md")
        results = store.search_daily("content", limit=10)
        ids = [r["id"] for r in results]
        assert "daily/2026-04-11.md#section-b" in ids
        assert "daily/2026-04-10.md#section-a" not in ids

    def test_search_daily_date_range(self, store):
        for date_str, section in [("2026-04-01", "old"), ("2026-04-10", "mid"), ("2026-04-15", "new")]:
            store.upsert_chunk(
                chunk_id=f"daily/{date_str}.md#{section}",
                source_file=f"daily/{date_str}.md",
                text=f"{section} content",
                metadata={"section": section, "date": date_str},
            )
        results = store.search_daily(
            "content", limit=10, date_from="2026-04-05", date_to="2026-04-12"
        )
        ids = [r["id"] for r in results]
        assert "daily/2026-04-10.md#mid" in ids
        assert "daily/2026-04-01.md#old" not in ids
        assert "daily/2026-04-15.md#new" not in ids

    def test_upsert_chunk_flattens_list_metadata(self, store):
        """upsert_chunk must flatten lists the same way upsert_article does."""
        store.upsert_chunk(
            chunk_id="daily/2026-04-12.md#tagged",
            source_file="daily/2026-04-12.md",
            text="chunk with tags",
            metadata={"section": "Tagged", "date": "2026-04-12", "tags": ["foo", "bar"]},
        )
        results = store.search_daily("chunk with tags", limit=5)
        assert len(results) >= 1
        assert results[0]["metadata"].get("tags") == "foo,bar"

    def test_upsert_article_rejects_invalid_zone(self, store):
        with pytest.raises(ValueError, match="zone"):
            store.upsert_article(
                slug="concepts/bad", title="Bad", zone="other",
                text="x",
                metadata={"type": "fact", "confidence": 0.9, "quarantined": False, "updated": "2026-04-12"},
            )
