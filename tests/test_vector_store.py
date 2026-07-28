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

    def test_failed_article_replacement_retains_last_complete_generation(
        self, store, monkeypatch
    ):
        store.upsert_article(
            slug="concepts/durable",
            title="Durable",
            zone="observed",
            text="last complete generation",
            metadata={
                "type": "fact",
                "confidence": 0.9,
                "quarantined": False,
                "updated": "2026-07-28",
            },
        )
        collection = store._articles_collection()

        class FailingCollection:
            def get(self, *args, **kwargs):
                return collection.get(*args, **kwargs)

            def upsert(self, *args, **kwargs):
                raise RuntimeError("embedding failed")

            def delete(self, *args, **kwargs):
                return collection.delete(*args, **kwargs)

        monkeypatch.setattr(store, "_articles_collection", FailingCollection)
        with pytest.raises(RuntimeError, match="embedding failed"):
            store.replace_article(
                "concepts/durable",
                "Durable",
                {"observed": "incomplete replacement"},
                {
                    "type": "fact",
                    "confidence": 0.9,
                    "quarantined": False,
                    "updated": "2026-07-28",
                },
            )

        result = collection.get(
            where={"slug": {"$eq": "concepts/durable"}},
            include=["documents"],
        )
        assert result["documents"] == ["last complete generation"]

    def test_article_replacement_removes_a_zone_that_no_longer_exists(self, store):
        metadata = {
            "type": "fact",
            "confidence": 0.9,
            "quarantined": False,
            "updated": "2026-07-28",
        }
        store.replace_article(
            "concepts/shrinking",
            "Shrinking",
            {"observed": "observed facts", "synthesized": "old inference"},
            metadata,
        )
        store.replace_article(
            "concepts/shrinking",
            "Shrinking",
            {"observed": "new observed facts"},
            metadata,
        )

        result = store._articles_collection().get(
            where={"slug": {"$eq": "concepts/shrinking"}},
            include=["documents", "metadatas"],
        )
        assert result["documents"] == ["new observed facts"]
        assert [item["zone"] for item in result["metadatas"]] == ["observed"]

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

    def test_daily_embedding_uses_loss_safe_source_replacement(
        self, tmp_path, monkeypatch,
    ):
        import utils
        import vector_store

        knowledge = tmp_path / "knowledge"
        daily = knowledge / "daily" / "2026-07-28.md"
        daily.parent.mkdir(parents=True)
        daily.write_text(
            "# Daily Log\n\n## Sessions\n\n### Session\n\nKeep this.\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(utils, "KNOWLEDGE_DIR", knowledge)
        replacements = []
        monkeypatch.setattr(
            vector_store,
            "replace_chunks_for_source",
            lambda source, chunks: replacements.append((source, chunks)),
        )
        monkeypatch.setattr(
            vector_store,
            "delete_chunks_for_daily",
            lambda *_args: pytest.fail("daily embedding deleted the live generation first"),
        )

        count = utils.embed_daily_file(daily)

        assert count > 0
        assert replacements[0][0] == "daily/2026-07-28.md"
        assert len(replacements[0][1]) == count

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

    def test_upsert_chunks_indexes_a_batch(self, store):
        store.upsert_chunks([
            {
                "chunk_id": "daily/transcripts/s1.jsonl#1",
                "source_file": "daily/transcripts/s1.jsonl",
                "text": "first unique transcript payload",
                "metadata": {"section": "record 1", "date": "2026-07-28"},
            },
            {
                "chunk_id": "daily/transcripts/s1.jsonl#2",
                "source_file": "daily/transcripts/s1.jsonl",
                "text": "second unique transcript payload",
                "metadata": {"section": "record 2", "date": "2026-07-28"},
            },
        ])

        assert store.stats()["daily_chunks"] == 2
        results = store.search_daily("second unique transcript payload", limit=2)
        assert {result["id"] for result in results} == {
            "daily/transcripts/s1.jsonl#1",
            "daily/transcripts/s1.jsonl#2",
        }

    def test_search_daily_exact_finds_literal_url(self, store):
        url = "https://claude.ai/code/artifact/exact-identifier"
        store.upsert_chunk(
            chunk_id="daily/transcripts/s1.jsonl#url",
            source_file="daily/transcripts/s1.jsonl",
            text=f'{{"text": "Open {url} for the design."}}',
            metadata={"section": "assistant record", "date": "2026-07-28"},
        )

        results = store.search_daily_exact(url, limit=5)

        assert [result["id"] for result in results] == [
            "daily/transcripts/s1.jsonl#url"
        ]
        assert url in results[0]["text"]

    def test_get_daily_chunk_returns_full_long_reference(self, store):
        url = "https://example.test/artifact/" + ("a" * 700)
        chunk_id = "daily/transcripts/s1.jsonl#long-url"
        store.upsert_chunk(
            chunk_id=chunk_id,
            source_file="daily/transcripts/s1.jsonl",
            text=f'{{"text": "Open {url} for the design."}}',
            metadata={"section": "assistant record", "date": "2026-07-28"},
        )

        result = store.get_daily_chunk(chunk_id)

        assert url in result["text"]
        assert result["metadata"]["source_file"] == "daily/transcripts/s1.jsonl"

    def test_replace_chunks_for_source_removes_stale_chunks(self, store):
        source = "daily/transcripts/s1.jsonl"
        store.upsert_chunk(
            chunk_id="stale",
            source_file=source,
            text="obsolete payload",
            metadata={"section": "old", "date": "2026-07-28"},
        )

        store.replace_chunks_for_source(source, [{
            "chunk_id": "current",
            "source_file": source,
            "text": "current payload",
            "metadata": {"section": "new", "date": "2026-07-28"},
        }])

        assert store.search_daily_exact("obsolete payload") == []
        current = store.search_daily_exact("current payload")
        assert len(current) == 1
        assert current[0]["text"] == "current payload"

    def test_failed_source_replacement_keeps_previous_chunks_searchable(
        self,
        store,
        monkeypatch,
    ):
        source = "daily/transcripts/s1.jsonl"
        store.upsert_chunk(
            chunk_id="existing",
            source_file=source,
            text="irreplaceable previous payload",
            metadata={"section": "old", "date": "2026-07-28"},
        )
        monkeypatch.setattr(
            store,
            "_upsert_prepared_chunks",
            lambda prepared: (_ for _ in ()).throw(RuntimeError("embedding failed")),
        )

        with pytest.raises(RuntimeError, match="embedding failed"):
            store.replace_chunks_for_source(source, [{
                "chunk_id": "replacement",
                "source_file": source,
                "text": "new payload",
                "metadata": {"section": "new", "date": "2026-07-28"},
            }])

        assert store.search_daily_exact("irreplaceable previous payload")
        assert store.search_daily_exact("new payload") == []

    def test_failed_replacement_promotion_keeps_complete_new_chunks_searchable(
        self,
        store,
        monkeypatch,
    ):
        source = "daily/transcripts/s1.jsonl"
        store.upsert_chunk(
            chunk_id="existing",
            source_file=source,
            text="previous payload",
            metadata={"section": "old", "date": "2026-07-28"},
        )
        monkeypatch.setattr(
            store,
            "_update_prepared_metadata",
            lambda prepared: (_ for _ in ()).throw(RuntimeError("promotion failed")),
            raising=False,
        )

        with pytest.raises(RuntimeError, match="promotion failed"):
            store.replace_chunks_for_source(source, [{
                "chunk_id": "replacement",
                "source_file": source,
                "text": "complete staged replacement",
                "metadata": {"section": "new", "date": "2026-07-28"},
            }])

        assert store.search_daily_exact("complete staged replacement")

    def test_upsert_article_rejects_invalid_zone(self, store):
        with pytest.raises(ValueError, match="zone"):
            store.upsert_article(
                slug="concepts/bad", title="Bad", zone="other",
                text="x",
                metadata={"type": "fact", "confidence": 0.9, "quarantined": False, "updated": "2026-04-12"},
            )
