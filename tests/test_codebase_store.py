"""Tests for the ChromaDB codebase collection wrapper."""
from __future__ import annotations

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    import config
    import codebase_store

    monkeypatch.setattr(config, "CHROMA_DB_DIR", tmp_path / "chroma")
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient._identifier_to_system = {}
    except (ImportError, AttributeError):
        pass
    codebase_store._client = None
    return codebase_store


class TestCodebaseStore:
    def test_upsert_and_search(self, store):
        store.upsert_chunk(
            chunk_id="src/Security/AppCustomAuthenticator.php::0",
            rel_path="src/Security/AppCustomAuthenticator.php",
            text="class AppCustomAuthenticator extends AbstractLoginFormAuthenticator\n{\n    public function authenticate(Request $request): Passport\n    {\n        $email = $request->request->get('email');\n    }",
            metadata={"file_type": "php", "start_line": 1, "end_line": 5, "symbols": "AppCustomAuthenticator,authenticate"},
        )
        results = store.search_codebase("how does authentication work?", limit=3)
        assert len(results) >= 1
        assert "AppCustomAuthenticator" in results[0]["rel_path"] or "authenticate" in results[0]["text"]

    def test_filter_by_file_type(self, store):
        store.upsert_chunk(
            chunk_id="src/Security/Foo.php::0",
            rel_path="src/Security/Foo.php",
            text="class Foo implements AuthInterface { public function check() {} }",
            metadata={"file_type": "php", "start_line": 1, "end_line": 1},
        )
        store.upsert_chunk(
            chunk_id="assets/controllers/auth_controller.js::0",
            rel_path="assets/controllers/auth_controller.js",
            text="export default class extends Controller { connect() { this.authenticate(); } }",
            metadata={"file_type": "js", "start_line": 1, "end_line": 1},
        )
        php_results = store.search_codebase("auth", limit=5, file_type="php")
        assert all(r["metadata"]["file_type"] == "php" for r in php_results)

    def test_delete_chunks_for_file(self, store):
        store.upsert_chunk(
            chunk_id="src/Foo.php::0",
            rel_path="src/Foo.php",
            text="class Foo {}",
            metadata={"file_type": "php", "start_line": 1, "end_line": 1},
        )
        store.delete_chunks_for_file("src/Foo.php")
        results = store.search_codebase("Foo class", limit=5)
        assert not any(r["rel_path"] == "src/Foo.php" for r in results)

    def test_stats(self, store):
        store.upsert_chunk(
            chunk_id="src/Bar.php::0",
            rel_path="src/Bar.php",
            text="class Bar { public function run(): void {} }",
            metadata={"file_type": "php", "start_line": 1, "end_line": 1},
        )
        s = store.stats()
        assert s["codebase_chunks"] >= 1

    def test_failed_replacement_keeps_previous_complete_file(
        self, store, monkeypatch,
    ):
        rel = "src/Foo.php"
        store.upsert_chunk(
            chunk_id=f"{rel}::old",
            rel_path=rel,
            text="class FooOldImplementation {}",
            metadata={"file_type": "php", "start_line": 1, "end_line": 1},
        )
        collection = store._codebase_collection()

        def fail_during_stage(prepared):
            chunk_id, text, metadata = prepared[0]
            collection.upsert(
                ids=[chunk_id], documents=[text], metadatas=[metadata],
            )
            raise RuntimeError("embedding failed")

        monkeypatch.setattr(
            store, "_upsert_prepared_chunks", fail_during_stage, raising=False,
        )

        with pytest.raises(RuntimeError, match="embedding failed"):
            store.replace_chunks_for_file(
                rel,
                [{
                    "chunk_id": f"{rel}::0",
                    "text": "class FooNewImplementation {}",
                    "metadata": {
                        "file_type": "php",
                        "start_line": 1,
                        "end_line": 1,
                    },
                }],
            )

        rows = collection.get(
            where={"rel_path": {"$eq": rel}},
            include=["documents"],
        )
        assert rows["documents"] == ["class FooOldImplementation {}"]

    def test_successful_replacement_removes_previous_generation(self, store):
        rel = "src/Foo.php"
        store.upsert_chunk(
            chunk_id=f"{rel}::old",
            rel_path=rel,
            text="class FooOldImplementation {}",
            metadata={"file_type": "php", "start_line": 1, "end_line": 1},
        )

        store.replace_chunks_for_file(
            rel,
            [{
                "chunk_id": f"{rel}::0",
                "text": "class FooNewImplementation {}",
                "metadata": {
                    "file_type": "php",
                    "start_line": 1,
                    "end_line": 1,
                },
            }],
        )

        rows = store._codebase_collection().get(
            where={"rel_path": {"$eq": rel}},
            include=["documents"],
        )
        assert rows["documents"] == ["class FooNewImplementation {}"]

    def test_identical_replacement_keeps_the_promoted_generation(self, store):
        rel = "src/Idempotent.php"
        chunks = [{
            "chunk_id": f"{rel}::0",
            "text": "class Idempotent {}",
            "metadata": {
                "file_type": "php",
                "start_line": 1,
                "end_line": 1,
            },
        }]

        store.replace_chunks_for_file(rel, chunks)
        store.replace_chunks_for_file(rel, chunks)

        rows = store._codebase_collection().get(
            where={"rel_path": {"$eq": rel}},
            include=["documents"],
        )
        assert rows["documents"] == ["class Idempotent {}"]

    def test_failed_identical_replacement_restores_existing_generation(
        self, store, monkeypatch,
    ):
        rel = "src/Retry.php"
        chunks = [{
            "chunk_id": f"{rel}::0",
            "text": "class Retry {}",
            "metadata": {
                "file_type": "php",
                "start_line": 1,
                "end_line": 1,
            },
        }]
        store.replace_chunks_for_file(rel, chunks)
        collection = store._codebase_collection()

        def fail_after_overwriting_metadata(prepared):
            chunk_id, text, metadata = prepared[0]
            collection.upsert(
                ids=[chunk_id],
                documents=[text],
                metadatas=[metadata],
            )
            raise RuntimeError("stage interrupted")

        monkeypatch.setattr(
            store, "_upsert_prepared_chunks", fail_after_overwriting_metadata,
        )

        with pytest.raises(RuntimeError, match="stage interrupted"):
            store.replace_chunks_for_file(rel, chunks)

        rows = collection.get(
            where={"rel_path": {"$eq": rel}},
            include=["documents"],
        )
        assert rows["documents"] == ["class Retry {}"]

    def test_cleanup_failure_never_deletes_the_only_existing_generation(
        self, store, monkeypatch,
    ):
        rel = "src/CleanupRetry.php"
        chunks = [{
            "chunk_id": f"{rel}::0",
            "text": "class CleanupRetry {}",
            "metadata": {
                "file_type": "php",
                "start_line": 1,
                "end_line": 1,
            },
        }]
        store.replace_chunks_for_file(rel, chunks)
        collection = store._codebase_collection()
        existing = collection.get(
            where={"rel_path": {"$eq": rel}},
            include=[],
        )
        existing_id = existing["ids"][0]

        def fail_after_overwriting_metadata(prepared):
            chunk_id, text, metadata = prepared[0]
            collection.upsert(
                ids=[chunk_id],
                documents=[text],
                metadatas=[metadata],
            )
            raise RuntimeError("stage interrupted")

        class CleanupFailingCollection:
            def get(self, *args, **kwargs):
                return collection.get(*args, **kwargs)

            def delete(self, *args, **kwargs):
                return collection.delete(*args, **kwargs)

            def update(self, *args, **kwargs):
                raise RuntimeError("metadata restore failed")

        monkeypatch.setattr(
            store, "_upsert_prepared_chunks", fail_after_overwriting_metadata,
        )
        monkeypatch.setattr(
            store, "_codebase_collection", CleanupFailingCollection,
        )

        with pytest.raises(RuntimeError, match="metadata restore failed"):
            store.replace_chunks_for_file(rel, chunks)

        retained = collection.get(ids=[existing_id], include=["documents"])
        assert retained["documents"] == ["class CleanupRetry {}"]
