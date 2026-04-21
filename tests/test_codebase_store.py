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
