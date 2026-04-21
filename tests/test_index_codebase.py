"""Tests for index_codebase — chunker and indexer."""
from __future__ import annotations

import pytest
from pathlib import Path


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Isolated ChromaDB + project root for indexer tests."""
    import config
    import codebase_store

    monkeypatch.setattr(config, "CHROMA_DB_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path / "project")
    (tmp_path / "project").mkdir()
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient._identifier_to_system = {}
    except (ImportError, AttributeError):
        pass
    codebase_store._client = None
    return tmp_path


class TestChunkFile:
    def test_single_chunk_for_short_file(self):
        from index_codebase import chunk_file
        lines = ["line {}\n".format(i) for i in range(50)]
        chunks = chunk_file("".join(lines))
        assert len(chunks) == 1
        start, end, text = chunks[0]
        assert start == 1
        assert end == 50

    def test_overlap_produces_multiple_chunks(self):
        from index_codebase import chunk_file, CHUNK_SIZE, CHUNK_OVERLAP
        lines = ["line {}\n".format(i) for i in range(200)]
        chunks = chunk_file("".join(lines))
        assert len(chunks) >= 2
        # Second chunk starts at CHUNK_SIZE - CHUNK_OVERLAP + 1
        _, _, first_text = chunks[0]
        _, _, second_text = chunks[1]
        # Overlap: last lines of first chunk appear in second
        first_lines = first_text.splitlines()
        second_lines = second_text.splitlines()
        shared = set(first_lines[-CHUNK_OVERLAP:]) & set(second_lines[:CHUNK_OVERLAP])
        assert len(shared) > 0

    def test_empty_file_returns_no_chunks(self):
        from index_codebase import chunk_file
        assert chunk_file("") == []

    def test_start_end_line_numbers_are_1_based(self):
        from index_codebase import chunk_file
        text = "a\nb\nc\n"
        chunks = chunk_file(text)
        assert chunks[0][0] == 1  # start_line


class TestExtractSymbols:
    def test_extracts_class_and_method(self):
        from index_codebase import _extract_symbols
        php = "class AppCustomAuthenticator extends AbstractLoginFormAuthenticator {\n    public function authenticate(Request $r): Passport {}\n}"
        symbols = _extract_symbols(php, "php")
        assert "AppCustomAuthenticator" in symbols
        assert "authenticate" in symbols

    def test_non_php_returns_empty(self):
        from index_codebase import _extract_symbols
        assert _extract_symbols("export default class extends Controller {}", "js") == ""

    def test_no_symbols_returns_empty(self):
        from index_codebase import _extract_symbols
        assert _extract_symbols("just some plain text without any classes", "php") == ""


class TestIndexFile:
    def test_indexes_php_file(self, isolated):
        import config
        import codebase_store
        from index_codebase import index_file

        src_dir = config.PROJECT_ROOT / "src" / "Security"
        src_dir.mkdir(parents=True)
        php_file = src_dir / "AppCustomAuthenticator.php"
        php_file.write_text(
            "<?php\nclass AppCustomAuthenticator extends AbstractLoginFormAuthenticator\n{\n    public function authenticate(Request $request): Passport\n    {\n        return new Passport();\n    }\n}\n",
            encoding="utf-8",
        )
        n = index_file(php_file)
        assert n >= 1
        results = codebase_store.search_codebase("how does authentication work?", limit=5)
        assert any("AppCustomAuthenticator" in r["rel_path"] for r in results)

    def test_skips_empty_file(self, isolated):
        import config
        from index_codebase import index_file

        src_dir = config.PROJECT_ROOT / "src"
        src_dir.mkdir(parents=True)
        empty = src_dir / "Empty.php"
        empty.write_text("", encoding="utf-8")
        n = index_file(empty)
        assert n == 0


class TestReindexSingle:
    def test_ignores_non_indexed_extensions(self, isolated):
        import config
        from index_codebase import reindex_single

        f = config.PROJECT_ROOT / "README.md"
        f.write_text("# readme", encoding="utf-8")
        n = reindex_single(str(f))
        assert n == 0

    def test_indexes_php_file(self, isolated):
        import config
        from index_codebase import reindex_single

        src_dir = config.PROJECT_ROOT / "src"
        src_dir.mkdir(parents=True)
        php = src_dir / "Foo.php"
        php.write_text("<?php\nclass Foo {}\n", encoding="utf-8")
        n = reindex_single(str(php))
        assert n >= 1

    def test_updates_hash_cache(self, isolated):
        import config
        from index_codebase import reindex_single
        from utils import load_state

        src_dir = config.PROJECT_ROOT / "src"
        src_dir.mkdir(parents=True)
        php = src_dir / "Bar.php"
        php.write_text("<?php\nclass Bar {}\n", encoding="utf-8")
        reindex_single(str(php))
        state = load_state()
        rel = str(php.relative_to(config.PROJECT_ROOT)).replace("\\", "/")
        assert rel in state.get("codebase_hashes", {})
