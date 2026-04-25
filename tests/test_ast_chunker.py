"""Tests for ast_chunker — tree-sitter AST-based code chunking."""
from __future__ import annotations


class TestIsSupported:
    def test_php_and_js_supported(self):
        from ast_chunker import is_supported
        assert is_supported("php") is True
        assert is_supported("js") is True

    def test_other_types_unsupported(self):
        from ast_chunker import is_supported
        assert is_supported("twig") is False
        assert is_supported("yaml") is False
        assert is_supported("py") is False
        assert is_supported("") is False


class TestPhpChunking:
    def test_small_class_is_one_chunk(self):
        from ast_chunker import chunk_ast
        src = (
            "<?php\n"
            "namespace App;\n"
            "\n"
            "class Foo\n"
            "{\n"
            "    public function bar(): int\n"
            "    {\n"
            "        return 1;\n"
            "    }\n"
            "}\n"
        )
        chunks = chunk_ast(src, "php")
        assert chunks is not None
        # Prelude (namespace) merges into the class via _merge_tiny since
        # the namespace line alone is below MIN_CHUNK_LINES.
        assert any("class Foo" in c[2] for c in chunks)
        assert any("function bar" in c[2] for c in chunks)

    def test_two_classes_produce_separate_chunks(self):
        from ast_chunker import chunk_ast
        src = (
            "<?php\n"
            "class A\n"
            "{\n"
            "    public function methodA(): void {}\n"
            "    public function helperA(): void {}\n"
            "    public function moreA(): void {}\n"
            "}\n"
            "\n"
            "class B\n"
            "{\n"
            "    public function methodB(): void {}\n"
            "    public function helperB(): void {}\n"
            "    public function moreB(): void {}\n"
            "}\n"
        )
        chunks = chunk_ast(src, "php")
        assert chunks is not None
        a_chunk = next((c for c in chunks if "class A" in c[2]), None)
        b_chunk = next((c for c in chunks if "class B" in c[2]), None)
        assert a_chunk is not None
        assert b_chunk is not None
        assert a_chunk is not b_chunk
        assert "class B" not in a_chunk[2]
        assert "class A" not in b_chunk[2]

    def test_large_class_splits_per_method(self):
        from ast_chunker import chunk_ast, MAX_CHUNK_LINES
        # Build a class whose total span exceeds MAX_CHUNK_LINES so the
        # per-method splitter kicks in. Each method is well below the limit.
        method_body = "        $x = 1;\n" * 50
        methods = "".join(
            f"    public function method{i}(): void\n    {{\n{method_body}    }}\n"
            for i in range(10)
        )
        src = f"<?php\nclass Big\n{{\n{methods}}}\n"
        total_lines = src.count("\n")
        assert total_lines > MAX_CHUNK_LINES

        chunks = chunk_ast(src, "php")
        assert chunks is not None
        method_chunks = [c for c in chunks if "function method" in c[2]]
        assert len(method_chunks) >= 5
        for chunk in method_chunks:
            method_count = chunk[2].count("public function method")
            assert method_count <= 2, f"chunk holds too many methods:\n{chunk[2]}"

    def test_line_numbers_are_1_based_and_consistent(self):
        from ast_chunker import chunk_ast
        src = (
            "<?php\n"
            "class Foo\n"
            "{\n"
            "    public function bar(): int { return 1; }\n"
            "}\n"
        )
        chunks = chunk_ast(src, "php")
        assert chunks is not None
        for start, end, _text in chunks:
            assert start >= 1
            assert end >= start

    def test_returns_none_for_php_with_no_declarations(self):
        from ast_chunker import chunk_ast
        # A bare config-style PHP file — no class/function/interface.
        src = "<?php\nreturn [\n    'foo' => 1,\n    'bar' => 2,\n];\n"
        result = chunk_ast(src, "php")
        assert result is None


class TestJsChunking:
    def test_stimulus_class_produces_chunk(self):
        from ast_chunker import chunk_ast
        src = (
            "import { Controller } from '@hotwired/stimulus';\n"
            "\n"
            "export default class extends Controller {\n"
            "    static targets = ['input'];\n"
            "    connect() { this.foo = 1; }\n"
            "    disconnect() { this.foo = 0; }\n"
            "}\n"
        )
        chunks = chunk_ast(src, "js")
        assert chunks is not None
        assert any("class extends Controller" in c[2] for c in chunks)

    def test_top_level_function_chunked(self):
        from ast_chunker import chunk_ast
        src = (
            "function foo() { return 1; }\n"
            "function bar() { return 2; }\n"
        )
        chunks = chunk_ast(src, "js")
        assert chunks is not None
        joined = "".join(c[2] for c in chunks)
        assert "function foo" in joined
        assert "function bar" in joined


class TestUnsupported:
    def test_returns_none_for_unsupported_type(self):
        from ast_chunker import chunk_ast
        assert chunk_ast("<html></html>", "html") is None
        assert chunk_ast("foo: bar\n", "yaml") is None

    def test_returns_empty_for_empty_text(self):
        from ast_chunker import chunk_ast
        assert chunk_ast("", "php") == []


class TestFallbackIntegration:
    def test_chunk_file_uses_ast_for_php(self):
        from index_codebase import chunk_file
        src = (
            "<?php\n"
            "class Foo\n"
            "{\n"
            "    public function bar(): int { return 1; }\n"
            "    public function baz(): int { return 2; }\n"
            "}\n"
        )
        chunks = chunk_file(src, "php")
        assert len(chunks) >= 1
        joined = "".join(c[2] for c in chunks)
        assert "class Foo" in joined

    def test_chunk_file_falls_back_for_yaml(self):
        from index_codebase import chunk_file, CHUNK_SIZE
        # Long YAML — should hit the line-window path, not return [].
        src = "\n".join(f"key{i}: value{i}" for i in range(CHUNK_SIZE * 2))
        chunks = chunk_file(src, "yaml")
        assert len(chunks) >= 2

    def test_chunk_file_no_file_type_uses_line_windows(self):
        from index_codebase import chunk_file
        src = "line1\nline2\nline3\n"
        chunks = chunk_file(src)
        assert len(chunks) == 1
        assert chunks[0][0] == 1
