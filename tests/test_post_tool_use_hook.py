"""Tests for the PostToolUse hook — tool digest extraction + JSONL append.

The hook file is ``hooks/post-tool-use.py``. Python's import machinery can't
dot-import hyphenated filenames, so we load it via ``importlib.util`` and
cache the result on the class. The hook is stdlib-only and has no network
or LLM dependencies, so these tests run fast and deterministically.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

HOOK_PATH = Path(__file__).resolve().parent.parent / "hooks" / "post-tool-use.py"


@pytest.fixture(scope="module")
def hook_module():
    """Load post-tool-use.py as a Python module under the name ``post_tool_use``.

    Module-scoped so the expensive importlib path runs once. We clean up
    ``sys.modules`` after the module fixture tears down to avoid leaking
    into unrelated tests that might import the ``hooks`` package.

    Important: the hook file has a module-level recursion guard that calls
    ``sys.exit(0)`` when ``CLAUDE_INVOKED_BY`` is set. ``scripts/flush.py``
    sets that env var at its own module-load time as a real-world guard
    against subagent recursion, which means if any prior test imported
    flush.py the var is sticky for the rest of the session. We save and
    strip it just before exec, then restore it, so the fixture remains
    order-independent against test_flush_tool_drawer.py and any other
    test that touches flush.
    """
    saved = os.environ.pop("CLAUDE_INVOKED_BY", None)
    try:
        spec = importlib.util.spec_from_file_location("post_tool_use", HOOK_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules["post_tool_use"] = module
        spec.loader.exec_module(module)
    finally:
        if saved is not None:
            os.environ["CLAUDE_INVOKED_BY"] = saved
    yield module
    sys.modules.pop("post_tool_use", None)


class TestBuildDigest:
    def test_read_keeps_only_file_path(self, hook_module):
        digest = hook_module.build_digest(
            "Read",
            {"file_path": "/tmp/foo.py", "offset": 10, "limit": 50},
        )
        assert digest == {"file_path": "/tmp/foo.py"}

    def test_edit_keeps_only_file_path(self, hook_module):
        digest = hook_module.build_digest(
            "Edit",
            {
                "file_path": "/tmp/foo.py",
                "old_string": "a" * 5000,  # must NOT leak into the drawer
                "new_string": "b" * 5000,
            },
        )
        assert digest == {"file_path": "/tmp/foo.py"}

    def test_bash_keeps_first_line_of_command_only(self, hook_module):
        digest = hook_module.build_digest(
            "Bash",
            {
                "command": "echo hello\nrm -rf /\necho done",
                "description": "test",
                "run_in_background": False,
            },
        )
        assert digest["command"] == "echo hello"
        assert digest["run_in_background"] is False
        # rest of the command must not appear in any field
        assert "rm -rf" not in json.dumps(digest)

    def test_bash_long_command_is_truncated(self, hook_module):
        long_cmd = "x" * 1000
        digest = hook_module.build_digest("Bash", {"command": long_cmd})
        cap = hook_module._MAX_DIGEST_CHARS
        assert len(digest["command"]) <= cap + 1  # +1 for the "…" glyph
        assert digest["command"].endswith("…")

    def test_grep_keeps_pattern_path_mode(self, hook_module):
        digest = hook_module.build_digest(
            "Grep",
            {
                "pattern": "def foo",
                "path": "src/",
                "output_mode": "content",
                "-n": True,
            },
        )
        assert digest == {
            "pattern": "def foo",
            "path": "src/",
            "output_mode": "content",
        }

    def test_task_keeps_description_and_subagent(self, hook_module):
        digest = hook_module.build_digest(
            "Task",
            {
                "description": "Compare repos",
                "subagent_type": "general-purpose",
                "prompt": "a very long prompt " * 500,
            },
        )
        assert digest == {
            "description": "Compare repos",
            "subagent_type": "general-purpose",
        }

    def test_todowrite_records_count_only(self, hook_module):
        digest = hook_module.build_digest(
            "TodoWrite",
            {"todos": [{"content": "x"}, {"content": "y"}, {"content": "z"}]},
        )
        assert digest == {"todo_count": 3}

    def test_unknown_tool_drops_large_values(self, hook_module):
        digest = hook_module.build_digest(
            "CustomThing",
            {
                "small": "ok",
                "big_string": "x" * 5000,
                "nested": {"leak": "this should drop"},
                "list_val": [1, 2, 3],
                "number": 42,
                "flag": True,
            },
        )
        # scalars survive untruncated
        assert digest["small"] == "ok"
        assert digest["number"] == 42
        assert digest["flag"] is True
        # big strings get truncated but still appear
        assert digest["big_string"].endswith("…")
        # lists and dicts are dropped entirely
        assert "nested" not in digest
        assert "list_val" not in digest

    def test_non_dict_input_is_wrapped(self, hook_module):
        digest = hook_module.build_digest("Weird", "just a string")
        assert digest == {"raw": "just a string"}


class TestMeasureResponse:
    def test_string_response_size(self, hook_module):
        size, ok = hook_module.measure_response("hello world")
        assert size == len("hello world")
        assert ok is True

    def test_none_response_is_zero_and_ok(self, hook_module):
        size, ok = hook_module.measure_response(None)
        assert size == 0
        assert ok is True

    def test_is_error_flag_marks_not_ok(self, hook_module):
        _, ok = hook_module.measure_response({"is_error": True, "content": "boom"})
        assert ok is False

    def test_error_key_marks_not_ok(self, hook_module):
        _, ok = hook_module.measure_response({"error": "oops"})
        assert ok is False

    def test_normal_dict_is_ok(self, hook_module):
        size, ok = hook_module.measure_response({"content": "fine"})
        assert ok is True
        assert size > 0


class TestAppendJsonl:
    def test_creates_parent_dir_and_appends(self, hook_module, tmp_path):
        target = tmp_path / "nested" / "dir" / "drawer.jsonl"
        hook_module.append_jsonl(target, {"a": 1})
        hook_module.append_jsonl(target, {"b": 2})

        lines = target.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}
        assert json.loads(lines[1]) == {"b": 2}

    def test_unicode_preserved_not_escaped(self, hook_module, tmp_path):
        target = tmp_path / "drawer.jsonl"
        hook_module.append_jsonl(target, {"text": "résumé 日本語"})
        raw = target.read_text(encoding="utf-8")
        assert "résumé" in raw
        assert "日本語" in raw

    def test_preserves_existing_lines(self, hook_module, tmp_path):
        target = tmp_path / "drawer.jsonl"
        target.write_text('{"pre":"existing"}\n', encoding="utf-8")
        hook_module.append_jsonl(target, {"new": True})
        lines = target.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"pre": "existing"}
        assert json.loads(lines[1]) == {"new": True}
