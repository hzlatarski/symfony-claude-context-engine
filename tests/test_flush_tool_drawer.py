"""Tests for the tool-drawer helpers in scripts/flush.py.

The PostToolUse hook writes structured events to a JSONL drawer; flush.py
reads the drawer per-session and renders a compact summary that Haiku
uses as ground truth when compiling the daily log. These tests cover
the load + format pipeline in isolation from the LLM call.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def drawer(tmp_path):
    """Factory that writes JSONL records to a dated drawer file.

    Returns ``(daily_dir, today_iso, write)`` where ``write(records)`` appends
    the given records to ``daily_dir/<today_iso>.tools.jsonl``. Tests use the
    factory so each can pick its own date / filename semantics without fighting
    the system clock.
    """
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    today_iso = "2026-04-14"
    path = daily_dir / f"{today_iso}.tools.jsonl"

    def write(records):
        lines = [json.dumps(r, ensure_ascii=False) for r in records]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return daily_dir, today_iso, write


class TestLoadToolEvents:
    def test_returns_empty_list_when_file_missing(self, tmp_path):
        from flush import load_tool_events
        assert load_tool_events("sess", tmp_path, today_iso="2026-04-14") == []

    def test_filters_by_session_id(self, drawer):
        daily_dir, today_iso, write = drawer
        write([
            {"ts": "t1", "session_id": "A", "tool": "Edit", "input": {"file_path": "/a"}, "result_size": 10, "ok": True},
            {"ts": "t2", "session_id": "B", "tool": "Edit", "input": {"file_path": "/b"}, "result_size": 10, "ok": True},
            {"ts": "t3", "session_id": "A", "tool": "Bash", "input": {"command": "ls"}, "result_size": 10, "ok": True},
        ])

        from flush import load_tool_events
        events = load_tool_events("A", daily_dir, today_iso=today_iso)
        assert len(events) == 2
        assert all(e["session_id"] == "A" for e in events)
        assert [e["tool"] for e in events] == ["Edit", "Bash"]

    def test_skips_malformed_lines(self, drawer):
        daily_dir, today_iso, _ = drawer
        path = daily_dir / f"{today_iso}.tools.jsonl"
        path.write_text(
            '{"session_id":"A","tool":"Edit","input":{"file_path":"/a"}}\n'
            'not valid json\n'
            '{"session_id":"A","tool":"Bash","input":{"command":"ls"}}\n'
            '\n'
            '   \n',
            encoding="utf-8",
        )
        from flush import load_tool_events
        events = load_tool_events("A", daily_dir, today_iso=today_iso)
        assert [e["tool"] for e in events] == ["Edit", "Bash"]

    def test_ignores_non_dict_json_values(self, drawer):
        daily_dir, today_iso, _ = drawer
        path = daily_dir / f"{today_iso}.tools.jsonl"
        path.write_text(
            '[1, 2, 3]\n'
            '"just a string"\n'
            '{"session_id":"A","tool":"Read","input":{"file_path":"/x"}}\n',
            encoding="utf-8",
        )
        from flush import load_tool_events
        events = load_tool_events("A", daily_dir, today_iso=today_iso)
        assert len(events) == 1
        assert events[0]["tool"] == "Read"


class TestFormatToolEvents:
    def test_empty_events_returns_empty_string(self):
        from flush import format_tool_events
        assert format_tool_events([]) == ""

    def test_includes_count_summary_line(self):
        from flush import format_tool_events
        events = [
            {"tool": "Edit", "input": {"file_path": "/a"}, "ok": True},
            {"tool": "Edit", "input": {"file_path": "/b"}, "ok": True},
            {"tool": "Bash", "input": {"command": "ls"}, "ok": True},
        ]
        out = format_tool_events(events)
        assert "Tool calls this session: 3" in out
        assert "Edit: 2" in out
        assert "Bash: 1" in out

    def test_notable_operations_prioritize_high_signal_tools(self):
        """Edits + commands surface ahead of Reads regardless of order."""
        from flush import format_tool_events
        events = [
            {"tool": "Read", "input": {"file_path": "/r1"}, "ok": True},
            {"tool": "Read", "input": {"file_path": "/r2"}, "ok": True},
            {"tool": "Edit", "input": {"file_path": "/e1"}, "ok": True},
            {"tool": "Bash", "input": {"command": "rm -rf /tmp/x"}, "ok": True},
        ]
        out = format_tool_events(events)
        notable_section = out.split("Notable operations")[1]
        edit_pos = notable_section.index("[Edit]")
        bash_pos = notable_section.index("[Bash]")
        read_pos = notable_section.index("[Read]")
        # Priority-3 tools (Edit, Bash) must appear before priority-1 Read.
        assert edit_pos < read_pos
        assert bash_pos < read_pos

    def test_todowrite_excluded_from_notable(self):
        """Internal planning tools have priority 0 and are noise in the log."""
        from flush import format_tool_events
        events = [
            {"tool": "TodoWrite", "input": {"todo_count": 5}, "ok": True},
            {"tool": "Edit", "input": {"file_path": "/a"}, "ok": True},
        ]
        out = format_tool_events(events)
        assert "[Edit] /a" in out
        assert "[TodoWrite]" not in out  # priority-0 is filtered
        # But TodoWrite still appears in the counts line
        assert "TodoWrite: 1" in out

    def test_error_events_marked_in_output(self):
        from flush import format_tool_events
        events = [
            {"tool": "Bash", "input": {"command": "false"}, "ok": False},
        ]
        out = format_tool_events(events)
        assert "[ERROR]" in out

    def test_max_notable_cap(self):
        """Cap limits the notable-operations list length."""
        from flush import format_tool_events
        events = [
            {"tool": "Edit", "input": {"file_path": f"/f{i}"}, "ok": True}
            for i in range(100)
        ]
        out = format_tool_events(events, max_notable=5)
        # 5 notable bullets, period
        assert out.count("- [Edit]") == 5
        # Total count still shows the real number
        assert "Tool calls this session: 100" in out

    def test_describes_task_with_subagent_type(self):
        from flush import format_tool_events
        events = [
            {
                "tool": "Task",
                "input": {"description": "Compare repos", "subagent_type": "general-purpose"},
                "ok": True,
            },
        ]
        out = format_tool_events(events)
        assert "[Task]" in out
        assert "[general-purpose]" in out
        assert "Compare repos" in out

    def test_describes_grep_with_pattern_and_path(self):
        from flush import format_tool_events
        events = [
            {"tool": "Grep", "input": {"pattern": "class Foo", "path": "src/"}, "ok": True},
        ]
        out = format_tool_events(events)
        assert "class Foo in src/" in out

    def test_describes_webfetch_url(self):
        from flush import format_tool_events
        events = [
            {"tool": "WebFetch", "input": {"url": "https://example.com/docs"}, "ok": True},
        ]
        out = format_tool_events(events)
        assert "https://example.com/docs" in out
