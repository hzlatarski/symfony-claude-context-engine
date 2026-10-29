"""What a transcript contributes to the search index: plain message text only."""

from __future__ import annotations

import json
import sys
import types


def _line(entry: dict) -> str:
    return json.dumps(entry)


def test_user_and_assistant_text_are_kept_without_json() -> None:
    from transcript import record_text

    user = _line({"type": "user", "message": {"role": "user", "content": "deploy the fix"}})
    assistant = _line({
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
    })

    assert record_text(user) == ("user", "deploy the fix")
    assert record_text(assistant) == ("assistant", "Done.")


def test_tool_call_and_tool_output_are_kept_as_text() -> None:
    from transcript import record_text

    call = _line({"message": {"role": "assistant", "content": [
        {"type": "tool_use", "name": "Bash", "input": {"command": "git status"}},
    ]}})
    output = _line({"message": {"role": "user", "content": [
        {"type": "tool_result", "content": [{"type": "text", "text": "\x1b[33mM\x1b[0m file.php"}]},
    ]}})

    assert record_text(call) == ("assistant", '[tool call Bash] {"command": "git status"}')
    assert record_text(output) == ("user", "[tool output] M file.php")


def test_bookkeeping_events_and_images_are_dropped() -> None:
    from transcript import record_text

    assert record_text(_line({"type": "api-request", "body": {"model": "x"}})) is None
    assert record_text(_line({"type": "attachment", "attachment": {"content": "big"}})) is None
    assert record_text(_line({"message": {"role": "user", "content": [
        {"type": "image", "source": {"data": "AAAA"}},
    ]}})) is None


def test_thinking_is_kept() -> None:
    from transcript import record_text

    line = _line({"message": {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "pick B because A breaks the ban ladder"},
        {"type": "image", "source": {"data": "AAAA"}},
    ]}})

    assert record_text(line) == ("assistant", "[thinking] pick B because A breaks the ban ladder")


def test_unparseable_line_is_still_indexed() -> None:
    from transcript import record_text

    assert record_text("not json {") == ("event", "not json {")


def test_embed_indexes_only_message_text(tmp_path, monkeypatch) -> None:
    import transcript

    captured: dict = {}
    fake_store = types.SimpleNamespace(
        replace_chunks_for_source=lambda source, chunks: captured.update(source=source, chunks=chunks),
    )
    monkeypatch.setitem(sys.modules, "vector_store", fake_store)
    monkeypatch.setattr(transcript, "KNOWLEDGE_DIR", tmp_path)
    path = tmp_path / "daily" / "transcripts" / "s1.jsonl"
    path.parent.mkdir(parents=True)
    url = "https://example.com/" + "a" * 40
    path.write_text("\n".join([
        _line({"type": "api-request", "timestamp": "2026-10-28T10:00:00Z", "body": "x" * 5000}),
        _line({"timestamp": "2026-10-28T10:00:01Z", "message": {"role": "user", "content": f"see {url}"}}),
    ]), encoding="utf-8")

    transcript.embed_transcript_file(path)

    texts = [chunk["text"] for chunk in captured["chunks"]]
    assert captured["source"] == "daily/transcripts/s1.jsonl"
    assert texts == [f"user: see {url}", url]
    assert captured["chunks"][0]["chunk_id"] == "transcript-s1-000002-001"
    assert captured["chunks"][0]["metadata"]["date"] == "2026-10-28"


def test_string_items_in_list_content_are_kept() -> None:
    from transcript import record_text

    line = _line({"message": {"role": "user", "content": ["Never delete customer records"]}})

    assert record_text(line) == ("user", "Never delete customer records")


def test_summary_and_system_records_are_kept() -> None:
    from transcript import record_text

    summary = _line({"type": "summary", "summary": "User asked to keep backups", "leafUuid": "abc"})
    system = _line({"type": "system", "content": "\x1b[1mCompacted\x1b[0m conversation"})

    assert record_text(summary) == ("summary", "User asked to keep backups")
    assert record_text(system) == ("system", "Compacted conversation")
