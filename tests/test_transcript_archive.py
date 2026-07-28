"""Zero-loss transcript archive and indexing tests."""
from __future__ import annotations

import json


def _write_raw_transcript(path):
    rows = [
        {
            "timestamp": "2026-07-28T10:00:00+00:00",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Review https://claude.ai/code/artifact/exact-id",
                    },
                    {
                        "type": "tool_use",
                        "name": "Read",
                        "input": {"file_path": "C:/private/source.txt"},
                    },
                ],
            },
        },
        {
            "timestamp": "2026-07-28T10:01:00+00:00",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": "UNIQUE_TOOL_OUTPUT_9f834",
                    }
                ],
            },
        },
    ]
    raw = "\n".join(json.dumps(row) for row in rows) + "\n"
    path.write_text(raw, encoding="utf-8")
    return raw


def test_archive_transcript_is_byte_for_byte_and_idempotent(tmp_path):
    from scripts.transcript import archive_transcript

    source = tmp_path / "source.jsonl"
    expected = _write_raw_transcript(source)

    archive = archive_transcript(source, "session-123", tmp_path / "archives")
    assert archive.read_text(encoding="utf-8") == expected

    source.write_text(expected + '{"new": "line"}\n', encoding="utf-8")
    same_archive = archive_transcript(source, "session-123", tmp_path / "archives")
    assert same_archive == archive
    assert archive.read_text(encoding="utf-8") == expected + '{"new": "line"}\n'


def test_stale_shorter_snapshot_cannot_truncate_archive(tmp_path):
    from scripts.transcript import archive_transcript

    source = tmp_path / "source.jsonl"
    complete = _write_raw_transcript(source)
    archive = archive_transcript(source, "session-123", tmp_path / "archives")

    source.write_text(complete.splitlines(keepends=True)[0], encoding="utf-8")
    archive_transcript(source, "session-123", tmp_path / "archives")

    assert archive.read_text(encoding="utf-8") == complete


def test_divergent_snapshot_is_rejected_instead_of_overwriting(tmp_path):
    import pytest
    from scripts.transcript import archive_transcript

    source = tmp_path / "source.jsonl"
    expected = _write_raw_transcript(source)
    archive = archive_transcript(source, "session-123", tmp_path / "archives")
    source.write_text('{"different": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="diverges"):
        archive_transcript(source, "session-123", tmp_path / "archives")

    assert archive.read_text(encoding="utf-8") == expected


def test_embed_transcript_indexes_text_tool_inputs_and_tool_results(tmp_path, monkeypatch):
    from scripts import transcript

    knowledge = tmp_path / "knowledge"
    archive_dir = knowledge / "daily" / "transcripts"
    archive_dir.mkdir(parents=True)
    source = archive_dir / "session-123.jsonl"
    _write_raw_transcript(source)

    indexed = []
    monkeypatch.setattr(transcript, "KNOWLEDGE_DIR", knowledge)
    import vector_store
    monkeypatch.setattr(
        vector_store,
        "replace_chunks_for_source",
        lambda source_file, chunks: indexed.extend(chunks),
    )

    count = transcript.embed_transcript_file(source)

    assert count == len(indexed) > 0
    assert all(
        item["source_file"] == "daily/transcripts/session-123.jsonl"
        for item in indexed
    )
    combined = "\n".join(item["text"] for item in indexed)
    assert "https://claude.ai/code/artifact/exact-id" in combined
    assert "C:/private/source.txt" in combined
    assert "UNIQUE_TOOL_OUTPUT_9f834" in combined
    assert all(item["metadata"]["date"] == "2026-07-28" for item in indexed)


def test_embed_transcript_dates_each_record_from_its_own_timestamp(
    tmp_path,
    monkeypatch,
):
    from scripts import transcript

    knowledge = tmp_path / "knowledge"
    archive_dir = knowledge / "daily" / "transcripts"
    archive_dir.mkdir(parents=True)
    source = archive_dir / "session-multiday.jsonl"
    source.write_text(
        "\n".join([
            json.dumps({
                "timestamp": "2026-07-28T23:59:00+00:00",
                "message": {"role": "assistant", "content": "first-day-marker"},
            }),
            json.dumps({
                "timestamp": "2026-07-29T00:01:00+00:00",
                "message": {"role": "user", "content": "second-day-marker"},
            }),
        ]) + "\n",
        encoding="utf-8",
    )

    indexed = []
    monkeypatch.setattr(transcript, "KNOWLEDGE_DIR", knowledge)
    import vector_store
    monkeypatch.setattr(
        vector_store,
        "replace_chunks_for_source",
        lambda source_file, chunks: indexed.extend(chunks),
    )

    transcript.embed_transcript_file(source)

    first = next(item for item in indexed if "first-day-marker" in item["text"])
    second = next(item for item in indexed if "second-day-marker" in item["text"])
    assert first["metadata"]["date"] == "2026-07-28"
    assert second["metadata"]["date"] == "2026-07-29"


def test_long_url_is_indexed_whole_even_when_it_crosses_chunk_boundary(
    tmp_path,
    monkeypatch,
):
    from scripts import transcript

    knowledge = tmp_path / "knowledge"
    archive_dir = knowledge / "daily" / "transcripts"
    archive_dir.mkdir(parents=True)
    source = archive_dir / "session-long.jsonl"
    long_url = "https://example.com/" + ("a" * 700)
    payload = ("x" * 5_900) + long_url
    source.write_text(
        json.dumps({
            "timestamp": "2026-07-28T10:00:00+00:00",
            "message": {"role": "assistant", "content": payload},
        }) + "\n",
        encoding="utf-8",
    )

    indexed = []
    monkeypatch.setattr(transcript, "KNOWLEDGE_DIR", knowledge)
    import vector_store
    monkeypatch.setattr(
        vector_store,
        "replace_chunks_for_source",
        lambda source_file, chunks: indexed.extend(chunks),
    )

    transcript.embed_transcript_file(source)

    assert any(long_url in item["text"] for item in indexed)


def test_exact_reference_documents_are_deduplicated_per_session(tmp_path, monkeypatch):
    from scripts import transcript

    knowledge = tmp_path / "knowledge"
    archive_dir = knowledge / "daily" / "transcripts"
    archive_dir.mkdir(parents=True)
    source = archive_dir / "session-refs.jsonl"
    url = "https://example.com/repeated-reference"
    source.write_text(
        "\n".join([
            json.dumps({"message": {"role": "user", "content": url}}),
            json.dumps({"message": {"role": "assistant", "content": url}}),
        ]) + "\n",
        encoding="utf-8",
    )

    indexed = []
    monkeypatch.setattr(transcript, "KNOWLEDGE_DIR", knowledge)
    import vector_store
    monkeypatch.setattr(
        vector_store,
        "replace_chunks_for_source",
        lambda source_file, chunks: indexed.extend(chunks),
    )

    transcript.embed_transcript_file(source)

    exact = [
        item for item in indexed
        if item["metadata"]["section"].startswith("exact reference")
        and item["text"] == url
    ]
    assert len(exact) == 1


def test_exact_urls_are_appended_without_llm_cooperation():
    from flush import append_exact_references

    context = """
    First https://example.com/a.
    Duplicate https://example.com/a.
    Artifact https://claude.ai/code/artifact/exact-id
    """
    result = append_exact_references("Concise summary without links.", context)

    assert "Concise summary without links." in result
    assert result.count("https://example.com/a") == 1
    assert "https://claude.ai/code/artifact/exact-id" in result
