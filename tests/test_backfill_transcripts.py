"""Historical Claude transcript backfill tests."""
from __future__ import annotations


def test_backfill_archives_and_indexes_selected_sessions(tmp_path, monkeypatch):
    from scripts import backfill_transcripts

    source_dir = tmp_path / "claude-project"
    source_dir.mkdir()
    wanted = source_dir / "wanted.jsonl"
    wanted.write_text('{"message":{"role":"user","content":"keep me"}}\n', encoding="utf-8")
    (source_dir / "other.jsonl").write_text("{}\n", encoding="utf-8")

    calls = []
    archive = tmp_path / "archive" / "wanted.jsonl"
    monkeypatch.setattr(
        backfill_transcripts,
        "archive_transcript",
        lambda source, session_id: calls.append(("archive", source.name, session_id))
        or archive,
    )
    monkeypatch.setattr(
        backfill_transcripts,
        "embed_transcript_file",
        lambda path: calls.append(("embed", path.name)) or 3,
    )

    result = backfill_transcripts.backfill(source_dir, session_ids={"wanted"})

    assert result == {
        "sessions": 1,
        "chunks": 3,
        "failed": 0,
        "failures": [],
    }
    assert calls == [
        ("archive", "wanted.jsonl", "wanted"),
        ("embed", "wanted.jsonl"),
    ]


def test_backfill_reports_bad_session_and_continues(tmp_path, monkeypatch):
    from scripts import backfill_transcripts

    source_dir = tmp_path / "claude-project"
    source_dir.mkdir()
    (source_dir / "bad.jsonl").write_text("{}\n", encoding="utf-8")
    (source_dir / "good.jsonl").write_text("{}\n", encoding="utf-8")
    archive = tmp_path / "archive" / "good.jsonl"
    embedded = []

    def archive_one(source, session_id):
        if session_id == "bad":
            raise ValueError("archive diverges from source")
        return archive

    monkeypatch.setattr(backfill_transcripts, "archive_transcript", archive_one)
    monkeypatch.setattr(
        backfill_transcripts,
        "embed_transcript_file",
        lambda path: embedded.append(path.name) or 4,
    )

    result = backfill_transcripts.backfill(source_dir)

    assert result == {
        "sessions": 1,
        "chunks": 4,
        "failed": 1,
        "failures": [
            {
                "session_id": "bad",
                "error": "archive diverges from source",
            }
        ],
    }
    assert embedded == ["good.jsonl"]


def test_default_source_dir_matches_current_project_encoding(tmp_path, monkeypatch):
    from scripts import backfill_transcripts

    projects = tmp_path / ".claude" / "projects"
    expected = projects / "c--wamp64-www-AiTutor"
    expected.mkdir(parents=True)
    monkeypatch.setattr(backfill_transcripts.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        backfill_transcripts,
        "PROJECT_ROOT",
        backfill_transcripts.Path(r"C:\wamp64\www\AiTutor"),
    )

    assert backfill_transcripts.default_source_dir() == expected
