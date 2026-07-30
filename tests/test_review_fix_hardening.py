"""Regressions for the 2026-07-30 review of the zero-loss remediation.

Each test here pins a defect found while verifying
``MEMORY-COMPILER-REVIEW-AND-REMEDIATION-PLAN.md`` and
``ZERO-LOSS-TRANSCRIPT-INGESTION.md`` against the implementation that claimed
to satisfy them.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _write_two_turn_transcript(path: Path) -> None:
    rows = [
        {
            "timestamp": "2026-07-30T10:00:00+00:00",
            "message": {"role": "user", "content": "first turn text"},
        },
        {
            "timestamp": "2026-07-30T10:01:00+00:00",
            "message": {"role": "assistant", "content": "second turn text"},
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def _load_hook(hook_name: str):
    hook_path = Path(__file__).parents[1] / "hooks" / f"{hook_name}.py"
    spec = importlib.util.spec_from_file_location(
        f"hardening_{hook_name.replace('-', '_')}", hook_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ── 1. The authoritative archive must be durable, like the cursor ──────


def test_transcript_archive_is_fsynced_before_publish(tmp_path, monkeypatch):
    """IR15 fsynced the daily log and cursor but left the archive cached."""
    import transcript

    calls: list[str] = []
    monkeypatch.setattr(transcript.os, "fsync", lambda _fd: calls.append("fsync"))

    source = tmp_path / "live.jsonl"
    source.write_bytes(b'{"a":1}\n')

    transcript.archive_transcript(source, "sess-1", archive_dir=tmp_path / "arch")

    assert calls == ["fsync"], "archive content was published without an fsync"


# ── 2. A corrupt marker must not poison the whole retry queue ──────────


def test_corrupt_pending_marker_is_quarantined_not_fatal(tmp_path):
    """IR11 claims automatic retry; one bad record must not stop every job."""
    from pending_flush import create_pending_flush, load_pending_flushes

    corrupt = tmp_path / "aaa-corrupt.md.pending.json"
    corrupt.write_text("{not valid json", encoding="utf-8")

    good = create_pending_flush(
        tmp_path, "session-flush", "ctx", "sess-good", 5, tmp_path / "a.jsonl"
    )

    jobs = load_pending_flushes(tmp_path)

    assert [job["context_file"] for _marker, job in jobs] == [str(good)]
    assert not corrupt.exists()
    assert list(tmp_path.glob("*.corrupt")), "corrupt marker was not quarantined"


def test_incomplete_pending_marker_is_quarantined_not_fatal(tmp_path):
    from pending_flush import load_pending_flushes

    marker = tmp_path / "bbb-incomplete.md.pending.json"
    marker.write_text(json.dumps({"session_id": "s"}), encoding="utf-8")

    assert load_pending_flushes(tmp_path) == []
    assert list(tmp_path.glob("*.corrupt"))


# ── 3. Pending jobs need a claim, a ceiling, and a real ordering ───────


def test_exhausted_pending_job_is_quarantined(tmp_path):
    """A permanently failing job must stop being respawned forever."""
    from pending_flush import (
        MAX_FLUSH_ATTEMPTS,
        load_pending_flushes,
        queue_pending_flush,
    )

    context = tmp_path / "session-flush-exhausted.md"
    marker = queue_pending_flush(
        context,
        "sess-dead",
        3,
        tmp_path / "a.jsonl",
        context="payload",
        attempts=MAX_FLUSH_ATTEMPTS,
    )

    assert load_pending_flushes(tmp_path) == []
    assert not marker.exists()
    assert list(tmp_path.glob("*.exhausted"))


def test_loading_a_job_does_not_consume_an_attempt(tmp_path):
    """Only a confirmed worker failure may count — a spawn that never ran
    must not burn the budget, or a hook crash silently discards real work."""
    import pending_flush

    context = tmp_path / "session-flush-claim.md"
    marker = pending_flush.queue_pending_flush(
        context, "sess-claim", 4, tmp_path / "a.jsonl", context="payload"
    )

    # Hand the job out many times over, expiring the spawn lease between rounds
    # so it is genuinely re-offered rather than merely suppressed.
    for _ in range(10):
        jobs = pending_flush.load_pending_flushes(tmp_path)
        assert len(jobs) == 1, "job was retired without ever running"
        assert jobs[0][1].get("attempts", 0) == 0

        payload = json.loads(marker.read_text(encoding="utf-8"))
        payload["spawned_at"] = 0.0
        marker.write_text(json.dumps(payload), encoding="utf-8")

    assert json.loads(marker.read_text(encoding="utf-8"))["attempts"] == 0


def test_worker_failure_records_an_attempt(tmp_path):
    from pending_flush import (
        load_pending_flushes,
        queue_pending_flush,
        record_failed_attempt,
    )

    context = tmp_path / "session-flush-fails.md"
    queue_pending_flush(
        context, "sess-fails", 4, tmp_path / "a.jsonl", context="payload"
    )

    assert record_failed_attempt(context) == 1
    assert record_failed_attempt(context) == 2
    assert load_pending_flushes(tmp_path)[0][1]["attempts"] == 2


def test_worker_exception_records_an_attempt(tmp_path, monkeypatch):
    """An unhandled crash is a real failure; if it does not count, the job
    retries forever with an attempt budget that never moves."""
    import flush
    from pending_flush import load_pending_flushes, queue_pending_flush

    context = tmp_path / "session-flush-boom.md"
    queue_pending_flush(
        context, "sess-boom", 4, tmp_path / "a.jsonl", context="payload"
    )

    def exploding_main():
        raise OSError("sharing violation reading context")

    monkeypatch.setattr(flush, "main", exploding_main)
    monkeypatch.setattr(sys, "argv", ["flush.py", str(context), "sess-boom"])

    # The traceback still propagates (so the process exits non-zero and the
    # cause is visible), but the attempt must be counted on the way out.
    with pytest.raises(OSError):
        flush._main_with_attempt_accounting()

    assert load_pending_flushes(tmp_path)[0][1]["attempts"] == 1


def test_recently_spawned_job_is_not_handed_out_again(tmp_path):
    """Two hooks firing back to back must not both launch the same worker."""
    from pending_flush import load_pending_flushes, queue_pending_flush

    context = tmp_path / "session-flush-lease.md"
    queue_pending_flush(
        context, "sess-lease", 4, tmp_path / "a.jsonl", context="payload"
    )

    assert len(load_pending_flushes(tmp_path)) == 1
    assert load_pending_flushes(tmp_path) == [], "job was handed out twice"


def test_stale_spawn_lease_is_reclaimed(tmp_path):
    """A worker that never ran must come back — the lease suppresses
    duplicates, it must never strand a job."""
    import pending_flush

    context = tmp_path / "session-flush-stale.md"
    pending_flush.queue_pending_flush(
        context, "sess-stale", 4, tmp_path / "a.jsonl", context="payload"
    )

    assert len(pending_flush.load_pending_flushes(tmp_path)) == 1
    assert pending_flush.load_pending_flushes(tmp_path) == []

    marker = pending_flush.pending_job_path(context)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["spawned_at"] = payload["spawned_at"] - (
        pending_flush.SPAWN_LEASE_SECONDS + 60
    )
    marker.write_text(json.dumps(payload), encoding="utf-8")

    reclaimed = pending_flush.load_pending_flushes(tmp_path)
    assert len(reclaimed) == 1
    assert reclaimed[0][1]["session_id"] == "sess-stale"


def test_unreadable_marker_is_left_for_retry_not_destroyed(tmp_path):
    """A transient I/O error is not corruption; deleting the marker would
    permanently lose the only record of the session, cursor, and archive."""
    import pending_flush

    context = tmp_path / "session-flush-locked.md"
    marker = pending_flush.queue_pending_flush(
        context, "sess-io", 9, tmp_path / "a.jsonl", context="payload"
    )

    original = Path.read_text

    def flaky(self, *args, **kwargs):
        if self == marker:
            raise OSError("sharing violation")
        return original(self, *args, **kwargs)

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "read_text", flaky)
        assert pending_flush.load_pending_flushes(tmp_path) == []

    assert marker.exists(), "a transiently unreadable marker was destroyed"
    assert not list(tmp_path.glob("*.corrupt"))
    assert pending_flush.load_pending_flushes(tmp_path)[0][1]["session_id"] == "sess-io"


def test_failed_quarantine_never_deletes_the_marker(tmp_path, monkeypatch):
    import pending_flush

    marker = tmp_path / "ccc-bad.md.pending.json"
    marker.write_text("{broken", encoding="utf-8")

    def refuse(*_args, **_kwargs):
        raise OSError("rename refused")

    monkeypatch.setattr(pending_flush.os, "replace", refuse)

    assert pending_flush.load_pending_flushes(tmp_path) == []
    assert marker.exists(), "quarantine failure fell back to deleting the job"


def test_pending_jobs_are_returned_oldest_first(tmp_path):
    """The docstring promises oldest-first; filenames sort by random UUID."""
    from pending_flush import load_pending_flushes, queue_pending_flush

    queue_pending_flush(
        tmp_path / "zzz-older.md",
        "sess-older",
        1,
        tmp_path / "a.jsonl",
        context="older",
        created_at=100.0,
    )
    queue_pending_flush(
        tmp_path / "aaa-newer.md",
        "sess-newer",
        2,
        tmp_path / "b.jsonl",
        context="newer",
        created_at=200.0,
    )

    jobs = load_pending_flushes(tmp_path)

    assert [job["session_id"] for _marker, job in jobs] == [
        "sess-older",
        "sess-newer",
    ]


# ── 4. The Chroma migration marker must survive a power loss ───────────


def test_migration_marker_is_fsynced(tmp_path, monkeypatch):
    """IR05 promises crash-resume; an unsynced marker wedges the store."""
    import chroma_path

    root = tmp_path / "chroma"
    (root / "12345678-1234-1234-1234-123456789abc").mkdir(parents=True)
    (root / "chroma.sqlite3").write_bytes(b"db")

    calls: list[str] = []
    monkeypatch.setattr(chroma_path.os, "fsync", lambda _fd: calls.append("fsync"))

    chroma_path.ensure_active_chroma_dir(root)

    assert calls, "migration marker was written without an fsync"


# ── 5. A legitimate no-op compile must not retry forever ───────────────


def _drive_compile_writing(tmp_path, monkeypatch, writer):
    import compile as compiler

    daily = tmp_path / "daily"
    concepts = tmp_path / "concepts"
    connections = tmp_path / "connections"
    for directory in (daily, concepts, connections):
        directory.mkdir()
    log = daily / "2026-07-30.md"
    log.write_text("# source", encoding="utf-8")
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# schema", encoding="utf-8")

    monkeypatch.setattr(compiler, "AGENTS_FILE", agents)
    monkeypatch.setattr(compiler, "CONCEPTS_DIR", concepts)
    monkeypatch.setattr(compiler, "CONNECTIONS_DIR", connections)
    monkeypatch.setattr(compiler, "KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(compiler, "COMPILED_TRUTH_FILE", tmp_path / "truth.md")
    monkeypatch.setattr(compiler, "update_state", lambda _mutator: None)
    monkeypatch.setattr(
        compiler,
        "read_wiki_index",
        lambda *, compact=False: "compact-index",
    )
    monkeypatch.setattr(compiler.dedup, "similar_to_text", lambda *_a, **_k: [])
    monkeypatch.setattr(
        compiler.dedup, "format_preflight_block", lambda *_a, **_k: ""
    )

    def fake_run(*_args, **_kwargs):
        writer(tmp_path)
        return subprocess.CompletedProcess(
            args=["claude"], returncode=0, stdout="done", stderr=""
        )

    monkeypatch.setattr(compiler.subprocess, "run", fake_run)
    state: dict = {"ingested_daily": {}}
    result = asyncio.run(compiler.compile_daily_log(log, state))
    return state, result


def test_compile_accepts_a_credited_log_only_mutation(tmp_path, monkeypatch):
    """A source fully covered by existing articles still credits log.md."""

    def write_log(root: Path) -> None:
        (root / "log.md").write_text(
            "## [2026-07-30] compile | 2026-07-30.md\n"
            "- Source: daily/2026-07-30.md\n",
            encoding="utf-8",
        )

    state, result = _drive_compile_writing(tmp_path, monkeypatch, write_log)

    assert result is True
    assert list(state["ingested_daily"]) == ["2026-07-30.md"]


def test_compile_still_rejects_an_uncredited_mutation(tmp_path, monkeypatch):
    """Broadening the gate must not accept unrelated churn."""

    def write_unrelated(root: Path) -> None:
        (root / "log.md").write_text("## unrelated entry\n", encoding="utf-8")

    state, result = _drive_compile_writing(tmp_path, monkeypatch, write_unrelated)

    assert result is False
    assert state["ingested_daily"] == {}


def test_compile_rejects_a_bare_mention_without_a_compile_entry(
    tmp_path, monkeypatch
):
    """Naming the source in passing is not proof that it was compiled."""

    def write_mention(root: Path) -> None:
        (root / "log.md").write_text(
            "## [2026-07-30] lint\n- skipped daily/2026-07-30.md\n",
            encoding="utf-8",
        )

    state, result = _drive_compile_writing(tmp_path, monkeypatch, write_mention)

    assert result is False
    assert state["ingested_daily"] == {}


def test_compile_requires_a_compile_entry_not_just_a_source_line(
    tmp_path, monkeypatch
):
    """A lone Source: line is not the instructed compile entry."""

    def write_source_line_only(root: Path) -> None:
        (root / "log.md").write_text(
            "## [2026-07-30] some other activity\n"
            "- Source: daily/2026-07-30.md\n",
            encoding="utf-8",
        )

    state, result = _drive_compile_writing(
        tmp_path, monkeypatch, write_source_line_only
    )

    assert result is False
    assert state["ingested_daily"] == {}


def test_compile_accepts_the_heading_with_a_daily_prefix(tmp_path, monkeypatch):
    """The heading is the required signal; tolerate the daily/ prefix form."""

    def write_prefixed_heading(root: Path) -> None:
        (root / "log.md").write_text(
            "## [2026-07-30T12:00:00] compile | daily/2026-07-30.md\n",
            encoding="utf-8",
        )

    state, result = _drive_compile_writing(
        tmp_path, monkeypatch, write_prefixed_heading
    )

    assert result is True
    assert list(state["ingested_daily"]) == ["2026-07-30.md"]


def test_compile_rejects_an_old_occurrence_in_a_rewritten_log(
    tmp_path, monkeypatch
):
    """A truncated or rewritten log.md must not credit a pre-existing entry."""
    log = tmp_path / "log.md"

    def rewrite_log(root: Path) -> None:
        # The prior content already credited the source; the run replaces the
        # file rather than appending, so nothing new was actually recorded.
        (root / "log.md").write_text(
            "## [2026-01-01] compile | 2026-07-30.md\n"
            "- Source: daily/2026-07-30.md\n",
            encoding="utf-8",
        )

    log.write_text(
        "## [2026-01-01] compile | 2026-07-30.md\n"
        "- Source: daily/2026-07-30.md\n"
        "## [2026-01-02] compile | other.md\n",
        encoding="utf-8",
    )

    state, result = _drive_compile_writing(tmp_path, monkeypatch, rewrite_log)

    assert result is False
    assert state["ingested_daily"] == {}


def test_compile_skips_a_source_after_max_failed_attempts(tmp_path):
    """Repeated failure must stop costing an LLM call on every run."""
    import compile as compiler

    log = tmp_path / "2026-07-30.md"
    log.write_text("# source", encoding="utf-8")
    digest = compiler.file_hash(log)
    state = {
        "ingested_daily": {},
        "failed_daily": {
            "2026-07-30.md": {
                "hash": digest,
                "attempts": compiler.MAX_COMPILE_ATTEMPTS,
            }
        },
    }

    assert compiler.select_logs_to_compile([log], state) == []


def test_exhausted_sources_are_not_reported_as_up_to_date(tmp_path, monkeypatch, capsys):
    """Silently calling a never-compiled source 'up to date' hides real loss."""
    import compile as compiler

    log = tmp_path / "2026-07-30.md"
    log.write_text("# never compiled", encoding="utf-8")
    digest = compiler.file_hash(log)

    monkeypatch.setattr(compiler, "list_raw_files", lambda: [log])
    monkeypatch.setattr(
        compiler,
        "update_state",
        lambda _mutator: {
            "ingested_daily": {},
            "failed_daily": {
                "2026-07-30.md": {
                    "hash": digest,
                    "attempts": compiler.MAX_COMPILE_ATTEMPTS,
                }
            },
        },
    )

    args = argparse.Namespace(all=False, file=None, dry_run=False)
    status = compiler._main_unlocked(args)
    output = capsys.readouterr()
    combined = output.out + output.err

    assert "up to date" not in combined
    assert "2026-07-30.md" in combined
    assert status != 0, "quarantined sources were reported as success"


def test_compile_retries_a_failed_source_that_changed(tmp_path):
    import compile as compiler

    log = tmp_path / "2026-07-30.md"
    log.write_text("# edited source", encoding="utf-8")
    state = {
        "ingested_daily": {},
        "failed_daily": {
            "2026-07-30.md": {
                "hash": "stale-hash-from-before-the-edit",
                "attempts": compiler.MAX_COMPILE_ATTEMPTS,
            }
        },
    }

    assert compiler.select_logs_to_compile([log], state) == [log]


# ── 6. A swallowed daily-index failure must leave a repair record ──────


def test_daily_index_failure_records_a_stale_source(tmp_path, monkeypatch):
    """The append/index pair is not atomic, so failure must be recoverable."""
    import flush
    import utils

    monkeypatch.setattr(flush, "DAILY_DIR", tmp_path / "daily")
    monkeypatch.setattr(utils, "STATE_FILE", tmp_path / "state.json")

    def boom(_path):
        raise RuntimeError("chroma unavailable")

    monkeypatch.setattr(utils, "embed_daily_file", boom)

    flush.append_to_daily_log("entry that made it to disk")

    state = utils.load_state(tmp_path / "state.json")
    stale = state.get("stale_daily_index", {})
    assert stale, "a failed daily embed left no repair record"
    assert all(name.endswith(".md") for name in stale)


def test_successful_daily_index_clears_a_stale_record(tmp_path, monkeypatch):
    """A later successful embed must retire the repair record it left."""
    import flush
    import utils

    monkeypatch.setattr(flush, "DAILY_DIR", tmp_path / "daily")
    monkeypatch.setattr(utils, "STATE_FILE", tmp_path / "state.json")

    def boom(_path):
        raise RuntimeError("chroma unavailable")

    monkeypatch.setattr(utils, "embed_daily_file", boom)
    flush.append_to_daily_log("entry written while chroma was down")
    assert utils.load_state(tmp_path / "state.json")["stale_daily_index"]

    monkeypatch.setattr(utils, "embed_daily_file", lambda _path: 3)
    flush.append_to_daily_log("entry written after chroma recovered")

    state = utils.load_state(tmp_path / "state.json")
    assert not state.get("stale_daily_index")


# ── 7. Distinct session IDs must not share one archive ─────────────────


def test_distinct_session_ids_do_not_share_an_archive(tmp_path):
    """IR13 hardened pending paths but left the archive namespace lossy."""
    import transcript

    first_source = tmp_path / "a.jsonl"
    first_source.write_bytes(b'{"x":1}\n')
    second_source = tmp_path / "b.jsonl"
    second_source.write_bytes(b'{"y":2}\n')
    archive_dir = tmp_path / "arch"

    first = transcript.archive_transcript(first_source, "s/1", archive_dir=archive_dir)
    second = transcript.archive_transcript(second_source, "s?1", archive_dir=archive_dir)

    assert first != second
    assert first.read_bytes() == b'{"x":1}\n'
    assert second.read_bytes() == b'{"y":2}\n'


def test_uuid_session_id_keeps_its_plain_archive_name(tmp_path):
    """Existing recovered archives must not be orphaned by the fix."""
    import transcript

    source = tmp_path / "live.jsonl"
    source.write_bytes(b'{"x":1}\n')

    archived = transcript.archive_transcript(
        source,
        "8f013255-edee-4b5f-b461-63de65dcd0ab",
        archive_dir=tmp_path / "arch",
    )

    assert archived.name == "8f013255-edee-4b5f-b461-63de65dcd0ab.jsonl"


# ── 8. Bounded extraction must honor its documented ceiling ────────────


def test_bounded_extraction_respects_a_limit_smaller_than_the_first_turn(tmp_path):
    import transcript

    source = tmp_path / "live.jsonl"
    _write_two_turn_transcript(source)

    context, cursor, count = transcript.extract_conversation_context(
        source, max_context_chars=1
    )

    assert context == ""
    assert cursor == 0
    assert count == 0


def test_unbounded_extraction_still_returns_every_fresh_turn(tmp_path):
    import transcript

    source = tmp_path / "live.jsonl"
    _write_two_turn_transcript(source)

    context, cursor, count = transcript.extract_conversation_context(source)

    assert cursor == 2
    assert count == 2
    assert "first turn text" in context
    assert "second turn text" in context


# ── 9. The update check must not depend on bare `uv` ──────────────────


def test_update_check_uses_current_python(monkeypatch):
    """F11 removed bare `uv` from the capture path but not from here."""
    module = _load_hook("session-start")

    captured: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        captured.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module.get_update_notice()

    assert captured, "update check did not spawn a child process"
    assert captured[0][0] == sys.executable
