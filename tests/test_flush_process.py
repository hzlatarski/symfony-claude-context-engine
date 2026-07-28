"""Regression tests for flush process and child-command handling."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace


def test_nonzero_claude_exit_with_stdout_is_flush_error(monkeypatch) -> None:
    import flush

    monkeypatch.setattr(
        flush.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess(
            ["claude"], 1, stdout="Prompt is too long", stderr="",
        ),
    )

    response, cost = asyncio.run(flush.run_flush("important context"))

    assert response == "FLUSH_ERROR: claude CLI exited 1"
    assert cost == 0.0


def test_compilation_child_uses_current_python(tmp_path, monkeypatch) -> None:
    import flush

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "compile.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(flush, "SCRIPTS_DIR", scripts)
    monkeypatch.setattr(flush, "ROOT", tmp_path)
    monkeypatch.setattr(flush, "COMPILE_AFTER_HOUR", 0)
    monkeypatch.setattr(flush, "DAILY_DIR", tmp_path / "daily")
    spawned = []
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda command, **kwargs: spawned.append(command),
    )

    flush.maybe_trigger_compilation()

    assert spawned
    assert spawned[0][0] == sys.executable
    assert "uv" not in spawned[0]


def test_daily_append_transaction_is_serial_across_sessions(
    tmp_path, monkeypatch,
) -> None:
    import flush
    import utils

    monkeypatch.setattr(flush, "DAILY_DIR", tmp_path / "daily")
    active = 0
    maximum_active = 0
    guard = threading.Lock()
    ready = threading.Barrier(3)

    def slow_embed(_path):
        nonlocal active, maximum_active
        with guard:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.03)
        with guard:
            active -= 1
        return 1

    monkeypatch.setattr(utils, "embed_daily_file", slow_embed)

    def worker(content):
        ready.wait()
        flush.append_to_daily_log(content)

    threads = [
        threading.Thread(target=worker, args=("SESSION A",)),
        threading.Thread(target=worker, args=("SESSION B",)),
    ]
    for thread in threads:
        thread.start()
    ready.wait()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    daily_files = list((tmp_path / "daily").glob("*.md"))
    assert len(daily_files) == 1
    content = daily_files[0].read_text(encoding="utf-8")
    assert "SESSION A" in content
    assert "SESSION B" in content
    assert maximum_active == 1


def test_flush_metadata_updates_do_not_lose_cross_session_costs(
    tmp_path, monkeypatch,
) -> None:
    import flush

    monkeypatch.setattr(flush, "STATE_FILE", tmp_path / "last-flush.json")
    ready = threading.Barrier(3)

    def worker(session_id):
        ready.wait()
        flush.update_flush_state(
            lambda state: state.setdefault("flush_costs", []).append(
                {"session_id": session_id},
            ),
        )

    threads = [
        threading.Thread(target=worker, args=("a",)),
        threading.Thread(target=worker, args=("b",)),
    ]
    for thread in threads:
        thread.start()
    ready.wait()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    sessions = {
        item["session_id"]
        for item in flush.load_flush_state()["flush_costs"]
    }
    assert sessions == {"a", "b"}


def test_raw_embed_failure_is_nonzero_and_keeps_retry_context(
    tmp_path, monkeypatch,
) -> None:
    import flush

    context_file = tmp_path / "session-flush-test.md"
    context_file.write_text("important context", encoding="utf-8")
    transcript = tmp_path / "archive.jsonl"
    transcript.write_text('{"type":"user","message":"keep me"}\n', encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["flush.py", str(context_file), "session", "1", str(transcript)],
    )
    monkeypatch.setitem(
        sys.modules,
        "transcript",
        SimpleNamespace(
            embed_transcript_file=lambda _path: (_ for _ in ()).throw(
                RuntimeError("index unavailable")
            )
        ),
    )
    monkeypatch.setattr(flush, "session_flush_lock", lambda _session: _NullContext())

    assert flush.main() == 1
    assert context_file.exists()


def test_successful_raw_only_retry_removes_durable_job(
    tmp_path, monkeypatch,
) -> None:
    import flush
    from pending_flush import pending_job_path, queue_pending_flush

    context_file = tmp_path / "session-flush-test.md"
    context_file.write_text("", encoding="utf-8")
    transcript = tmp_path / "archive.jsonl"
    transcript.write_text('{"type":"tool_result","result":"keep me"}\n', encoding="utf-8")
    queue_pending_flush(context_file, "session", 0, transcript)
    monkeypatch.setattr(
        sys,
        "argv",
        ["flush.py", str(context_file), "session", "0", str(transcript)],
    )
    monkeypatch.setitem(
        sys.modules,
        "transcript",
        SimpleNamespace(embed_transcript_file=lambda _path: 1),
    )
    monkeypatch.setattr(flush, "session_flush_lock", lambda _session: _NullContext())
    monkeypatch.setattr(flush, "load_cursor", lambda _session: -1)

    assert flush.main() == 0
    assert not context_file.exists()
    assert not pending_job_path(context_file).exists()


def test_pending_marker_can_recover_failed_context_publication(
    tmp_path, monkeypatch,
) -> None:
    import pending_flush

    archive = tmp_path / "archive.jsonl"
    archive.write_text("{}\n", encoding="utf-8")
    real_write = pending_flush._write_context_atomic
    monkeypatch.setattr(
        pending_flush,
        "_write_context_atomic",
        lambda *_args: (_ for _ in ()).throw(OSError("context publish failed")),
    )

    with __import__("pytest").raises(OSError, match="context publish failed"):
        pending_flush.create_pending_flush(
            tmp_path,
            "session-flush",
            "important context",
            "../../unsafe",
            1,
            archive,
        )

    marker = next(tmp_path.glob("*.md.pending.json"))
    payload = __import__("json").loads(marker.read_text(encoding="utf-8"))
    assert payload["context"] == "important context"
    context_file = Path(payload["context_file"])
    assert not context_file.exists()

    monkeypatch.setattr(pending_flush, "_write_context_atomic", real_write)
    jobs = pending_flush.load_pending_flushes(tmp_path)

    assert len(jobs) == 1
    assert context_file.read_text(encoding="utf-8") == "important context"


def test_daily_append_is_fsynced_before_success(
    tmp_path, monkeypatch,
) -> None:
    import flush
    import utils

    calls = []
    monkeypatch.setattr(flush, "DAILY_DIR", tmp_path / "daily")
    monkeypatch.setattr(flush.os, "fsync", lambda _fd: calls.append("fsync"))
    monkeypatch.setattr(utils, "embed_daily_file", lambda _path: 1)

    flush.append_to_daily_log("durable entry")

    assert calls == ["fsync"]


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False
