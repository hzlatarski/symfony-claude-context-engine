"""Cross-process writes must never erase another process's vectors.

Reproduces the 2026-10-08 live-store loss (see scripts/chroma_client.py):
a process loads the index, another process writes and saves, then the first
process writes. Runs real processes because the bug only exists between them.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

WORKER = Path(__file__).parent / "fixtures" / "chroma_mp_worker.py"


def _run(role: str, store: Path, mode: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(WORKER), role, str(store), mode],
        capture_output=True, text=True, timeout=180,
    )


def _scenario(tmp_path: Path, mode: str) -> dict:
    store = tmp_path / "store"
    assert _run("seed", store, mode).returncode == 0
    stale = subprocess.Popen(
        [sys.executable, str(WORKER), "stale", str(store), mode],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert stale.stdout.readline().strip() == "loaded"
    writer = _run("writer", store, mode)
    assert writer.returncode == 0, writer.stderr
    (tmp_path / "go.flag").write_text("go")
    stale.wait(timeout=180)
    check = _run("check", store, mode)
    if check.returncode != 0:  # a corrupted store can crash the reader outright
        return {"crashed": True, "unreadable": None, "self_hits": 0}
    return json.loads(check.stdout.strip().splitlines()[-1])


def test_write_session_keeps_other_process_vectors(tmp_path):
    result = _scenario(tmp_path, "session")
    assert result.get("crashed") is None, result
    assert result["unreadable"] == []
    assert result["self_hits"] == 20
    assert result["count"] == 85


def test_harness_detects_the_bug_without_write_session(tmp_path):
    """Guards the test itself: the pre-fix write path must still fail here."""
    result = _scenario(tmp_path, "naive")
    lost = result.get("crashed") or bool(result["unreadable"]) or result["self_hits"] < 20
    assert lost, result


def test_every_store_write_reloads_from_disk(tmp_path, monkeypatch):
    """Each write path must use chroma_client.write_session (reset under lock)."""
    import chroma_client
    import codebase_store
    import config
    import vector_store

    monkeypatch.setenv("MC_EMBEDDER", "minilm")  # fast; the path is what matters
    monkeypatch.setattr(config, "CHROMA_DB_DIR", tmp_path / "chroma")
    chroma_client.reset()
    resets: list[str] = []
    real_reset = chroma_client.reset
    monkeypatch.setattr(chroma_client, "reset", lambda: (resets.append("r"), real_reset())[1])

    meta = {"type": "fact", "confidence": 0.9, "quarantined": False, "updated": "2026-10-08"}
    chunk = {"chunk_id": "d#1", "source_file": "daily/x.md", "text": "t", "metadata": {"date": "2026-10-08"}}
    writes = {
        "upsert_article": lambda: vector_store.upsert_article("concepts/a", "A", "observed", "x", meta),
        "replace_article": lambda: vector_store.replace_article("concepts/a", "A", {"observed": "y"}, meta),
        "delete_article": lambda: vector_store.delete_article("concepts/a"),
        "upsert_chunk": lambda: vector_store.upsert_chunk("d#0", "daily/x.md", "t", {"date": "2026-10-08"}),
        "upsert_chunks": lambda: vector_store.upsert_chunks([chunk]),
        "replace_chunks_for_source": lambda: vector_store.replace_chunks_for_source("daily/x.md", [chunk]),
        "delete_chunks_for_daily": lambda: vector_store.delete_chunks_for_daily("daily/x.md"),
        "codebase.upsert_chunk": lambda: codebase_store.upsert_chunk("f::0", "f.php", "code", {}),
        "codebase.replace_chunks_for_file": lambda: codebase_store.replace_chunks_for_file(
            "f.php", [{"chunk_id": "f::0", "text": "code", "metadata": {}}]
        ),
        "codebase.delete_chunks_for_file": lambda: codebase_store.delete_chunks_for_file("f.php"),
    }
    for name, write in writes.items():
        before = len(resets)
        write()
        assert len(resets) == before + 1, f"{name} wrote without reloading from disk"


def test_long_lived_reader_sees_writes_from_other_processes(tmp_path):
    """A running MCP server must not keep serving the index it loaded at start."""
    store = tmp_path / "store"
    assert _run("seed", store, "session").returncode == 0
    reader = subprocess.Popen(
        [sys.executable, str(WORKER), "reader", str(store), "session"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert reader.stdout.readline().strip() == "loaded"
    assert _run("writer", store, "session").returncode == 0
    (tmp_path / "go.flag").write_text("go")
    out, err = reader.communicate(timeout=180)
    assert json.loads(out.strip().splitlines()[-1]) == {"found": 20}, err


def test_write_session_stops_the_replaced_system(tmp_path, monkeypatch):
    """Each reload must free the old system, or every write leaks one."""
    import chroma_client
    import config

    monkeypatch.setattr(config, "CHROMA_DB_DIR", tmp_path / "chroma")
    chroma_client.reset()
    old_system = chroma_client.get_client()._system
    with chroma_client.write_session("anything"):
        pass
    assert old_system._running is False


def test_reader_reloads_only_when_generation_changes(tmp_path, monkeypatch):
    import os

    import chroma_client
    import config

    monkeypatch.setattr(config, "CHROMA_DB_DIR", tmp_path / "chroma")
    chroma_client.reset()
    resets: list[int] = []
    real_reset = chroma_client.reset
    monkeypatch.setattr(chroma_client, "reset", lambda: (resets.append(1), real_reset())[1])

    chroma_client.get_client()
    chroma_client.get_client()
    assert resets == []  # no write anywhere: keep the loaded index

    marker = chroma_client._db_dir() / chroma_client._GENERATION_FILE
    marker.write_text("another process wrote", encoding="utf-8")
    # A fixed, clearly different mtime (Windows stores 100 ns ticks, so a
    # tiny value would round to 0, the "no file" generation).
    os.utime(marker, ns=(10**18, 10**18))
    chroma_client.get_client()
    assert resets == [1]
    chroma_client.get_client()
    assert resets == [1]  # reloaded once, then current again


def test_other_writer_bumping_the_marker_does_not_reload_inside_a_session(tmp_path, monkeypatch):
    """Codex review 2, P1: a marker bump from another process mid-session must
    not stop the system under handles the session still uses."""
    import os

    import chroma_client
    import config

    monkeypatch.setattr(config, "CHROMA_DB_DIR", tmp_path / "chroma")
    chroma_client.reset()
    with chroma_client.write_session("codebase") as client:
        col = client.get_or_create_collection("codebase")
        col.add(ids=["a"], documents=["first"], embeddings=[[0.1, 0.2, 0.3]])
        marker = chroma_client._db_dir() / chroma_client._GENERATION_FILE
        marker.write_text("another process wrote", encoding="utf-8")
        os.utime(marker, ns=(10**18, 10**18))
        assert chroma_client.get_client() is client
        with chroma_client.write_session("articles"):  # nested: same client
            pass
        col.add(ids=["b"], documents=["second"], embeddings=[[0.3, 0.2, 0.1]])
        assert col.count() == 2
    assert chroma_client._session_depth == 0
