import json
from pathlib import Path
from typing import Literal

import pytest

from whisper_tray.history import History, HistoryEntry


def test_history_entry_fields():
    e = HistoryEntry(transcript="hello", enhanced_prompt="Hello world", mode="rewrite")
    assert e.transcript == "hello"
    assert e.enhanced_prompt == "Hello world"
    assert e.mode == "rewrite"


def test_history_append_and_last(tmp_path):
    h = History(persist_path=tmp_path / "history.json")
    h.append(HistoryEntry("t1", "p1", "verbatim"))
    assert h.last() is not None
    assert h.last().transcript == "t1"


def test_history_ring_buffer_maxlen(tmp_path):
    h = History(persist_path=tmp_path / "history.json", maxlen=3)
    for i in range(5):
        h.append(HistoryEntry(f"t{i}", f"p{i}", "verbatim"))
    assert len(h.entries()) == 3
    assert h.entries()[0].transcript == "t2"


def test_history_last_returns_none_when_empty(tmp_path):
    h = History(persist_path=tmp_path / "history.json")
    assert h.last() is None


def test_history_persists_to_json(tmp_path):
    path = tmp_path / "history.json"
    h = History(persist_path=path)
    h.append(HistoryEntry("t1", "p1", "rewrite"))
    data = json.loads(path.read_text())
    assert data[0]["transcript"] == "t1"
    assert data[0]["enhanced_prompt"] == "p1"
    assert data[0]["mode"] == "rewrite"


def test_history_loads_from_existing_json(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps([
        {"transcript": "saved", "enhanced_prompt": "saved prompt", "mode": "clean"}
    ]))
    h = History(persist_path=path)
    assert h.last().transcript == "saved"


def test_history_entries_returns_oldest_first(tmp_path):
    h = History(persist_path=tmp_path / "history.json")
    h.append(HistoryEntry("first", "p1", "verbatim"))
    h.append(HistoryEntry("second", "p2", "verbatim"))
    entries = h.entries()
    assert entries[0].transcript == "first"
    assert entries[1].transcript == "second"
