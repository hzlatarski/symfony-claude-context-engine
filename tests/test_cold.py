"""Tests for cold storage — the reversible, non-destructive article archive.

Covers mark/restore round-trip and idempotence, provenance records, the slug
set vs full-record views, ``.md`` stripping, the loss-bearing load contract
(missing = empty, corrupt = raise), and best-effort audit logging.
"""
import json

import pytest

import cold


NOW = "2026-10-21T10:00:00+00:00"


# ── mark / restore round-trip ────────────────────────────────────────

def test_mark_cold_creates_store_with_provenance(tmp_path):
    store = tmp_path / "cold.json"
    assert cold.mark_cold(
        "concepts/dup", reason="near-duplicate of concepts/keep",
        by="maintain", now=NOW, path=store, audit=False,
    ) is True
    rec = cold.load_cold_records(store)["concepts/dup"]
    assert rec == {"reason": "near-duplicate of concepts/keep", "at": NOW, "by": "maintain"}
    assert cold.load_cold(store) == {"concepts/dup"}
    assert cold.is_cold("concepts/dup", path=store)


def test_mark_cold_is_idempotent_and_keeps_first_reason(tmp_path):
    store = tmp_path / "cold.json"
    cold.mark_cold("concepts/dup", reason="first", now=NOW, path=store, audit=False)
    assert cold.mark_cold("concepts/dup", reason="second", now=NOW, path=store, audit=False) is False
    assert cold.load_cold_records(store)["concepts/dup"]["reason"] == "first"


def test_restore_removes_and_is_reversible(tmp_path):
    store = tmp_path / "cold.json"
    cold.mark_cold("concepts/dup", now=NOW, path=store, audit=False)
    assert cold.restore("concepts/dup", path=store, audit=False) is True
    assert cold.load_cold(store) == set()
    assert cold.is_cold("concepts/dup", path=store) is False


def test_restore_absent_slug_returns_false(tmp_path):
    store = tmp_path / "cold.json"
    assert cold.restore("concepts/never", path=store, audit=False) is False


def test_mark_and_restore_strip_md_suffix(tmp_path):
    store = tmp_path / "cold.json"
    cold.mark_cold("concepts/dup.md", now=NOW, path=store, audit=False)
    assert "concepts/dup" in cold.load_cold(store)
    assert cold.restore("concepts/dup.md", path=store, audit=False) is True


# ── load contract ────────────────────────────────────────────────────

def test_load_missing_file_is_empty(tmp_path):
    assert cold.load_cold(tmp_path / "nope.json") == set()
    assert cold.load_cold_records(tmp_path / "nope.json") == {}


def test_load_corrupt_file_raises(tmp_path):
    store = tmp_path / "cold.json"
    store.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError):
        cold.load_cold(store)


def test_load_wrong_schema_raises(tmp_path):
    store = tmp_path / "cold.json"
    store.write_text(json.dumps({"cold": ["concepts/dup"]}), encoding="utf-8")  # list, not object
    with pytest.raises(RuntimeError):
        cold.load_cold_records(store)


def test_records_sorted_on_disk(tmp_path):
    store = tmp_path / "cold.json"
    cold.mark_cold("concepts/zeta", now=NOW, path=store, audit=False)
    cold.mark_cold("concepts/alpha", now=NOW, path=store, audit=False)
    on_disk = json.loads(store.read_text(encoding="utf-8"))
    assert list(on_disk["cold"].keys()) == ["concepts/alpha", "concepts/zeta"]


# ── audit logging (best-effort) ──────────────────────────────────────

def test_audit_appends_a_line_to_log(tmp_path, monkeypatch):
    store = tmp_path / "cold.json"
    log = tmp_path / "log.md"
    monkeypatch.setattr(cold.config, "LOG_FILE", log, raising=False)
    cold.mark_cold("concepts/dup", reason="dup of keep", now=NOW, path=store, audit=True)
    cold.restore("concepts/dup", path=store, audit=True)
    text = log.read_text(encoding="utf-8")
    assert "cold | mark concepts/dup" in text
    assert "cold | restore concepts/dup" in text


def test_audit_failure_does_not_abort_mark(tmp_path, monkeypatch):
    store = tmp_path / "cold.json"
    # Point the log at a directory path so opening it for append raises.
    monkeypatch.setattr(cold.config, "LOG_FILE", tmp_path, raising=False)
    assert cold.mark_cold("concepts/dup", now=NOW, path=store, audit=True) is True
    assert cold.is_cold("concepts/dup", path=store)
