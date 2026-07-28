"""Tests for safe migration to the replaceable active Chroma directory."""

from __future__ import annotations

from pathlib import Path

import pytest


def _resolver():
    try:
        from chroma_path import ensure_active_chroma_dir
    except ModuleNotFoundError:
        pytest.fail("chroma_path.ensure_active_chroma_dir is not implemented")
    return ensure_active_chroma_dir


def test_legacy_store_moves_into_active(tmp_path: Path) -> None:
    root = tmp_path / "chroma"
    legacy_segment = root / "12345678-1234-1234-1234-123456789abc"
    legacy_segment.mkdir(parents=True)
    (root / "chroma.sqlite3").write_text("db", encoding="utf-8")
    (legacy_segment / "index.bin").write_text("index", encoding="utf-8")

    active = _resolver()(root)

    assert active == root / "active"
    assert (active / "chroma.sqlite3").read_text(encoding="utf-8") == "db"
    assert (active / legacy_segment.name / "index.bin").exists()
    assert not (root / "chroma.sqlite3").exists()


def test_active_store_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "chroma"
    active = root / "active"
    active.mkdir(parents=True)
    marker = active / "chroma.sqlite3"
    marker.write_text("active", encoding="utf-8")

    assert _resolver()(root) == active
    assert marker.read_text(encoding="utf-8") == "active"


def test_mixed_active_and_legacy_store_fails_loudly(tmp_path: Path) -> None:
    root = tmp_path / "chroma"
    (root / "active").mkdir(parents=True)
    (root / "chroma.sqlite3").write_text("legacy", encoding="utf-8")

    with pytest.raises(RuntimeError, match="both active and legacy"):
        _resolver()(root)


def test_empty_root_creates_active(tmp_path: Path) -> None:
    root = tmp_path / "chroma"

    active = _resolver()(root)

    assert active.is_dir()


def test_interrupted_migration_resumes_when_marker_is_present(tmp_path: Path) -> None:
    root = tmp_path / "chroma"
    active = root / "active"
    moved_segment = active / "12345678-1234-1234-1234-123456789abc"
    remaining_segment = root / "87654321-4321-4321-4321-cba987654321"
    moved_segment.mkdir(parents=True)
    remaining_segment.mkdir(parents=True)
    (root / ".active-migration-in-progress").write_text(
        "legacy-to-active\n", encoding="utf-8",
    )

    assert _resolver()(root) == active
    assert moved_segment.is_dir()
    assert (active / remaining_segment.name).is_dir()
    assert not (root / ".active-migration-in-progress").exists()
