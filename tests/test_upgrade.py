"""Upgrade must fail closed when required setup cannot complete."""

from __future__ import annotations

import sys


def _prepare_main(tmp_path, monkeypatch):
    import upgrade

    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").mkdir()
    version = root / "VERSION"
    version.write_text("1.0.0", encoding="utf-8")
    state = tmp_path / "state"
    monkeypatch.setattr(upgrade, "_ROOT", root)
    monkeypatch.setattr(upgrade, "VERSION_FILE", version)
    monkeypatch.setattr(upgrade, "_state_dir", lambda: state)
    monkeypatch.setattr(upgrade, "_ensure_clean_or_stash", lambda: None)
    monkeypatch.setattr(upgrade, "_run", lambda *_a, **_k: (0, "", ""))
    monkeypatch.setattr(
        sys,
        "argv",
        ["upgrade.py", "--remote", "origin", "--branch", "main"],
    )
    return upgrade, state


def test_dependency_failure_returns_nonzero_without_success_marker(
    tmp_path, monkeypatch,
) -> None:
    upgrade, state = _prepare_main(tmp_path, monkeypatch)

    class Failed:
        returncode = 9

    monkeypatch.setattr(upgrade.subprocess, "run", lambda *_a, **_k: Failed())
    monkeypatch.setattr(upgrade, "_run_migrations", lambda *_a: None)

    assert upgrade.main() != 0
    assert not (state / "just-upgraded-from").exists()


def test_migration_failure_returns_nonzero_without_success_marker(
    tmp_path, monkeypatch,
) -> None:
    upgrade, state = _prepare_main(tmp_path, monkeypatch)

    class Success:
        returncode = 0

    monkeypatch.setattr(upgrade.subprocess, "run", lambda *_a, **_k: Success())
    monkeypatch.setattr(
        upgrade,
        "_run_migrations",
        lambda *_a: (_ for _ in ()).throw(RuntimeError("migration failed")),
    )

    assert upgrade.main() != 0
    assert not (state / "just-upgraded-from").exists()
