"""Regression tests for daily-log compilation safety."""

from __future__ import annotations

import asyncio
import argparse
import subprocess
import sys
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import compile as compiler  # noqa: E402


def _drive_compile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    returncode: int,
    stdout: str,
    write_article: bool,
) -> tuple[dict, object, str]:
    daily = tmp_path / "daily"
    concepts = tmp_path / "concepts"
    connections = tmp_path / "connections"
    daily.mkdir()
    concepts.mkdir()
    connections.mkdir()
    log = daily / "2026-07-28.md"
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
        compiler, "read_wiki_index",
        lambda *, compact=False: "compact-index" if compact else "full-index",
    )
    monkeypatch.setattr(compiler.dedup, "similar_to_text", lambda *_a, **_k: [])
    monkeypatch.setattr(compiler.dedup, "format_preflight_block", lambda *_a, **_k: "")

    prompt = ""

    def fake_run(*_args, **kwargs):
        nonlocal prompt
        prompt = kwargs["input"]
        if write_article:
            (concepts / "written.md").write_text(
                "fact [src:daily/2026-07-28.md]", encoding="utf-8",
            )
        return subprocess.CompletedProcess(
            args=["claude"], returncode=returncode, stdout=stdout, stderr="",
        )

    monkeypatch.setattr(compiler.subprocess, "run", fake_run)
    state: dict = {"ingested_daily": {}}
    result = asyncio.run(compiler.compile_daily_log(log, state))
    return state, result, prompt


def test_nonzero_exit_with_stdout_is_not_recorded(tmp_path, monkeypatch) -> None:
    state, result, _ = _drive_compile(
        tmp_path,
        monkeypatch,
        returncode=1,
        stdout="Prompt is too long",
        write_article=False,
    )
    assert state["ingested_daily"] == {}
    assert result is False


def test_zero_exit_without_knowledge_mutation_is_not_recorded(
    tmp_path, monkeypatch,
) -> None:
    state, result, _ = _drive_compile(
        tmp_path,
        monkeypatch,
        returncode=0,
        stdout="done",
        write_article=False,
    )
    assert state["ingested_daily"] == {}
    assert result is False


def test_zero_exit_with_source_anchored_mutation_is_recorded(
    tmp_path, monkeypatch,
) -> None:
    state, result, _ = _drive_compile(
        tmp_path,
        monkeypatch,
        returncode=0,
        stdout="done",
        write_article=True,
    )
    assert list(state["ingested_daily"]) == ["2026-07-28.md"]
    assert result is True


def test_compile_prompt_uses_compact_index(tmp_path, monkeypatch) -> None:
    _, _, prompt = _drive_compile(
        tmp_path,
        monkeypatch,
        returncode=1,
        stdout="failed",
        write_article=False,
    )
    assert "compact-index" in prompt
    assert "full-index" not in prompt


def test_compile_logs_is_oldest_first_and_serial(monkeypatch) -> None:
    active = 0
    maximum_active = 0
    order: list[str] = []

    async def fake_compile(path: Path, state: dict):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        order.append(path.name)
        await asyncio.sleep(0)
        active -= 1
        return 0.0

    monkeypatch.setattr(compiler, "compile_daily_log", fake_compile)
    paths = [Path("2026-07-28.md"), Path("2026-07-26.md"), Path("2026-07-27.md")]

    asyncio.run(compiler.compile_logs(paths, {}))

    assert order == ["2026-07-26.md", "2026-07-27.md", "2026-07-28.md"]
    assert maximum_active == 1


def test_command_returns_nonzero_when_any_daily_compile_fails(
    tmp_path, monkeypatch,
) -> None:
    log = tmp_path / "2026-07-28.md"
    log.write_text("# daily", encoding="utf-8")
    monkeypatch.setattr(
        compiler, "update_state", lambda _mutator: {"ingested_daily": {}},
    )

    async def failed_compile(_logs, _state):
        return [False]

    monkeypatch.setattr(compiler, "compile_logs", failed_compile)
    monkeypatch.setattr(compiler, "list_wiki_articles", lambda: [])
    monkeypatch.setattr(compiler, "regenerate_truth", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "reindex",
        types.SimpleNamespace(reindex_articles=lambda force=False: (0, 0)),
    )

    result = compiler._main_unlocked(
        argparse.Namespace(all=False, file=str(log), dry_run=False),
    )

    assert result == 1
