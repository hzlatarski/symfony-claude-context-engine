"""Hook regressions for raw transcript indexing."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("hook_name", ["session-end", "pre-compact"])
def test_raw_only_transcript_still_spawns_background_index(
    hook_name,
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("CLAUDE_INVOKED_BY", raising=False)
    hook_path = Path(__file__).parents[1] / "hooks" / f"{hook_name}.py"
    spec = importlib.util.spec_from_file_location(
        f"test_{hook_name.replace('-', '_')}",
        hook_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    transcript = tmp_path / "raw.jsonl"
    transcript.write_text('{"type":"tool_result","result":"keep me"}\n', encoding="utf-8")
    archive = tmp_path / "archive.jsonl"
    archive.write_bytes(transcript.read_bytes())
    spawned = []
    extracted_from = []

    monkeypatch.setattr(module, "STATE_DIR", tmp_path)
    monkeypatch.setattr(module, "archive_transcript", lambda *_: archive)
    monkeypatch.setattr(
        module,
        "extract_conversation_context",
        lambda path, **_kwargs: (extracted_from.append(path) or ("", 0, 0)),
    )
    monkeypatch.setattr(module, "load_cursor", lambda _session_id: 0)
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda command, **kwargs: spawned.append((command, kwargs)),
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({
            "session_id": "raw-only",
            "transcript_path": str(transcript),
        })),
    )

    assert module.main() == 0

    assert len(spawned) == 1
    command = spawned[0][0]
    assert command[0] == sys.executable
    assert "uv" not in command
    assert command[-1] == str(archive)
    assert Path(command[-4]).read_text(encoding="utf-8") == ""
    assert extracted_from == [archive]


@pytest.mark.parametrize("hook_name", ["session-end", "pre-compact"])
def test_missing_transcript_is_reported_as_hook_failure(
    hook_name,
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.delenv("CLAUDE_INVOKED_BY", raising=False)
    hook_path = Path(__file__).parents[1] / "hooks" / f"{hook_name}.py"
    spec = importlib.util.spec_from_file_location(
        f"test_missing_{hook_name.replace('-', '_')}",
        hook_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({
            "session_id": "missing",
            "transcript_path": str(tmp_path / "does-not-exist.jsonl"),
        })),
    )

    assert module.main() == 1
    assert "transcript missing" in capsys.readouterr().err


@pytest.mark.parametrize("hook_name", ["session-end", "pre-compact"])
def test_spawn_failure_is_nonzero_and_leaves_durable_retry(
    hook_name,
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("CLAUDE_INVOKED_BY", raising=False)
    hook_path = Path(__file__).parents[1] / "hooks" / f"{hook_name}.py"
    spec = importlib.util.spec_from_file_location(
        f"test_spawn_failure_{hook_name.replace('-', '_')}",
        hook_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    transcript = tmp_path / "raw.jsonl"
    transcript.write_text('{"type":"user","message":"keep me"}\n', encoding="utf-8")
    archive = tmp_path / "archive.jsonl"
    archive.write_bytes(transcript.read_bytes())
    monkeypatch.setattr(module, "STATE_DIR", tmp_path)
    monkeypatch.setattr(module, "archive_transcript", lambda *_: archive)
    monkeypatch.setattr(
        module,
        "extract_conversation_context",
        lambda *_args, **_kwargs: ("USER: keep me", 1, 1),
    )
    monkeypatch.setattr(module, "load_cursor", lambda _session_id: 0)
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("spawn failed")),
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({
            "session_id": "retry-me",
            "transcript_path": str(transcript),
        })),
    )

    assert module.main() == 1
    assert len(list(tmp_path.glob("*.md.pending.json"))) == 1


@pytest.mark.parametrize("hook_name", ["session-end", "pre-compact"])
def test_untrusted_session_id_cannot_escape_state_directory(
    hook_name,
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("CLAUDE_INVOKED_BY", raising=False)
    hook_path = Path(__file__).parents[1] / "hooks" / f"{hook_name}.py"
    spec = importlib.util.spec_from_file_location(
        f"test_session_path_{hook_name.replace('-', '_')}",
        hook_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    transcript = tmp_path / "raw.jsonl"
    transcript.write_text('{"type":"user","message":"keep me"}\n', encoding="utf-8")
    archive = tmp_path / "archive.jsonl"
    archive.write_bytes(transcript.read_bytes())
    commands = []
    monkeypatch.setattr(module, "STATE_DIR", tmp_path)
    monkeypatch.setattr(module, "archive_transcript", lambda *_: archive)
    monkeypatch.setattr(
        module,
        "extract_conversation_context",
        lambda *_args, **_kwargs: ("USER: keep me", 1, 1),
    )
    monkeypatch.setattr(module, "load_cursor", lambda _session_id: 0)
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda command, **_kwargs: commands.append(command),
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({
            "session_id": "../../outside",
            "transcript_path": str(transcript),
        })),
    )

    assert module.main() == 0
    context_file = Path(commands[0][-4])
    assert context_file.parent == tmp_path
    assert ".." not in context_file.name
