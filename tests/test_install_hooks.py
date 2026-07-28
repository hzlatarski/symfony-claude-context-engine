"""Tests for install.py hook generation and repair.

Regression cover for the 2026-07-26 incident: the installer emitted hook
commands calling a bare ``uv``, which Git Bash on Windows cannot resolve. Hooks
failed silently and three projects stopped capturing knowledge — two for about
a month, one had never captured anything.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def inst() -> types.ModuleType:
    """Load install.py without executing main()."""
    src = (_ROOT / "install.py").read_text(encoding="utf-8")
    mod = types.ModuleType("install_under_test")
    mod.__file__ = str(_ROOT / "install.py")
    saved = sys.argv
    sys.argv = ["install.py"]
    try:
        exec(compile(src, str(_ROOT / "install.py"), "exec"), mod.__dict__)
    finally:
        sys.argv = saved
    return mod


BARE = "cd .claude/memory-compiler && uv run python hooks/session-end.py"


# ── command construction ─────────────────────────────────────────────────────

def test_generated_hooks_do_not_depend_on_uv_path(inst, monkeypatch) -> None:
    monkeypatch.setattr(inst, "_find_uv", lambda: "C:\\tools\\uv\\uv.exe")
    for entries in inst._build_hooks().values():
        command = entries[0]["hooks"][0]["command"]
        assert ".venv/Scripts/python.exe" in command
        assert "uv run" not in command
        assert not inst._is_broken_uv_command(command), command


def test_generated_hooks_execute_project_virtualenv_python(inst) -> None:
    for entries in inst._build_hooks().values():
        command = entries[0]["hooks"][0]["command"]
        assert "uv run" not in command
        assert ".venv/" in command
        assert "python" in command.lower()


def test_prefix_never_degrades_to_bare_uv_when_uv_is_absent(inst, monkeypatch) -> None:
    monkeypatch.setattr(inst, "_find_uv", lambda: None)
    prefix = inst._uv_hook_prefix()
    assert "uv run" not in prefix
    assert ".venv/Scripts/python.exe" in prefix


def test_prefix_unsets_virtualenv(inst) -> None:
    # An activated venv makes uv resolve the wrong environment.
    assert "unset VIRTUAL_ENV" in inst._uv_hook_prefix()


def test_build_hooks_covers_every_spec(inst) -> None:
    built = inst._build_hooks()
    assert set(built) == {event for event, _s, _t in inst._HOOK_SPECS}
    for event, script, timeout in inst._HOOK_SPECS:
        entry = built[event][0]["hooks"][0]
        assert entry["command"].endswith(script)
        assert entry["timeout"] == timeout


@pytest.mark.parametrize("path_str, expected", [
    ("C:\\Users\\me\\Scripts", "/c/Users/me/Scripts"),
    ("D:\\tools", "/d/tools"),
])
def test_windows_paths_convert_for_git_bash(inst, path_str, expected) -> None:
    assert inst._to_bash_path(Path(path_str)) == expected


# ── ownership / brokenness predicates ────────────────────────────────────────

def test_bare_uv_is_broken_but_prefixed_is_not(inst) -> None:
    assert inst._is_broken_uv_command(BARE)
    assert not inst._is_broken_uv_command(
        'cd .claude/memory-compiler && PATH="$PATH:/c/uv" uv run python hooks/session-end.py'
    )


@pytest.mark.parametrize("command", [
    # An incidental "PATH=" substring must not mask a genuinely dead hook —
    # otherwise it is never repaired and the project silently stops capturing.
    "cd .claude/memory-compiler && MYPATH=x && uv run python hooks/session-end.py",
    "cd .claude/memory-compiler && echo PATH=missing && uv run python hooks/session-end.py",
])
def test_incidental_path_substring_still_counts_as_broken(inst, command) -> None:
    assert inst._is_broken_uv_command(command)


def test_user_hook_sharing_our_filename_is_not_ours(inst) -> None:
    """The suffix-collision hijack: must not claim someone else's hook."""
    assert not inst._is_our_hook(
        "python .claude/my/hooks/session-end.py", "hooks/session-end.py"
    )
    assert inst._is_our_hook(BARE, "hooks/session-end.py")


@pytest.mark.parametrize("command", [
    # Recognising these matters: an unrecognised hook of ours gets a second
    # copy appended, so the event fires twice.
    "cd .claude\\memory-compiler && uv run python hooks\\session-end.py",
    "cd .claude/memory-compiler && uv run python hooks/session-end.py 2>>hook.err",
    "cd C:/project/.claude/memory-compiler && uv run python hooks/session-end.py",
])
def test_our_hook_recognised_across_spellings(inst, command) -> None:
    assert inst._is_our_hook(command, "hooks/session-end.py")


@pytest.mark.parametrize("entry", [
    {"hooks": []}, {"hooks": "nonsense"}, {}, "not-a-dict", {"hooks": [{}]},
])
def test_malformed_entries_do_not_raise(inst, entry) -> None:
    assert inst._hook_commands(entry) == []
    assert inst._hook_command(entry) == ""


def test_grouped_entry_finds_our_hook_beyond_the_first(inst) -> None:
    entry = {"hooks": [
        {"type": "command", "command": "echo unrelated"},
        {"type": "command", "command": BARE},
    ]}
    assert BARE in inst._hook_commands(entry)


# ── merge behaviour ──────────────────────────────────────────────────────────

def _settings(tmp_path: Path) -> Path:
    (tmp_path / ".claude").mkdir(parents=True, exist_ok=True)
    return tmp_path / ".claude" / "settings.json"


def _run_merge(inst, tmp_path: Path) -> dict:
    inst.CLAUDE_DIR = tmp_path / ".claude"
    inst.PROJECT_ROOT = tmp_path
    inst.merge_settings_json()
    return json.loads(_settings(tmp_path).read_text(encoding="utf-8"))


def test_fresh_install_writes_all_hooks(inst, tmp_path: Path) -> None:
    _settings(tmp_path)
    data = _run_merge(inst, tmp_path)
    assert sum(len(v) for v in data["hooks"].values()) == len(inst._HOOK_SPECS)


def test_stale_bare_uv_hooks_are_repaired_without_duplicating(inst, tmp_path) -> None:
    stale = {"hooks": {
        event: [{"matcher": "", "hooks": [{
            "type": "command",
            "command": f"cd .claude/memory-compiler && uv run python {script}",
            "timeout": timeout,
        }]}]
        for event, script, timeout in inst._HOOK_SPECS
    }}
    _settings(tmp_path).write_text(json.dumps(stale), encoding="utf-8")

    data = _run_merge(inst, tmp_path)

    commands = [b["hooks"][0]["command"] for v in data["hooks"].values() for b in v]
    assert len(commands) == len(inst._HOOK_SPECS), "must repair, not duplicate"
    assert not any(inst._is_broken_uv_command(c) for c in commands)


def test_merge_is_idempotent(inst, tmp_path: Path) -> None:
    _settings(tmp_path)
    first = _run_merge(inst, tmp_path)
    assert _run_merge(inst, tmp_path) == first


def test_deliberate_customisation_is_preserved(inst, tmp_path: Path) -> None:
    """A working custom command must survive a re-run."""
    custom = ('cd .claude/memory-compiler && FOO=1 && PATH="$PATH:/c/uv" '
              'uv run python hooks/session-end.py')
    existing = {"hooks": {"SessionEnd": [
        {"matcher": "", "hooks": [{"type": "command", "command": custom, "timeout": 99}]}
    ]}}
    _settings(tmp_path).write_text(json.dumps(existing), encoding="utf-8")

    data = _run_merge(inst, tmp_path)

    session_end = [b["hooks"][0] for b in data["hooks"]["SessionEnd"]]
    assert any(h["command"] == custom and h["timeout"] == 99 for h in session_end)


def test_unrelated_user_hook_is_left_alone(inst, tmp_path: Path) -> None:
    """A user hook whose filename collides must not be rewritten."""
    theirs = "python .claude/my/hooks/session-end.py"
    existing = {"hooks": {"SessionEnd": [
        {"matcher": "", "hooks": [{"type": "command", "command": theirs, "timeout": 7}]}
    ]}}
    _settings(tmp_path).write_text(json.dumps(existing), encoding="utf-8")

    data = _run_merge(inst, tmp_path)

    commands = [b["hooks"][0]["command"] for b in data["hooks"]["SessionEnd"]]
    assert theirs in commands, "user's own hook must survive untouched"
    assert len(commands) == 2, "ours should be added alongside theirs"


def test_malformed_existing_entry_does_not_abort_install(inst, tmp_path: Path) -> None:
    existing = {"hooks": {"SessionEnd": [{"matcher": "", "hooks": []}]}}
    _settings(tmp_path).write_text(json.dumps(existing), encoding="utf-8")

    data = _run_merge(inst, tmp_path)  # must not raise IndexError

    commands = [inst._hook_command(b) for b in data["hooks"]["SessionEnd"]]
    assert any(c.endswith("hooks/session-end.py") for c in commands)


def test_backslash_spelled_hook_is_repaired_not_duplicated(inst, tmp_path) -> None:
    """A Windows-spelled command must be recognised as ours, else it doubles."""
    existing = {"hooks": {"SessionEnd": [{"matcher": "", "hooks": [{
        "type": "command",
        "command": "cd .claude\\memory-compiler && uv run python hooks\\session-end.py",
        "timeout": 10,
    }]}]}}
    _settings(tmp_path).write_text(json.dumps(existing), encoding="utf-8")

    data = _run_merge(inst, tmp_path)

    commands = [b["hooks"][0]["command"] for b in data["hooks"]["SessionEnd"]]
    assert len(commands) == 1, "must repair in place, not append a second copy"
    assert not inst._is_broken_uv_command(commands[0])


def test_every_shipped_hook_file_is_registered(inst) -> None:
    """A hook that ships but is never wired is dead weight.

    hooks/user-prompt-submit.py shipped in every install yet went unregistered,
    so 4 of 7 projects silently lacked auto-injection.
    """
    shipped = {p.name for p in (_ROOT / "hooks").glob("*.py")
               if not p.name.startswith("_")}
    registered = {Path(script).name for _e, script, _t in inst._HOOK_SPECS}
    assert shipped == registered, f"unregistered hook files: {shipped - registered}"
