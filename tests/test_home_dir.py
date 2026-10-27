"""Installed compiler state must survive replacing its entire code directory."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def probe(code, home=None):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "scripts"))
    env.pop("MEMORY_COMPILER_HOME", None)
    env.pop("CLAUDE_INVOKED_BY", None)
    if home is not None:
        env["MEMORY_COMPILER_HOME"] = str(home)
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env,
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_home_moves_everything(tmp_path):
    home = tmp_path / "José Müller" / "wiki"
    paths = probe("""
import config, json
names = ('HOME_DIR', 'STATE_DIR', 'STATE_FILE', 'KNOWLEDGE_DIR', 'REPORTS_DIR',
         'SOURCES_FILE', 'ENV_FILE', 'FLUSH_STATE_FILE', 'FLUSH_CURSOR_FILE',
         'FLUSH_LOG_FILE', 'COMPILE_LOG_FILE', 'COMPILE_LOCK_FILE', 'WIP_FILE', 'PROJECT_ROOT')
print(json.dumps({name: str(getattr(config, name, 'missing')) for name in names}))
""", home)
    assert paths == {
        "HOME_DIR": str(home), "STATE_DIR": str(home / "state"),
        "STATE_FILE": str(home / "state/state.json"),
        "KNOWLEDGE_DIR": str(home / "knowledge"), "REPORTS_DIR": str(home / "reports"),
        "SOURCES_FILE": str(home / "sources.yaml"), "ENV_FILE": str(home / ".env.local"),
        "FLUSH_STATE_FILE": str(home / "state/last-flush.json"),
        "FLUSH_CURSOR_FILE": str(home / "state/flush-cursors.json"),
        "FLUSH_LOG_FILE": str(home / "state/flush.log"),
        "COMPILE_LOG_FILE": str(home / "state/compile.log"),
        "COMPILE_LOCK_FILE": str(home / "state/compile.lock"),
        "WIP_FILE": str(home / "wip.md"),
        "PROJECT_ROOT": str(home),
    }


def test_unset_is_unchanged():
    assert probe("""
import config as c, json
print(json.dumps({
    'state': c.STATE_FILE == c.SCRIPTS_DIR / 'state.json',
    'knowledge': c.KNOWLEDGE_DIR == c.PROJECT_ROOT / 'knowledge',
    'sources': c.SOURCES_FILE == c.ROOT_DIR / 'sources.yaml',
    'reports': c.REPORTS_DIR == c.ROOT_DIR / 'reports',
    'home': getattr(c, 'HOME_DIR', None),
}))
""") == {"state": True, "knowledge": True, "sources": True, "reports": True, "home": None}


def test_writers_share_home_paths_and_create_missing_state_directory(tmp_path):
    home = tmp_path / "José Müller" / "wiki"
    result = probe("""
import config as c, json
import flush, flush_cursor, cost_report
from utils import update_state
update_state(lambda state: state.update(test_marker=7))
flush.update_flush_state(lambda state: state.update(test_marker=8))
flush_cursor.save_cursor('home-test', 3)
print(json.dumps({
    'compile_state': json.loads(c.STATE_FILE.read_text())['test_marker'],
    'flush_state': json.loads(c.FLUSH_STATE_FILE.read_text())['test_marker'],
    'cursor': flush_cursor.load_cursor('home-test'),
    'shared_state': cost_report.STATE_FILE == c.STATE_FILE,
    'shared_flush': cost_report.FLUSH_STATE_FILE == flush.STATE_FILE == c.FLUSH_STATE_FILE,
}))
""", home)
    assert result == {"compile_state": 7, "flush_state": 8, "cursor": 3,
                      "shared_state": True, "shared_flush": True}
    assert (home / "state/flush.log").exists()


@pytest.mark.parametrize("target", [".", ".."])
def test_agent_refuses_home_or_ancestor(tmp_path, target):
    home = tmp_path / "José Müller" / "wiki"
    assert probe("""
import config as c, json
try:
    c.compiler_agent_cwd(c.HOME_DIR / %s)
except ValueError:
    print(json.dumps(True))
else:
    print(json.dumps(False))
""" % repr(target), home) is True


def test_agent_accepts_home_knowledge(tmp_path):
    home = tmp_path / "José Müller" / "wiki"
    assert probe("""
import config as c, json
print(json.dumps(c.compiler_agent_cwd(c.KNOWLEDGE_DIR)))
""", home) == str(home / "knowledge")


@pytest.mark.parametrize("hook", ["session-start", "session-end", "pre-compact", "post-tool-use"])
def test_hooks_use_the_same_home(tmp_path, hook):
    home = tmp_path / "José Müller" / "wiki"
    assert probe("""
import json, runpy, config as c
hook = runpy.run_path('hooks/%s.py')
paths = ('STATE_FILE', 'FLUSH_STATE_FILE', 'STATE_DIR', 'WIP_FILE', 'DAILY_DIR', 'FLUSH_LOG_FILE')
print(json.dumps({name: str(hook[name]) for name in paths if name in hook}))
""" % hook, home) == {
        "session-start": {"STATE_FILE": str(home / "state/state.json"),
                          "FLUSH_STATE_FILE": str(home / "state/last-flush.json"),
                          "WIP_FILE": str(home / "wip.md"),
                          "DAILY_DIR": str(home / "knowledge/daily")},
        "session-end": {"STATE_DIR": str(home / "state"),
                        "FLUSH_LOG_FILE": str(home / "state/flush.log")},
        "pre-compact": {"STATE_DIR": str(home / "state"),
                        "FLUSH_LOG_FILE": str(home / "state/flush.log")},
        "post-tool-use": {"DAILY_DIR": str(home / "knowledge/daily"),
                          "FLUSH_LOG_FILE": str(home / "state/flush.log")},
    }[hook]


def test_home_dotenv_is_loaded_without_overriding_environment(tmp_path, monkeypatch):
    home = tmp_path / "José Müller" / "wiki"
    home.mkdir(parents=True)
    (home / ".env.local").write_text(
        "INSTALLER_HOME_MARKER=from-file\nINSTALLER_ENV_MARKER=from-file\n", encoding="utf-8")
    monkeypatch.setenv("INSTALLER_ENV_MARKER", "from-environment")
    assert probe("""
import config, os, json
print(json.dumps([os.environ.get('INSTALLER_HOME_MARKER'), os.environ.get('INSTALLER_ENV_MARKER')]))
""", home) == ["from-file", "from-environment"]
