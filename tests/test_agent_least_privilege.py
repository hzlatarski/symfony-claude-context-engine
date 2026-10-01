"""The wiki-compiling `claude -p` runs must not bypass permission checks.

ingest.py and compile.py feed the agent untrusted text (transcripts, attached
documents, session logs). They used to pass --dangerously-skip-permissions,
so a prompt-injected instruction could run shell commands or edit any file in
the host project. These tests pin the least-privilege argv at both call sites.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import config  # noqa: E402
import compile as compiler  # noqa: E402
import ingest  # noqa: E402
import utils  # noqa: E402


def _flag_value(cmd: list[str], flag: str) -> str:
    assert flag in cmd, f"{flag} missing from {cmd}"
    return cmd[cmd.index(flag) + 1]


def _assert_least_privilege(cmd: list[str], cwd: str, knowledge: Path) -> None:
    assert "--dangerously-skip-permissions" not in cmd
    assert "--allow-dangerously-skip-permissions" not in cmd
    assert "bypassPermissions" not in cmd

    # The agent runs inside the knowledge dir: the working directory is the
    # only place it may read without approval, and dontAsk denies the rest.
    assert Path(cwd).resolve() == knowledge.resolve()

    tools = _flag_value(cmd, "--tools").split(",")
    assert sorted(tools) == sorted(["Read", "Glob", "Grep", "Write", "Edit"])

    # One Edit rule (covers Write) scoped to the cwd — never a bare tool name
    # that would match everywhere, never a host path interpolated into a glob.
    assert _flag_value(cmd, "--allowedTools") == "Edit(./**)"

    denied = _flag_value(cmd, "--disallowedTools").split(",")
    for tool in ("Bash", "PowerShell", "WebFetch", "WebSearch"):
        assert tool in denied

    assert _flag_value(cmd, "--permission-mode") == "dontAsk"
    assert "--strict-mcp-config" in cmd
    assert _flag_value(cmd, "--mcp-config") == '{"mcpServers":{}}'
    assert _flag_value(cmd, "--setting-sources") == ""

    # Unchanged: turn cap and print mode.
    assert _flag_value(cmd, "--max-turns") == "30"
    assert "-p" in cmd


def _capture_run(monkeypatch, module, on_run=None) -> dict:
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        seen["cwd"] = kwargs.get("cwd")
        seen["env"] = kwargs.get("env")
        if on_run:
            on_run()
        return subprocess.CompletedProcess(args=cmd, returncode=1,
                                           stdout="", stderr="stop")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    return seen


def test_ingest_runs_claude_with_least_privilege(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    concepts = tmp_path / "concepts"
    concepts.mkdir()
    monkeypatch.setattr(ingest, "CONCEPTS_DIR", concepts)
    monkeypatch.setattr(ingest, "CONNECTIONS_DIR", tmp_path / "connections")
    knowledge = tmp_path / "kb-not-yet-created"
    monkeypatch.setattr(ingest, "KNOWLEDGE_DIR", knowledge)
    monkeypatch.setattr(ingest, "update_state", lambda *_a, **_k: None)
    monkeypatch.setattr(ingest.dedup, "similar_to_text", lambda *a, **k: [])
    monkeypatch.setattr(ingest.dedup, "format_preflight_block", lambda *a, **k: "")
    seen = _capture_run(monkeypatch, ingest)

    source = tmp_path / "doc.md"
    source.write_text("# a source document", encoding="utf-8")
    group = utils.SourceGroup(
        id="docs", type="markdown", include=[], exclude=[],
        category="research", description="test",
    )
    state = {"ingested_sources": {}}
    _cost, ok = asyncio.run(ingest.ingest_source_file(group, source, state))

    assert ok is False and state["ingested_sources"] == {}  # exit 1 propagates
    assert knowledge.is_dir(), "cwd must be created, not crash the spawn"
    _assert_least_privilege(seen["cmd"], seen["cwd"], knowledge)
    assert seen["cmd"][seen["cmd"].index("--model") + 1] == config.MODEL_INGEST
    assert "ANTHROPIC_API_KEY" not in seen["env"]


def test_compile_runs_claude_with_least_privilege(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    daily = tmp_path / "daily"
    daily.mkdir()
    log = daily / "2026-07-28.md"
    log.write_text("# source", encoding="utf-8")
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# schema", encoding="utf-8")
    monkeypatch.setattr(compiler, "AGENTS_FILE", agents)
    monkeypatch.setattr(compiler, "CONCEPTS_DIR", tmp_path / "concepts")
    monkeypatch.setattr(compiler, "CONNECTIONS_DIR", tmp_path / "connections")
    monkeypatch.setattr(compiler, "COMPILED_TRUTH_FILE", tmp_path / "truth.md")
    monkeypatch.setattr(compiler, "KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(compiler, "update_state", lambda _m: None)
    monkeypatch.setattr(compiler, "read_wiki_index", lambda *, compact=False: "")
    monkeypatch.setattr(compiler.dedup, "similar_to_text", lambda *_a, **_k: [])
    monkeypatch.setattr(compiler.dedup, "format_preflight_block", lambda *_a, **_k: "")
    monkeypatch.setattr(compiler, "_record_compile_failure", lambda *_a, **_k: None)
    seen = _capture_run(monkeypatch, compiler)

    state = {"ingested_daily": {}}
    ok = asyncio.run(compiler.compile_daily_log(log, state))

    assert ok is False and state["ingested_daily"] == {}  # exit 1 propagates
    _assert_least_privilege(seen["cmd"], seen["cwd"], tmp_path)
    assert seen["cmd"][seen["cmd"].index("--model") + 1] == config.MODEL_COMPILE
    assert "ANTHROPIC_API_KEY" not in seen["env"]


def test_both_call_sites_share_one_flag_set() -> None:
    """No duplicated tool lists: both scripts go through the one helper."""
    for script in ("ingest.py", "compile.py"):
        text = (SCRIPTS / script).read_text(encoding="utf-8")
        assert "compiler_agent_permission_args(" in text
        assert "cwd=compiler_agent_cwd(KNOWLEDGE_DIR)" in text
        assert "--dangerously-skip-permissions" not in text
        assert '"--tools"' not in text


def test_flag_set_is_a_constant_with_no_host_path(monkeypatch, tmp_path) -> None:
    """No host path is interpolated: directory names cannot widen or break it."""
    expected = [
        "--tools", "Read,Glob,Grep,Write,Edit",
        "--allowedTools", "Edit(./**)",
        "--disallowedTools", "Bash,PowerShell,WebFetch,WebSearch",
        "--permission-mode", "dontAsk",
        "--strict-mcp-config",
        "--mcp-config", '{"mcpServers":{}}',
        "--setting-sources", "",
    ]
    assert config.compiler_agent_permission_args() == expected
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", tmp_path / "[ab] dir" / "knowledge")
    assert config.compiler_agent_permission_args() == expected


def test_agent_cwd_is_created(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path / "a")
    kdir = tmp_path / "a" / "knowledge"
    assert config.compiler_agent_cwd(kdir) == str(kdir)
    assert kdir.is_dir()


def test_agent_cwd_refuses_project_root_or_above(tmp_path, monkeypatch) -> None:
    """A knowledge link pointing at the project (or higher) must not become
    the sandbox — Edit(./**) would then cover the whole project."""
    project = tmp_path / "host"
    project.mkdir()
    monkeypatch.setattr(config, "PROJECT_ROOT", project)
    for bad in (project, tmp_path, project / "knowledge" / ".."):
        with pytest.raises(ValueError):
            config.compiler_agent_cwd(bad)


def test_agent_cwd_refuses_link_to_project_root(tmp_path, monkeypatch) -> None:
    project = tmp_path / "host"
    project.mkdir()
    monkeypatch.setattr(config, "PROJECT_ROOT", project)
    link = project / "knowledge"
    try:
        if sys.platform == "win32":
            subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(project)],
                           check=True, capture_output=True)
        else:
            link.symlink_to(project, target_is_directory=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"cannot create a directory link here: {exc}")
    with pytest.raises(ValueError):
        config.compiler_agent_cwd(link)


def test_agent_cwd_refusal_is_a_normal_failure(tmp_path, monkeypatch) -> None:
    """ingest: the refusal surfaces as a failed file, nothing recorded."""
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(ingest, "KNOWLEDGE_DIR", tmp_path)  # == project root
    monkeypatch.setattr(ingest, "CONCEPTS_DIR", tmp_path / "concepts")
    monkeypatch.setattr(ingest, "CONNECTIONS_DIR", tmp_path / "connections")
    monkeypatch.setattr(ingest.dedup, "similar_to_text", lambda *a, **k: [])
    monkeypatch.setattr(ingest.dedup, "format_preflight_block", lambda *a, **k: "")
    calls = []
    monkeypatch.setattr(ingest.subprocess, "run", lambda *a, **k: calls.append(a))
    source = tmp_path / "doc.md"
    source.write_text("# doc", encoding="utf-8")
    group = utils.SourceGroup(id="docs", type="markdown", include=[], exclude=[],
                              category="research", description="test")
    state = {"ingested_sources": {}}
    _cost, ok = asyncio.run(ingest.ingest_source_file(group, source, state))
    assert ok is False and state["ingested_sources"] == {}
    assert calls == [], "claude must never be spawned in the project root"


# ── prompt paths: the agent's cwd is knowledge/, so it must be told paths
#    relative to it. `knowledge/concepts/x.md` would land in knowledge/knowledge/.

def _prompt_for(module_name: str, tmp_path: Path, monkeypatch) -> tuple[str, Path]:
    kdir = tmp_path / "knowledge"
    (kdir / "concepts").mkdir(parents=True)
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# schema stub", encoding="utf-8")
    module = ingest if module_name == "ingest" else compiler
    monkeypatch.setattr(module, "AGENTS_FILE", agents)
    monkeypatch.setattr(module, "KNOWLEDGE_DIR", kdir)
    monkeypatch.setattr(module, "CONCEPTS_DIR", kdir / "concepts")
    monkeypatch.setattr(module, "CONNECTIONS_DIR", kdir / "connections")
    monkeypatch.setattr(module, "COMPILED_TRUTH_FILE", kdir / "compiled-truth.md")
    monkeypatch.setattr(module, "update_state", lambda *_a, **_k: None)
    monkeypatch.setattr(module.dedup, "similar_to_text", lambda *a, **k: [])
    monkeypatch.setattr(module.dedup, "format_preflight_block", lambda *a, **k: "")
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["prompt"] = kwargs["input"]
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="x")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    if module_name == "ingest":
        source = tmp_path / "doc.md"
        source.write_text("# a source", encoding="utf-8")
        group = utils.SourceGroup(id="docs", type="markdown", include=[], exclude=[],
                                  category="research", description="test")
        asyncio.run(ingest.ingest_source_file(group, source, {"ingested_sources": {}}))
    else:
        monkeypatch.setattr(compiler, "read_wiki_index", lambda *, compact=False: "")
        monkeypatch.setattr(compiler, "_record_compile_failure", lambda *_a, **_k: None)
        daily = kdir / "daily"
        daily.mkdir()
        log = daily / "2026-07-28.md"
        log.write_text("# log", encoding="utf-8")
        asyncio.run(compiler.compile_daily_log(log, {"ingested_daily": {}}))
    return seen["prompt"], kdir


import re  # noqa: E402


@pytest.mark.parametrize("module_name", ["ingest", "compile"])
def test_prompt_paths_are_relative_to_the_knowledge_cwd(
    module_name, tmp_path, monkeypatch,
) -> None:
    prompt, kdir = _prompt_for(module_name, tmp_path, monkeypatch)

    # The cwd is stated explicitly.
    assert config.COMPILER_AGENT_CWD_NOTE in prompt
    assert "Your working directory is the knowledge base root" in prompt

    # Outside that note (which maps the schema's `knowledge/` naming), no
    # path the agent is told may start with knowledge/ ...
    rest = prompt.replace(config.COMPILER_AGENT_CWD_NOTE, "")
    assert not re.search(r"(?<![\w-])knowledge[/\\]", rest), (
        "agent prompt names a knowledge/ path; with cwd = knowledge/ it would "
        "write to knowledge/knowledge/"
    )
    # ... and no absolute path into the knowledge dir is handed out either.
    assert str(kdir) not in rest and kdir.as_posix() not in rest

    # The write targets are given relative to the cwd.
    for target in ("concepts/", "connections/", "index.md", "log.md"):
        assert f": {target}" in rest, f"File paths section lacks {target}"


def test_agent_path_is_relative_to_knowledge_dir(tmp_path) -> None:
    kdir = tmp_path / "knowledge"
    assert config.agent_path(kdir / "concepts", kdir, is_dir=True) == "concepts/"
    assert config.agent_path(kdir / "index.md", kdir) == "index.md"
    outside = tmp_path / "elsewhere" / "x.md"
    assert config.agent_path(outside, kdir) == str(outside)
