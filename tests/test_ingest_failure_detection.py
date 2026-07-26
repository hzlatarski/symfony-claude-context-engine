"""Tests for ingest failure detection and prompt sizing.

Regression cover for 2026-07-26: the ingest prompt outgrew the model's context
window (the wiki index alone reached 338 KB / ~86k tokens at 651 articles), the
CLI returned exit 1 with "Prompt is too long" on *stdout*, and the guard
``returncode != 0 and not stdout.strip()`` let it through to the success path.
Hashes were recorded, "Done." was printed, no article was ever written, and
every later run skipped the file as already-ingested.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import utils  # noqa: E402


# ── prompt sizing: the wiki index must not dominate the prompt ───────────────

INDEX_HEADER = (
    "# Knowledge Base Index\n\n"
    "| Article | Summary | Compiled From | Updated |\n"
    "|---------|---------|---------------|---------|\n"
)


def _write_index(tmp_path: Path, rows: int, summary_len: int = 400) -> Path:
    body = "".join(
        f"| [[concepts/article-{i}]] | {'x' * summary_len} | daily/2026-01-0{i % 9 + 1}.md | 2026-07-0{i % 9 + 1} |\n"
        for i in range(rows)
    )
    path = tmp_path / "index.md"
    path.write_text(INDEX_HEADER + body, encoding="utf-8")
    return path


@pytest.fixture
def index(tmp_path, monkeypatch):
    path = _write_index(tmp_path, rows=650)
    monkeypatch.setattr(utils, "INDEX_FILE", path)
    return path


def test_compact_index_is_dramatically_smaller(index) -> None:
    full = utils.read_wiki_index()
    compact = utils.read_wiki_index(compact=True)
    assert len(compact) < len(full) * 0.25, (
        f"compact index is {len(compact)} vs full {len(full)} — not enough "
        "reduction to keep the prompt inside the context window"
    )


def test_compact_index_keeps_every_article_row(index) -> None:
    """Slugs are load-bearing: wikilink targets and UPDATE-vs-CREATE decisions."""
    full = utils.read_wiki_index()
    compact = utils.read_wiki_index(compact=True)
    for i in range(650):
        assert f"[[concepts/article-{i}]]" in compact, f"lost article-{i}"
    assert full.count("[[concepts/article-") == compact.count("[[concepts/article-")


def test_compact_index_drops_the_summary_column(index) -> None:
    compact = utils.read_wiki_index(compact=True)
    assert "xxxxx" not in compact, "summary prose should be gone"
    assert "2026-07-" in compact, "updated date should be kept"


def test_default_is_still_the_full_index(index) -> None:
    """compact must be opt-in — other callers rely on the summaries."""
    assert "xxxxx" in utils.read_wiki_index()


def test_missing_index_returns_header_in_both_modes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(utils, "INDEX_FILE", tmp_path / "nope.md")
    assert "Knowledge Base Index" in utils.read_wiki_index()
    assert "Knowledge Base Index" in utils.read_wiki_index(compact=True)


def test_bom_prefixed_index_still_compacts(tmp_path, monkeypatch) -> None:
    """The real index is written with a BOM; it must not defeat row parsing."""
    path = tmp_path / "index.md"
    path.write_text("﻿" + INDEX_HEADER +
                    "| [[concepts/a]] | summary here | src.md | 2026-07-01 |\n",
                    encoding="utf-8")
    monkeypatch.setattr(utils, "INDEX_FILE", path)
    compact = utils.read_wiki_index(compact=True)
    assert "[[concepts/a]]" in compact
    assert "summary here" not in compact


def test_malformed_rows_pass_through_untouched(tmp_path, monkeypatch) -> None:
    path = tmp_path / "index.md"
    path.write_text(INDEX_HEADER + "| only-two | cells |\nnot a table row\n",
                    encoding="utf-8")
    monkeypatch.setattr(utils, "INDEX_FILE", path)
    compact = utils.read_wiki_index(compact=True)
    assert "not a table row" in compact
    assert "only-two" in compact


# ── write verification ───────────────────────────────────────────────────────

def _articles_env(tmp_path, monkeypatch):
    import ingest
    concepts = tmp_path / "concepts"
    concepts.mkdir()
    monkeypatch.setattr(ingest, "CONCEPTS_DIR", concepts)
    monkeypatch.setattr(ingest, "CONNECTIONS_DIR", tmp_path / "connections")
    return ingest, concepts


SRC = "sources/research-output/thing.md"


def test_new_article_crediting_the_source_counts(tmp_path, monkeypatch) -> None:
    ingest, concepts = _articles_env(tmp_path, monkeypatch)
    before = ingest._articles_snapshot()
    (concepts / "new.md").write_text(f"a claim [src:{SRC}]", encoding="utf-8")
    assert ingest._articles_written_for(before, SRC)


def test_nothing_written_is_a_failure(tmp_path, monkeypatch) -> None:
    ingest, _ = _articles_env(tmp_path, monkeypatch)
    before = ingest._articles_snapshot()
    assert not ingest._articles_written_for(before, SRC)


def test_concurrent_unrelated_write_is_not_accepted(tmp_path, monkeypatch) -> None:
    """The P1: another process editing the KB must not prove OUR ingest worked."""
    ingest, concepts = _articles_env(tmp_path, monkeypatch)
    before = ingest._articles_snapshot()
    (concepts / "someone-else.md").write_text(
        "unrelated [src:sources/other/xyz.md]", encoding="utf-8")
    assert not ingest._articles_written_for(before, SRC)


def test_touch_without_content_change_is_not_accepted(tmp_path, monkeypatch) -> None:
    """mtime-only detection accepted a bare touch; content hashing must not."""
    import os
    ingest, concepts = _articles_env(tmp_path, monkeypatch)
    existing = concepts / "old.md"
    existing.write_text(f"claim [src:{SRC}]", encoding="utf-8")
    before = ingest._articles_snapshot()
    st = existing.stat()
    os.utime(existing, (st.st_atime + 99, st.st_mtime + 99))
    assert not ingest._articles_written_for(before, SRC), "touch is not evidence"


def test_same_tick_content_change_is_detected(tmp_path, monkeypatch) -> None:
    """The mtime false-negative: content changed, timestamp may not have."""
    import os
    ingest, concepts = _articles_env(tmp_path, monkeypatch)
    existing = concepts / "old.md"
    existing.write_text("before", encoding="utf-8")
    before = ingest._articles_snapshot()
    st = existing.stat()
    existing.write_text(f"after [src:{SRC}]", encoding="utf-8")
    os.utime(existing, (st.st_atime, st.st_mtime))  # pin mtime back
    assert ingest._articles_written_for(before, SRC)


# ── retry bounding ───────────────────────────────────────────────────────────

@pytest.fixture
def isolated_state(monkeypatch):
    """Stub save_state.

    record_failure persists through ingest.save_state, which writes the REAL
    scripts/state.json. Without this stub these tests overwrite it with their
    own throwaway dict — which is exactly what happened on 2026-07-26,
    destroying 494 ingested-source records and 2197 codebase hashes.
    """
    import ingest
    monkeypatch.setattr(ingest, "save_state", lambda *_a, **_k: None)
    return ingest


def test_failures_are_bounded_then_skipped(isolated_state, monkeypatch) -> None:
    """A permanently-failing file must not be retried forever."""
    ingest = isolated_state
    state: dict = {}
    key, h = "research-output/bad.md", "hash-1"
    for _ in range(ingest.MAX_INGEST_ATTEMPTS):
        assert not ingest.should_skip_after_failures(state, key, h)
        ingest.record_failure(state, key, h)
    assert ingest.should_skip_after_failures(state, key, h)


def test_editing_a_failed_source_resets_its_attempts(isolated_state) -> None:
    ingest = isolated_state
    state: dict = {}
    key = "research-output/bad.md"
    for _ in range(ingest.MAX_INGEST_ATTEMPTS):
        ingest.record_failure(state, key, "hash-1")
    assert ingest.should_skip_after_failures(state, key, "hash-1")
    assert not ingest.should_skip_after_failures(state, key, "hash-2")


def test_failures_never_land_in_ingested_sources(isolated_state) -> None:
    """Recording a success hash is what makes a file invisible — keep them apart."""
    ingest = isolated_state
    state: dict = {"ingested_sources": {}}
    ingest.record_failure(state, "research-output/bad.md", "h")
    assert state["ingested_sources"] == {}
    assert "research-output/bad.md" in state["failed_sources"]


# ── the actual regression: a CLI failure that explains itself on stdout ───────

def _drive_ingest(tmp_path, monkeypatch, *, returncode, stdout, write_article):
    """Run ingest_source_file against a faked claude CLI. Returns (ok, state)."""
    import asyncio
    import subprocess as sp
    import ingest

    concepts = tmp_path / "concepts"
    concepts.mkdir()
    monkeypatch.setattr(ingest, "CONCEPTS_DIR", concepts)
    monkeypatch.setattr(ingest, "CONNECTIONS_DIR", tmp_path / "connections")

    source = tmp_path / "doc.md"
    source.write_text("# a source document", encoding="utf-8")

    group = utils.SourceGroup(
        id="research-output", type="markdown", include=[], exclude=[],
        category="research", description="test",
    )
    rel_source = f"sources/{group.id}/{source.name}"

    def fake_run(*_a, **_kw):
        if write_article:
            (concepts / "written.md").write_text(
                f"claim [src:{rel_source}]", encoding="utf-8")
        return sp.CompletedProcess(args=["claude"], returncode=returncode,
                                   stdout=stdout, stderr="")

    monkeypatch.setattr(ingest.subprocess, "run", fake_run)
    monkeypatch.setattr(ingest, "save_state", lambda *_a, **_k: None)
    monkeypatch.setattr(ingest.dedup, "similar_to_text", lambda *a, **k: [])
    monkeypatch.setattr(ingest.dedup, "format_preflight_block", lambda *a, **k: "")

    state: dict = {"ingested_sources": {}}
    _cost, ok = asyncio.run(ingest.ingest_source_file(group, source, state))
    return ok, state


def test_prompt_too_long_on_stdout_is_a_failure(tmp_path, monkeypatch) -> None:
    """THE regression: exit 1 + explanation on stdout was treated as success."""
    ok, state = _drive_ingest(tmp_path, monkeypatch, returncode=1,
                              stdout="Prompt is too long\n", write_article=False)
    assert ok is False
    assert state["ingested_sources"] == {}, (
        "recording the hash is what made the file permanently invisible"
    )


def test_exit_zero_without_an_article_is_a_failure(tmp_path, monkeypatch) -> None:
    ok, state = _drive_ingest(tmp_path, monkeypatch, returncode=0,
                              stdout="I thought about it.\n", write_article=False)
    assert ok is False
    assert state["ingested_sources"] == {}


def test_exit_zero_with_an_article_succeeds(tmp_path, monkeypatch) -> None:
    ok, state = _drive_ingest(tmp_path, monkeypatch, returncode=0,
                              stdout="done\n", write_article=True)
    assert ok is True
    assert list(state["ingested_sources"]) == ["research-output/doc.md"]


def test_aliased_wikilink_survives_compaction(tmp_path, monkeypatch) -> None:
    """[[slug|Label]] contains a pipe — splitting the row on | truncates it."""
    path = tmp_path / "index.md"
    path.write_text(
        INDEX_HEADER
        + "| [[concepts/foo|Foo Label]] | a summary | src.md | 2026-07-01 |\n"
        + "| [[concepts/bar]] | has a | pipe inside | src.md | 2026-07-02 |\n",
        encoding="utf-8")
    monkeypatch.setattr(utils, "INDEX_FILE", path)
    compact = utils.read_wiki_index(compact=True)
    assert "[[concepts/foo|Foo Label]]" in compact, "aliased link was truncated"
    assert "[[concepts/bar]]" in compact
    assert "a summary" not in compact
