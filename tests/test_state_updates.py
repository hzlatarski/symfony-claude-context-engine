"""Regression tests for atomic, merge-safe shared state."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import utils  # noqa: E402


def test_stale_snapshots_do_not_clobber_unrelated_nested_updates(tmp_path) -> None:
    path = tmp_path / "state.json"
    utils.save_state({"ingested_daily": {}, "codebase_hashes": {}}, path)

    compiler_snapshot = utils.load_state(path)
    indexer_snapshot = utils.load_state(path)
    compiler_snapshot["ingested_daily"]["2026-07-28.md"] = {"hash": "daily"}
    indexer_snapshot["codebase_hashes"]["src/Foo.php"] = "code"

    utils.save_state(compiler_snapshot, path)
    utils.save_state(indexer_snapshot, path)

    state = utils.load_state(path)
    assert state["ingested_daily"]["2026-07-28.md"]["hash"] == "daily"
    assert state["codebase_hashes"]["src/Foo.php"] == "code"


def test_concurrent_mutations_to_same_mapping_both_survive(tmp_path) -> None:
    path = tmp_path / "state.json"
    utils.save_state({"ingested_daily": {}}, path)
    ready = threading.Barrier(3)

    def worker(name: str) -> None:
        ready.wait()

        def mutate(state: dict) -> None:
            state.setdefault("ingested_daily", {})[name] = {"hash": name}

        utils.update_state(mutate, path)

    threads = [
        threading.Thread(target=worker, args=("a.md",)),
        threading.Thread(target=worker, args=("b.md",)),
    ]
    for thread in threads:
        thread.start()
    ready.wait()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    state = json.loads(path.read_text(encoding="utf-8"))
    assert set(state["ingested_daily"]) == {"a.md", "b.md"}


def test_locked_mutation_can_remove_obsolete_key(tmp_path) -> None:
    path = tmp_path / "state.json"
    utils.save_state({"ingested": {"old.md": {}}, "ingested_daily": {}}, path)

    utils.update_state(lambda state: state.pop("ingested", None), path)

    assert "ingested" not in utils.load_state(path)


class TestLoadStateNormalization:
    """A state.json missing a top-level key must not crash its writers.

    Observed 2026-07-30: the live state.json had lost "ingested_sources"
    entirely. load_state returned the raw dict, and ingest.py's
    `state["ingested_sources"][key] = entry` raised KeyError after
    re-processing all 497 source files — so a full ingest run burned its
    work and wrote nothing. migrate_state_schema already guarded this,
    but load_state never called it.
    """

    def test_missing_ingested_sources_is_restored_as_empty(self, tmp_path):
        import json

        import utils

        path = tmp_path / "state.json"
        path.write_text(json.dumps({"ingested_daily": {}, "codebase_hashes": {}}), encoding="utf-8")

        state = utils.load_state(path)
        assert state["ingested_sources"] == {}

    def test_every_default_key_is_present(self, tmp_path):
        import json

        import utils

        path = tmp_path / "state.json"
        path.write_text(json.dumps({"ingested_daily": {}}), encoding="utf-8")

        state = utils.load_state(path)
        for key in utils._default_state():
            assert key in state, f"load_state dropped default key {key!r}"

    def test_existing_values_are_never_overwritten(self, tmp_path):
        import json

        import utils

        path = tmp_path / "state.json"
        path.write_text(
            json.dumps({"ingested_sources": {"a.md": {"hash": "x"}}, "query_count": 7}),
            encoding="utf-8",
        )

        state = utils.load_state(path)
        assert state["ingested_sources"] == {"a.md": {"hash": "x"}}
        assert state["query_count"] == 7

    def test_unknown_keys_are_preserved(self, tmp_path):
        import json

        import utils

        path = tmp_path / "state.json"
        path.write_text(json.dumps({"failed_sources": {"z.md": 1}}), encoding="utf-8")

        assert utils.load_state(path)["failed_sources"] == {"z.md": 1}


class TestLoadStateWarnsOnBackfill:
    """Backfilling a missing key must be visible, not silent.

    Inserting `{}` for an absent `ingested_sources` keeps the compiler
    running, but it also turns evidence of a corrupt or truncated write into
    ordinary-looking empty state. The warning is what distinguishes "new
    install" from "your state file lost 496 records".
    """

    def test_missing_key_emits_a_warning(self, tmp_path, capsys):
        import json

        import utils

        path = tmp_path / "state.json"
        path.write_text(json.dumps({"ingested_daily": {}}), encoding="utf-8")

        utils.load_state(path)
        err = capsys.readouterr().err
        assert "ingested_sources" in err

    def test_complete_state_is_silent(self, tmp_path, capsys):
        import json

        import utils

        path = tmp_path / "state.json"
        path.write_text(json.dumps(utils._default_state()), encoding="utf-8")

        utils.load_state(path)
        assert capsys.readouterr().err == ""
