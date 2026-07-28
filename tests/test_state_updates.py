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
