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


class TestIngestMigrationDoesNotWipeState:
    """`ingest.py` truncated the live state.json to `{}` on every run.

    Its migrate mutator did:

        migrated = migrate_state_schema(current)
        current.clear()
        current.update(migrated)

    `migrate_state_schema` mutates in place and returns the *same* object, so
    `current.clear()` emptied `migrated` too and the update copied nothing
    back. Observed twice in one week: 496 ingest records and 83 daily records
    destroyed, and a full re-ingest of 497 files launched because the history
    looked empty.
    """

    def test_migrate_state_schema_does_not_mutate_its_argument(self):
        # It used to mutate in place AND return the same object, which made
        # the natural `clear(); update(migrated)` idiom destroy the state.
        # Purity is what makes that idiom safe for every caller, present and
        # future — two callers wrote it independently and both lost data.
        import utils

        original = {"ingested_daily": {"a.md": {"hash": "x"}}, "total_cost": 1.5}
        snapshot = {"ingested_daily": {"a.md": {"hash": "x"}}, "total_cost": 1.5}

        result = utils.migrate_state_schema(original)

        assert original == snapshot, "migrate_state_schema mutated its input"
        assert result is not original

    def test_the_naive_clear_update_idiom_is_now_safe(self):
        # Exactly what compile.py and ingest.py both wrote.
        import utils

        current = {"ingested_daily": {"a.md": {}}, "ingested_sources": {"b.md": {}}}
        migrated = utils.migrate_state_schema(current)
        current.clear()
        current.update(migrated)

        assert current["ingested_sources"] == {"b.md": {}}
        assert current["ingested_daily"] == {"a.md": {}}

    def test_the_ingest_mutator_preserves_existing_records(self):
        import ingest

        state = {
            "ingested_daily": {"2026-04-09.md": {"hash": "a"}},
            "ingested_sources": {"design-specs/x.md": {"hash": "b"}},
            "total_cost": 208.18,
        }
        ingest.migrate_state_mutator(state)

        assert state["ingested_sources"] == {"design-specs/x.md": {"hash": "b"}}
        assert state["ingested_daily"] == {"2026-04-09.md": {"hash": "a"}}
        assert state["total_cost"] == 208.18

    def test_the_ingest_mutator_still_migrates_the_legacy_schema(self):
        import ingest

        state = {"ingested": {"2026-04-09.md": {"hash": "a"}}}
        ingest.migrate_state_mutator(state)

        assert state["ingested_daily"] == {"2026-04-09.md": {"hash": "a"}}
        assert state["ingested_sources"] == {}
        assert "ingested" not in state


class TestCompileMigrationDoesNotWipeState:
    """`compile.py` carried the same clobber as `ingest.py`.

    Fixing only ingest left the scheduler's periodic compile run destroying
    state.json on its own schedule — which is why the loss appeared to
    continue after the ingest fix was verified working.
    """

    def test_the_compile_mutator_preserves_existing_records(self):
        import compile as compile_module

        state = {
            "ingested_daily": {"2026-04-09.md": {"hash": "a"}},
            "ingested_sources": {"design-specs/x.md": {"hash": "b"}},
            "codebase_hashes": {"src/App.php": "deadbeef"},
            "total_cost": 208.18,
        }
        compile_module.migrate_state_mutator(state)

        assert state["ingested_sources"] == {"design-specs/x.md": {"hash": "b"}}
        assert state["ingested_daily"] == {"2026-04-09.md": {"hash": "a"}}
        assert state["codebase_hashes"] == {"src/App.php": "deadbeef"}
        assert state["total_cost"] == 208.18

    def test_ingest_and_compile_share_one_implementation(self):
        # Two copies drifted apart once already; one of them was never fixed.
        import compile as compile_module
        import ingest
        import utils

        assert ingest.migrate_state_mutator is utils.migrate_state_mutator
        assert compile_module.migrate_state_mutator is utils.migrate_state_mutator
