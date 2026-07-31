"""Test-suite guards for the developer's live state file.

``scripts/state.json`` is real working data: ingest hashes, daily records,
codebase hashes, lifetime cost. A test that writes to it destroys work.

That is not hypothetical. On 2026-07-31 a full-suite run truncated the real
state.json to ``{}``, losing 496 ingest records and 83 daily records — the
second such loss in a week. Running test files one at a time did not
reproduce it, so nothing short of a suite-wide guard would have caught it.

Two layers, deliberately redundant:

1. ``_redirect_state_file`` points ``config.STATE_FILE`` and
   ``utils.STATE_FILE`` at a per-test tmp file, so production code called
   from a test writes somewhere harmless by default. Tests that do their own
   redirection simply override this.
2. ``_protect_real_state_file`` checks that no top-level key *shrank or
   vanished* across each test, restores the previous content if one did, and
   fails that test by name. This catches anything layer 1 misses — a module
   that captured the path at import time, a subprocess, a direct open().

Layer 2 deliberately checks shrinkage rather than byte equality. A
legitimate background writer (an ingest run, the scheduler) may be appending
to this file while the suite runs; comparing bytes both raises false
failures and, far worse, restores a stale snapshot over that writer's
progress. Records appearing is normal — records disappearing is the bug.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_REAL_STATE = Path(__file__).resolve().parent.parent / "scripts" / "state.json"


@pytest.fixture(autouse=True)
def _redirect_state_file(tmp_path, monkeypatch):
    """Send state writes to a throwaway file unless a test says otherwise."""
    import config
    import utils

    sandbox = tmp_path / "state.json"
    monkeypatch.setattr(config, "STATE_FILE", sandbox, raising=False)
    monkeypatch.setattr(utils, "STATE_FILE", sandbox, raising=False)
    return sandbox


def _record_census(path: Path) -> dict[str, int] | None:
    """Top-level keys mapped to their record counts, or None if absent."""
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Mid-write or unreadable: treat as unknown rather than as loss.
        return None
    return {k: len(v) if isinstance(v, (dict, list)) else 1 for k, v in state.items()}


@pytest.fixture(autouse=True)
def _protect_real_state_file(request):
    """Backstop: fail any test that *destroys* records in the real file.

    Deliberately checks shrinkage rather than byte equality. A legitimate
    background process — an ingest run, the scheduler — writes to this file
    while the suite runs, so comparing bytes both raises false failures and,
    far worse, would restore a stale snapshot over that process's progress.

    Records appearing is normal. Records or whole keys *disappearing* is the
    failure mode that cost real data, and it is what this catches.
    """
    before = _record_census(_REAL_STATE)
    snapshot = _REAL_STATE.read_bytes() if _REAL_STATE.exists() else None

    yield

    after = _record_census(_REAL_STATE)
    if before is None or after is None:
        return

    lost = [
        key for key, count in before.items()
        if key not in after or after[key] < count
    ]
    if not lost:
        return

    if snapshot is not None:
        _REAL_STATE.write_bytes(snapshot)

    pytest.fail(
        f"{request.node.nodeid} destroyed records in the real state.json "
        f"({_REAL_STATE}): {', '.join(sorted(lost))} shrank or vanished. "
        f"Redirect config.STATE_FILE and utils.STATE_FILE to tmp_path. "
        f"The previous content has been restored."
    )
