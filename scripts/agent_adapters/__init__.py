"""Pluggable agent-history adapter registry.

Modeled on ``source_handlers/__init__.py``. Each adapter is a module that
exposes a pure ``parse(lines: list[str]) -> AgentSession`` function and a
``default_store() -> Path`` function (the agent's sessions root, expanded
from ``~``). Adapters register themselves via :func:`register`; the
built-in codex/claude adapters auto-register on import (see bottom).

The registry is the single extension point: to add a new agent (e.g.
Hermes, once its transcript format is confirmed) drop a module here that
calls ``register(...)`` at import time and add it to the auto-register
block below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# AgentSession is the normalized model shared across adapters. It lives in
# import_agent_history; we re-export it here so adapter modules can import it
# from the package root without a circular dependency on the CLI module.
try:
    from scripts.import_agent_history import AgentSession  # noqa: F401
except ImportError:  # pragma: no cover - exercised only outside package context
    from import_agent_history import AgentSession  # type: ignore  # noqa: F401


ParseFn = Callable[[list[str]], "AgentSession"]
StoreFn = Callable[[], Path]


@dataclass
class Adapter:
    """A registered agent-history adapter."""

    name: str
    parse: ParseFn
    default_store: StoreFn
    glob: str


_ADAPTERS: dict[str, Adapter] = {}


def register(name: str, parse: ParseFn, default_store: StoreFn, glob: str) -> None:
    _ADAPTERS[name] = Adapter(
        name=name, parse=parse, default_store=default_store, glob=glob
    )


def get_adapter(name: str) -> Adapter:
    if name not in _ADAPTERS:
        available = ", ".join(sorted(_ADAPTERS)) or "(none)"
        raise KeyError(
            f"No adapter registered for agent '{name}'. Available: {available}"
        )
    return _ADAPTERS[name]


def available_agents() -> list[str]:
    return sorted(_ADAPTERS)


# Auto-register built-in adapters on import.
try:  # pragma: no cover - import-path shim
    from scripts.agent_adapters import codex as _codex  # noqa: E402, F401
    from scripts.agent_adapters import claude as _claude  # noqa: E402, F401
except ImportError:  # pragma: no cover
    from agent_adapters import codex as _codex  # type: ignore  # noqa: E402, F401
    from agent_adapters import claude as _claude  # type: ignore  # noqa: E402, F401
