"""
Symfony code intelligence parsers.

Five pure-Python parsers that understand PHP dependencies, Symfony routes,
Twig templates, Stimulus controllers, and git history. Each parser exposes:

    parse(project_root: Path) -> dict   # full structured result (for MCP)
    summary(project_root: Path) -> str  # short markdown (for session-start)

No LLM calls. No vendor deps. Just regex on real files.
"""
from __future__ import annotations

import os
from pathlib import Path

# scripts/parsers/__init__.py → scripts/parsers → scripts → memory-compiler → .claude → project_root
from config import PROJECT_ROOT


def code_root(project_root: Path) -> Path:
    """Directory that holds the Symfony app (``src/``, ``templates/``, ``assets/``).

    Usually the project root. Some repos keep the app in a subfolder
    (e.g. ``app/``). Set ``MEMORY_COMPILER_CODE_DIR`` to that subfolder, or
    leave it unset to auto-detect: if ``<root>/src`` is missing and exactly
    one direct child has ``src/``, that child is used.
    """
    override = os.environ.get("MEMORY_COMPILER_CODE_DIR", "").strip().strip("/\\")
    if override:
        return project_root / override
    if (project_root / "src").is_dir():
        return project_root
    candidates = [
        d for d in sorted(project_root.iterdir())
        if d.is_dir() and not d.name.startswith(".") and (d / "src").is_dir()
    ] if project_root.is_dir() else []
    return candidates[0] if len(candidates) == 1 else project_root


CODE_ROOT = code_root(PROJECT_ROOT)
# Repo-relative prefix of CODE_ROOT, e.g. "app/" (empty when the app is at the root).
CODE_PREFIX = "" if CODE_ROOT == PROJECT_ROOT else CODE_ROOT.relative_to(PROJECT_ROOT).as_posix() + "/"


def to_code_rel(path: str) -> str:
    """Map a repo-relative path (``app/src/X.php``) to a graph key (``src/X.php``)."""
    path = path.replace("\\", "/")
    if CODE_PREFIX and path.startswith(CODE_PREFIX):
        return path[len(CODE_PREFIX):]
    return path
