"""
Symfony code intelligence parsers.

Five pure-Python parsers that understand PHP dependencies, Symfony routes,
Twig templates, Stimulus controllers, and git history. Each parser exposes:

    parse(project_root: Path) -> dict   # full structured result (for MCP)
    summary(project_root: Path) -> str  # short markdown (for session-start)

No LLM calls. No vendor deps. Just regex on real files.
"""
from __future__ import annotations

from pathlib import Path

# scripts/parsers/__init__.py → scripts/parsers → scripts → memory-compiler → .claude → project_root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
