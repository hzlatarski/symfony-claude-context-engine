"""Markdown source handler — reads .md files, optionally parses YAML frontmatter."""

from __future__ import annotations

from pathlib import Path

from source_handlers import SourceDocument, register


def extract(path: Path) -> SourceDocument:
    raw = path.read_text(encoding="utf-8")
    frontmatter: dict = {}
    content = raw

    if raw.startswith("---"):
        end = raw.find("---", 3)
        if end != -1:
            fm_text = raw[3:end].strip()
            content = raw[end + 3:].strip()
            try:
                import yaml
                frontmatter = yaml.safe_load(fm_text) or {}
            except Exception:
                frontmatter = {}

    return SourceDocument(
        content=content,
        path=path,
        frontmatter=frontmatter,
        mtime=path.stat().st_mtime,
    )


register("markdown", extract)
