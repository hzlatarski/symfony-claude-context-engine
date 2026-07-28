"""Safe resolution and migration of the replaceable Chroma active directory."""

from __future__ import annotations

import re
from pathlib import Path

from filelock import FileLock

_UUID_DIR = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_MIGRATION_MARKER = ".active-migration-in-progress"


def _legacy_artifacts(root: Path) -> list[Path]:
    if not root.exists():
        return []
    artifacts: list[Path] = []
    sqlite = root / "chroma.sqlite3"
    if sqlite.exists():
        artifacts.append(sqlite)
    artifacts.extend(
        path for path in root.iterdir()
        if path.is_dir() and _UUID_DIR.fullmatch(path.name)
    )
    return sorted(artifacts, key=lambda path: path.name)


def ensure_active_chroma_dir(root: Path) -> Path:
    """Migrate an unambiguous legacy store and return ``root/active``."""
    root = root.resolve()
    lock_path = root.parent / f".{root.name}.active-migration.lock"
    root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path)):
        root.mkdir(parents=True, exist_ok=True)
        active = root / "active"
        legacy = _legacy_artifacts(root)
        marker = root / _MIGRATION_MARKER
        resuming = marker.exists()
        if active.exists() and legacy and not resuming:
            names = ", ".join(path.name for path in legacy)
            raise RuntimeError(
                "Chroma contains both active and legacy storage; refusing an "
                f"ambiguous migration. Legacy artifacts: {names}"
            )
        if legacy or resuming:
            if not resuming:
                marker.write_text("legacy-to-active\n", encoding="utf-8")
            active.mkdir(exist_ok=True)
            moved: list[tuple[Path, Path]] = []
            try:
                for source in legacy:
                    destination = active / source.name
                    if destination.exists():
                        raise RuntimeError(
                            "Interrupted Chroma migration has the same artifact "
                            f"in both locations: {source.name}"
                        )
                    source.replace(destination)
                    moved.append((source, destination))
                marker.unlink(missing_ok=True)
            except (OSError, RuntimeError):
                for source, destination in reversed(moved):
                    if destination.exists() and not source.exists():
                        destination.replace(source)
                raise
        elif not active.exists():
            active.mkdir()
        return active


def ensure_configured_chroma_dir(configured: Path) -> Path:
    """Honor explicit test/custom paths; migrate only the standard active path."""
    configured = configured.resolve()
    if configured.name == "active":
        return ensure_active_chroma_dir(configured.parent)
    configured.mkdir(parents=True, exist_ok=True)
    return configured
