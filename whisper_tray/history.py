from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

HISTORY_DIR = Path.home() / ".whisper-tray"
HISTORY_FILE = HISTORY_DIR / "history.json"


@dataclass
class HistoryEntry:
    transcript: str
    enhanced_prompt: str
    mode: Literal["verbatim", "rewrite", "clean"]


class History:
    def __init__(self, persist_path: Path = HISTORY_FILE, maxlen: int = 10) -> None:
        self._path = persist_path
        self._buf: deque[HistoryEntry] = deque(maxlen=maxlen)
        self._load()

    def append(self, entry: HistoryEntry) -> None:
        self._buf.append(entry)
        self._save()

    def last(self) -> HistoryEntry | None:
        return self._buf[-1] if self._buf else None

    def entries(self) -> list[HistoryEntry]:
        return list(self._buf)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps([asdict(e) for e in self._buf], indent=2), encoding="utf-8"
        )

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for item in data:
                self._buf.append(HistoryEntry(**item))
        except (json.JSONDecodeError, TypeError, OSError):
            pass
