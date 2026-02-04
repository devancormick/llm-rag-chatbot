"""Lightweight registry for tracking document IDs across vector stores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import config


class DocumentTracker:
    """Tracks which document IDs have been indexed."""

    def __init__(self, path: Path | None = None):
        self.path = path or config.DATA_DIR / "vector_documents.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> set[str]:
        if not self.path.exists():
            return set()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return set(str(item) for item in data)
        except json.JSONDecodeError:
            pass
        return set()

    def _write(self, ids: Iterable[str]) -> None:
        unique = sorted(set(ids))
        self.path.write_text(json.dumps(unique, indent=2), encoding="utf-8")

    def add(self, document_id: str | None) -> None:
        if not document_id:
            return
        ids = self._read()
        if document_id in ids:
            return
        ids.add(document_id)
        self._write(ids)

    def remove(self, document_id: str | None) -> None:
        if not document_id:
            return
        ids = self._read()
        if document_id not in ids:
            return
        ids.remove(document_id)
        self._write(ids)

    def list(self) -> list[str]:
        return sorted(self._read())
