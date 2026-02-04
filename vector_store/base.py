"""Abstract interfaces shared by vector store providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from ingestion.chunker import TextChunk
from .document_tracker import DocumentTracker


@dataclass
class SearchResult:
    content: str
    metadata: dict
    distance: float


class VectorStore(ABC):
    """Abstract base class for vector store providers."""

    def __init__(self, tracker: DocumentTracker | None = None):
        self.tracker = tracker or DocumentTracker()

    @abstractmethod
    def add_chunks(
        self,
        chunks: List[TextChunk],
        document_id: Optional[str] = None,
    ) -> int:
        """Add chunks to the vector index."""

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int,
        similarity_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search the index for relevant chunks."""

    @abstractmethod
    def delete_by_document_id(self, document_id: str) -> None:
        """Remove all vectors belonging to a document."""

    def list_documents(self) -> List[str]:
        """Return a tracked list of document IDs."""
        return self.tracker.list()

    def _track_add(self, document_id: Optional[str]) -> None:
        self.tracker.add(document_id)

    def _track_remove(self, document_id: Optional[str]) -> None:
        self.tracker.remove(document_id)
