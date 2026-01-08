from typing import List
from dataclasses import dataclass

from .parsers import DocumentChunk

import config


@dataclass
class TextChunk:
    content: str
    metadata: dict


class TextChunker:
    """Splits document content into overlapping chunks for embedding."""

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents: List[DocumentChunk]) -> List[TextChunk]:
        """Split documents into overlapping text chunks."""
        all_chunks = []
        for doc in documents:
            chunks = self._chunk_text(doc.content, doc.metadata)
            all_chunks.extend(chunks)
        return all_chunks

    def _chunk_text(self, text: str, metadata: dict) -> List[TextChunk]:
        """Split text into overlapping chunks."""
        if not text or not text.strip():
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk_content = text[start:end]

            if not chunk_content.strip():
                start = end - self.chunk_overlap
                continue

            chunks.append(
                TextChunk(content=chunk_content.strip(), metadata=metadata.copy())
            )
            start = end - self.chunk_overlap

        return chunks
