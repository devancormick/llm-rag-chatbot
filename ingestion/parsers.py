from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from pypdf import PdfReader


@dataclass
class DocumentChunk:
    content: str
    metadata: dict
    source: str
    page: Optional[int] = None


class DocumentParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def parse(self, path: Path) -> List[DocumentChunk]:
        """Parse document and return raw content with metadata."""
        pass

    @abstractmethod
    def supports(self, path: Path) -> bool:
        """Check if this parser supports the given file type."""
        pass


class PDFParser(DocumentParser):
    """Parser for PDF documents."""

    def parse(self, path: Path) -> List[DocumentChunk]:
        chunks = []
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                chunks.append(
                    DocumentChunk(
                        content=text,
                        metadata={"source": path.name, "page": i + 1},
                        source=str(path),
                        page=i + 1,
                    )
                )
        return chunks

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"


class MarkdownParser(DocumentParser):
    """Parser for Markdown documents."""

    def parse(self, path: Path) -> List[DocumentChunk]:
        content = path.read_text(encoding="utf-8", errors="ignore")
        return [
            DocumentChunk(
                content=content,
                metadata={"source": path.name},
                source=str(path),
                page=None,
            )
        ]

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in (".md", ".markdown")
