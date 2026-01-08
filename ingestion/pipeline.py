from pathlib import Path
from typing import List, Optional

from .parsers import DocumentParser, PDFParser, MarkdownParser
from .chunker import TextChunker, TextChunk


class IngestionPipeline:
    """Orchestrates document parsing and chunking."""

    def __init__(self):
        self.parsers: List[DocumentParser] = [
            PDFParser(),
            MarkdownParser(),
        ]
        self.chunker = TextChunker()

    def _get_parser(self, path: Path) -> Optional[DocumentParser]:
        for parser in self.parsers:
            if parser.supports(path):
                return parser
        return None

    def process_file(self, path: Path) -> List[TextChunk]:
        parser = self._get_parser(path)
        if not parser:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        documents = parser.parse(path)
        return self.chunker.chunk_documents(documents)
