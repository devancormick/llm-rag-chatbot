from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np

import config
from ingestion.chunker import TextChunk
from vector_store.base import SearchResult, VectorStore
from vector_store.embeddings import get_embedding_function


class FaissVectorStore(VectorStore):
    """Vector store backed by FAISS (local index)."""

    def __init__(
        self,
        index_path: str | None = None,
        metadata_path: str | None = None,
        dimension: int | None = None,
        normalize: bool | None = None,
        tracker=None,
    ):
        super().__init__(tracker=tracker)
        self.index_path = Path(index_path or config.FAISS_INDEX_PATH)
        self.metadata_path = Path(metadata_path or config.FAISS_METADATA_PATH)
        self.dimension = dimension or config.FAISS_DIMENSION
        self.normalize = (
            normalize
            if normalize is not None
            else getattr(config, "FAISS_NORMALIZE", True)
        )
        self.embedding_fn = get_embedding_function()

        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss-cpu is required for the FAISS provider.") from exc

        self.faiss = faiss
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata = self._load_metadata()
        self.index = self._load_or_create_index()

    def _load_metadata(self) -> dict[str, dict]:
        if not self.metadata_path.exists():
            return {}
        try:
            data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        return {}

    def _save_metadata(self) -> None:
        self.metadata_path.write_text(
            json.dumps(self._metadata, indent=2), encoding="utf-8"
        )

    def _load_or_create_index(self):
        if self.index_path.exists():
            return self.faiss.read_index(str(self.index_path))
        base = self.faiss.IndexFlatIP(self.dimension)
        return self.faiss.IndexIDMap2(base)

    def _persist_index(self) -> None:
        self.faiss.write_index(self.index, str(self.index_path))

    def _embed(self, texts: List[str]) -> np.ndarray:
        vectors = np.array(self.embedding_fn(texts)).astype("float32")
        if self.normalize:
            self.faiss.normalize_L2(vectors)
        return vectors

    def add_chunks(
        self,
        chunks: List[TextChunk],
        document_id: Optional[str] = None,
    ) -> int:
        if not chunks:
            return 0

        embeddings = self._embed([c.content for c in chunks])
        ids = []
        for chunk in chunks:
            uid = uuid.uuid4().int & ((1 << 63) - 1)
            ids.append(uid)
            self._metadata[str(uid)] = {
                **chunk.metadata,
                "content": chunk.content,
                "document_id": document_id or "unknown",
            }

        id_array = np.array(ids, dtype="int64")
        self.index.add_with_ids(embeddings, id_array)
        self._save_metadata()
        self._persist_index()

        if document_id:
            self._track_add(document_id)
        return len(ids)

    def search(
        self,
        query: str,
        top_k: int,
        similarity_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        if self.index.ntotal == 0:
            return []
        embedding = self._embed([query])
        scores, ids = self.index.search(embedding, top_k)
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else getattr(config, "SIMILARITY_THRESHOLD", 0.7)
        )
        results: List[SearchResult] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            if score < threshold:
                continue
            metadata = self._metadata.get(str(idx), {})
            results.append(
                SearchResult(
                    content=metadata.get("content", ""),
                    metadata=metadata,
                    distance=1 - float(score),
                )
            )
        return results

    def delete_by_document_id(self, document_id: str) -> None:
        ids_to_remove = [
            int(k)
            for k, v in self._metadata.items()
            if v.get("document_id") == document_id
        ]
        if ids_to_remove:
            id_array = np.array(ids_to_remove, dtype="int64")
            self.index.remove_ids(id_array)
            for idx in ids_to_remove:
                self._metadata.pop(str(idx), None)
            self._save_metadata()
            self._persist_index()
        self._track_remove(document_id)
