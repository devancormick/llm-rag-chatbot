from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings

import config
from ingestion.chunker import TextChunk
from vector_store.base import SearchResult, VectorStore
from vector_store.embeddings import get_embedding_function


class ChromaVectorStore(VectorStore):
    """Vector store backed by ChromaDB."""

    COLLECTION_NAME = "chatbot_docs"

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        tracker=None,
    ):
        super().__init__(tracker=tracker)
        self.persist_dir = persist_directory or config.CHROMA_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._embed_fn = get_embedding_function()

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        return self._embed_fn(texts)

    def add_chunks(
        self,
        chunks: List[TextChunk],
        document_id: Optional[str] = None,
    ) -> int:
        if not chunks:
            return 0

        contents = [c.content for c in chunks]
        metadatas = [
            {**c.metadata, "document_id": document_id or "unknown"}
            for c in chunks
        ]
        embeddings = self._embed(contents)
        ids = [f"chunk_{i}_{hash(c.content) % 10**8}" for i, c in enumerate(chunks)]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
        )

        if document_id:
            self._track_add(document_id)
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = config.TOP_K_CHUNKS,
        similarity_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        query_embedding = self._embed([query])[0]
        fetch_k = max(top_k * 2, 20) if similarity_threshold else top_k
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
        )
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else config.SIMILARITY_THRESHOLD
        )

        search_results: List[SearchResult] = []
        if results["documents"] and results["documents"][0]:
            distances = results.get("distances")
            dist_list = (
                distances[0]
                if distances
                else [0.0] * len(results["documents"][0])
            )
            # Build (doc, meta, dist) and sort by distance (closest first)
            candidates = list(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    dist_list,
                )
            )
            for doc, meta, dist in candidates:
                similarity = 1 - (dist / 2.0) if dist <= 2 else 0
                if similarity >= threshold:
                    search_results.append(
                        SearchResult(content=doc, metadata=meta, distance=dist)
                    )
                    if len(search_results) >= top_k:
                        break
            # If threshold filtered everything out, still return top_k closest
            # so the LLM gets context and can answer (or say "not in context")
            if not search_results and candidates:
                for doc, meta, dist in candidates[:top_k]:
                    search_results.append(
                        SearchResult(content=doc, metadata=meta, distance=dist)
                    )
        return search_results

    def delete_by_document_id(self, document_id: str) -> None:
        results = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
        self._track_remove(document_id)
