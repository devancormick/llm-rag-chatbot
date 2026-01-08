from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

import config
from ingestion.chunker import TextChunk


@dataclass
class SearchResult:
    content: str
    metadata: dict
    distance: float


def _get_embedding_function():
    if config.EMBEDDING_PROVIDER == "openai" and config.OPENAI_API_KEY:
        from openai import OpenAI

        client = OpenAI(api_key=config.OPENAI_API_KEY)
        model = config.OPENAI_EMBEDDING_MODEL

        def embed(texts: List[str]) -> List[List[float]]:
            resp = client.embeddings.create(input=texts, model=model)
            return [e.embedding for e in resp.data]

        return embed
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(config.EMBEDDING_MODEL)

        def embed(texts: List[str]) -> List[List[float]]:
            return model.encode(texts).tolist()

        return embed


class VectorStore:
    """Vector store for document embeddings using ChromaDB."""

    COLLECTION_NAME = "chatbot_docs"

    def __init__(self, persist_directory: Optional[Path] = None):
        self.persist_dir = persist_directory or config.CHROMA_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._embed_fn = _get_embedding_function()

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
            {**c.metadata, "document_id": document_id or "unknown"} for c in chunks
        ]
        embeddings = self._embed(contents)
        ids = [f"chunk_{i}_{hash(c.content) % 10**8}" for i, c in enumerate(chunks)]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
        )
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

        search_results = []
        if results["documents"] and results["documents"][0]:
            distances = results.get("distances")
            dist_list = distances[0] if distances else [0.0] * len(results["documents"][0])
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                dist_list,
            ):
                similarity = 1 - (dist / 2.0) if dist <= 2 else 0
                if similarity >= threshold:
                    search_results.append(
                        SearchResult(content=doc, metadata=meta, distance=dist)
                    )
                    if len(search_results) >= top_k:
                        break
        return search_results

    def delete_by_document_id(self, document_id: str) -> None:
        results = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )
        if results["ids"]:
            self.collection.delete(ids=results["ids"])

    def list_documents(self) -> List[str]:
        results = self.collection.get(include=["metadatas"])
        doc_ids = set()
        for meta in results.get("metadatas", []):
            if meta and "document_id" in meta:
                doc_ids.add(meta["document_id"])
        return sorted(doc_ids)
