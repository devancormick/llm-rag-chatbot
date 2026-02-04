from __future__ import annotations

import uuid
from typing import List, Optional

import config
from ingestion.chunker import TextChunk
from vector_store.base import SearchResult, VectorStore
from vector_store.embeddings import get_embedding_function


class QdrantVectorStore(VectorStore):
    """Vector store backed by Qdrant."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
        prefer_grpc: bool | None = None,
        dimension: int | None = None,
        tracker=None,
    ):
        super().__init__(tracker=tracker)
        self.url = url or getattr(config, "QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or getattr(config, "QDRANT_API_KEY", None)
        self.collection_name = collection_name or getattr(
            config, "QDRANT_COLLECTION_NAME", "llm_rag_docs"
        )
        self.dimension = dimension or getattr(config, "QDRANT_DIMENSION", 384)
        self.prefer_grpc = (
            prefer_grpc if prefer_grpc is not None else getattr(config, "QDRANT_GRPC", False)
        )

        self.embedding_fn = get_embedding_function()
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qmodels
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is required for the Qdrant vector provider."
            ) from exc

        self.qmodels = qmodels
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            prefer_grpc=self.prefer_grpc,
        )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=self.qmodels.VectorParams(
                    size=self.dimension,
                    distance=self.qmodels.Distance.COSINE,
                ),
            )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_fn(texts)

    def add_chunks(
        self,
        chunks: List[TextChunk],
        document_id: Optional[str] = None,
    ) -> int:
        if not chunks:
            return 0

        embeddings = self._embed([c.content for c in chunks])
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            points.append(
                self.qmodels.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        **chunk.metadata,
                        "content": chunk.content,
                        "document_id": document_id or "unknown",
                    },
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)

        if document_id:
            self._track_add(document_id)
        return len(points)

    def search(
        self,
        query: str,
        top_k: int,
        similarity_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        embedding = self._embed([query])[0]
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=top_k,
        )
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else getattr(config, "SIMILARITY_THRESHOLD", 0.7)
        )
        search_results: List[SearchResult] = []
        for res in results:
            score = res.score or 0.0
            if score < threshold:
                continue
            search_results.append(
                SearchResult(
                    content=(res.payload or {}).get("content", ""),
                    metadata=res.payload or {},
                    distance=1 - score,
                )
            )
        return search_results

    def delete_by_document_id(self, document_id: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=self.qmodels.FilterSelector(
                filter=self.qmodels.Filter(
                    must=[
                        self.qmodels.FieldCondition(
                            key="document_id",
                            match=self.qmodels.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )
        self._track_remove(document_id)
