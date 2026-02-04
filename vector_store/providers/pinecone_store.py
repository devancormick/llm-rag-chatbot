from __future__ import annotations

import uuid
from typing import List, Optional

import config
from ingestion.chunker import TextChunk
from vector_store.base import SearchResult, VectorStore
from vector_store.embeddings import get_embedding_function


class PineconeVectorStore(VectorStore):
    """Vector store backed by Pinecone (managed service)."""

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
        namespace: str | None = None,
        dimension: int | None = None,
        metric: str | None = None,
        cloud: str | None = None,
        region: str | None = None,
        tracker=None,
    ):
        super().__init__(tracker=tracker)
        self.api_key = api_key or getattr(config, "PINECONE_API_KEY", "")
        if not self.api_key:
            raise ValueError("Pinecone API key is required")

        self.index_name = index_name or getattr(
            config, "PINECONE_INDEX_NAME", "llm-rag"
        )
        self.namespace = namespace or getattr(
            config, "PINECONE_NAMESPACE", "default"
        )
        self.dimension = dimension or getattr(
            config, "PINECONE_DIMENSION", 384
        )
        self.metric = metric or getattr(config, "PINECONE_METRIC", "cosine")
        self.cloud = cloud or getattr(config, "PINECONE_CLOUD", "aws")
        self.region = region or getattr(
            config, "PINECONE_REGION", "us-east-1"
        )

        self.embedding_fn = get_embedding_function()
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:
            raise ImportError(
                "pinecone-client is required for the Pinecone vector provider."
            ) from exc

        self._serverless_spec_cls = ServerlessSpec
        self.client = Pinecone(api_key=self.api_key)
        existing_indexes = {idx["name"] for idx in self.client.list_indexes()}

        if self.index_name not in existing_indexes:
            self.client.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=self._serverless_spec_cls(
                    cloud=self.cloud,
                    region=self.region,
                ),
            )

        self.index = self.client.Index(self.index_name)

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
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{document_id or 'doc'}_{i}_{uuid.uuid4().hex[:8]}"
            vectors.append(
                {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        **chunk.metadata,
                        "document_id": document_id or "unknown",
                        "content": chunk.content,
                    },
                }
            )

        self.index.upsert(vectors=vectors, namespace=self.namespace)

        if document_id:
            self._track_add(document_id)
        return len(vectors)

    def search(
        self,
        query: str,
        top_k: int,
        similarity_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        embedding = self._embed([query])[0]
        response = self.index.query(
            namespace=self.namespace,
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
        )
        matches = response.get("matches") or []
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else getattr(config, "SIMILARITY_THRESHOLD", 0.7)
        )
        results: List[SearchResult] = []
        for match in matches:
            score = match.get("score") or 0.0
            # Pinecone scores are similarities; convert to pseudo-distance.
            distance = 1 - score
            if score < threshold:
                continue
            results.append(
                SearchResult(
                    content=match.get("metadata", {}).get("content", ""),
                    metadata=match.get("metadata", {}),
                    distance=distance,
                )
            )
        return results

    def delete_by_document_id(self, document_id: str) -> None:
        self.index.delete(
            namespace=self.namespace,
            filter={"document_id": {"$eq": document_id}},
        )
        self._track_remove(document_id)
