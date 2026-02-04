"""Vector store backed by Weaviate (free self-hosted, hybrid search)."""

from __future__ import annotations

import uuid
from typing import List, Optional

import config
from ingestion.chunker import TextChunk
from vector_store.base import SearchResult, VectorStore
from vector_store.embeddings import get_embedding_function


class WeaviateVectorStore(VectorStore):
    """Vector store using Weaviate. Free self-hosted, vector + keyword + filters."""

    COLLECTION_NAME = "LlmRagChunk"

    def __init__(
        self,
        url: str | None = None,
        grpc_port: int | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
        dimension: int | None = None,
        tracker=None,
    ):
        super().__init__(tracker=tracker)
        self.url = url or getattr(config, "WEAVIATE_URL", "http://localhost:8080")
        self.grpc_port = grpc_port or getattr(config, "WEAVIATE_GRPC_PORT", 50051)
        self.api_key = api_key or getattr(config, "WEAVIATE_API_KEY", None)
        self.collection_name = collection_name or getattr(
            config, "WEAVIATE_COLLECTION_NAME", self.COLLECTION_NAME
        )
        self.dimension = dimension or getattr(config, "WEAVIATE_DIMENSION", 384)

        self.embedding_fn = get_embedding_function()
        try:
            import weaviate
            from weaviate.classes.config import Configure
        except ImportError as exc:
            raise ImportError(
                "weaviate-client is required for the Weaviate vector provider."
            ) from exc

        self.weaviate = weaviate
        self.Configure = Configure
        self.client = self._connect()
        self._ensure_collection()

    def _connect(self):
        if self.api_key:
            return self.weaviate.connect_to_weaviate_cloud(
                cluster_url=self.url.replace("https://", "").replace("http://", ""),
                auth_credentials=self.weaviate.auth.AuthApiKey(self.api_key),
                headers={"X-Weaviate-Api-Key": self.api_key},
            )
        parsed = urlparse(self.url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 8080)
        secure = parsed.scheme == "https"
        return self.weaviate.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=secure,
            grpc_port=self.grpc_port,
        )

    def _ensure_collection(self) -> None:
        if self.client.collections.exists(self.collection_name):
            return
        Prop = self.wvc_config.Property
        DataType = self.wvc_config.DataType
        self.client.collections.create(
            name=self.collection_name,
            vectorizer_config=self.wvc_config.Configure.Vectors.self_provided(),
            properties=[
                Prop(name="content", data_type=DataType.TEXT),
                Prop(name="document_id", data_type=DataType.TEXT),
                Prop(name="source", data_type=DataType.TEXT),
                Prop(name="page", data_type=DataType.INT),
            ],
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

        doc_id = document_id or "unknown"
        embeddings = self._embed([c.content for c in chunks])
        collection = self.client.collections.get(self.collection_name)

        with collection.batch.dynamic() as batch:
            for chunk, embedding in zip(chunks, embeddings):
                batch.add_object(
                    properties={
                        "content": chunk.content,
                        "document_id": doc_id,
                        "source": chunk.metadata.get("source", ""),
                        "page": chunk.metadata.get("page") or 0,
                    },
                    vector=embedding,
                )

        if document_id:
            self._track_add(document_id)
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int,
        similarity_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else getattr(config, "SIMILARITY_THRESHOLD", 0.7)
        )
        embedding = self._embed([query])[0]
        collection = self.client.collections.get(self.collection_name)

        response = collection.query.near_vector(
            near_vector=embedding,
            limit=top_k,
            return_metadata=self.weaviate.classes.query.MetadataQuery(distance=True),
        )

        results: List[SearchResult] = []
        for obj in response.objects:
            dist = obj.metadata.distance if obj.metadata else 1.0
            similarity = 1.0 - float(dist) if dist is not None else 0
            if similarity < threshold:
                continue
            props = obj.properties or {}
            results.append(
                SearchResult(
                    content=props.get("content", ""),
                    metadata={
                        "source": props.get("source"),
                        "page": props.get("page"),
                        "document_id": props.get("document_id"),
                    },
                    distance=float(dist) if dist is not None else 1.0,
                )
            )
        return results

    def delete_by_document_id(self, document_id: str) -> None:
        from weaviate.classes.query import Filter
        collection = self.client.collections.get(self.collection_name)
        objs = list(
            collection.query.fetch_objects(
                filters=Filter.by_property("document_id").equal(document_id),
                limit=10000,
            )
        )
        for obj in objs:
            collection.data.delete_by_id(obj.uuid)
        self._track_remove(document_id)
