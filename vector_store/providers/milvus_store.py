from __future__ import annotations

import uuid
from typing import List, Optional

import config
from ingestion.chunker import TextChunk
from vector_store.base import SearchResult, VectorStore
from vector_store.embeddings import get_embedding_function


class MilvusVectorStore(VectorStore):
    """Vector store backed by Milvus or Zilliz Cloud."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        collection_name: str | None = None,
        dimension: int | None = None,
        index_type: str | None = None,
        index_params: dict | None = None,
        search_params: dict | None = None,
        tracker=None,
    ):
        super().__init__(tracker=tracker)
        self.uri = uri or getattr(config, "MILVUS_URI", "http://localhost:19530")
        self.user = user or getattr(config, "MILVUS_USER", "root")
        self.password = password or getattr(config, "MILVUS_PASSWORD", "Milvus")
        self.collection_name = collection_name or getattr(
            config, "MILVUS_COLLECTION_NAME", "llm_rag_docs"
        )
        self.dimension = dimension or getattr(config, "MILVUS_DIMENSION", 384)
        self.index_type = index_type or getattr(
            config, "MILVUS_INDEX_TYPE", "IVF_FLAT"
        )
        self.index_params = index_params or getattr(
            config, "MILVUS_INDEX_PARAMS", {"nlist": 1024}
        )
        self.search_params = search_params or getattr(
            config, "MILVUS_SEARCH_PARAMS", {"nprobe": 16}
        )

        try:
            from pymilvus import (
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                connections,
                utility,
            )
        except ImportError as exc:
            raise ImportError(
                "pymilvus is required for the Milvus vector provider."
            ) from exc

        self.Collection = Collection
        self.CollectionSchema = CollectionSchema
        self.DataType = DataType
        self.FieldSchema = FieldSchema
        self.connections = connections
        self.utility = utility

        self.connections.connect(
            alias="default",
            uri=self.uri,
            user=self.user,
            password=self.password,
        )

        if not self.utility.has_collection(self.collection_name):
            self._create_collection()

        self.collection = self.Collection(name=self.collection_name)
        self.collection.load()
        self.embedding_fn = get_embedding_function()

    def _create_collection(self) -> None:
        fields = [
            self.FieldSchema(
                name="id",
                dtype=self.DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=128,
            ),
            self.FieldSchema(
                name="document_id",
                dtype=self.DataType.VARCHAR,
                max_length=128,
            ),
            self.FieldSchema(
                name="metadata",
                dtype=self.DataType.JSON,
            ),
            self.FieldSchema(
                name="embedding",
                dtype=self.DataType.FLOAT_VECTOR,
                dim=self.dimension,
            ),
        ]
        schema = self.CollectionSchema(fields, description="LLM RAG embeddings")
        collection = self.Collection(
            name=self.collection_name,
            schema=schema,
            using="default",
        )
        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": self.index_type,
                "metric_type": "COSINE",
                "params": self.index_params,
            },
        )
        collection.flush()

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
        ids = []
        doc_ids = []
        metadata_list = []
        for chunk, embedding in zip(chunks, embeddings):
            ids.append(f"{document_id or 'doc'}_{uuid.uuid4().hex[:8]}")
            doc_ids.append(document_id or "unknown")
            metadata_list.append({**chunk.metadata, "content": chunk.content})

        self.collection.insert(
            [
                ids,
                doc_ids,
                metadata_list,
                embeddings,
            ]
        )
        self.collection.flush()
        if document_id:
            self._track_add(document_id)
        return len(ids)

    def search(
        self,
        query: str,
        top_k: int,
        similarity_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        embedding = self._embed([query])[0]
        results = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param={
                "metric_type": "COSINE",
                "params": self.search_params,
            },
            limit=top_k,
            output_fields=["metadata"],
        )
        hits = results[0] if results else []
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else getattr(config, "SIMILARITY_THRESHOLD", 0.7)
        )
        search_results: List[SearchResult] = []
        for hit in hits:
            similarity = 1 - hit.distance
            if similarity < threshold:
                continue
            metadata = hit.entity.get("metadata", {}) if hasattr(hit, "entity") else {}
            search_results.append(
                SearchResult(
                    content=metadata.get("content", ""),
                    metadata=metadata,
                    distance=hit.distance,
                )
            )
        return search_results

    def delete_by_document_id(self, document_id: str) -> None:
        escaped = document_id.replace("'", "\\'")
        expr = f"document_id == '{escaped}'"
        self.collection.delete(expr)
        self.collection.flush()
        self._track_remove(document_id)
