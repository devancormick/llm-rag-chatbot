"""Vector store backed by PostgreSQL with pgvector extension (free, self-hosted)."""

from __future__ import annotations

import json
import uuid
from typing import List, Optional

import config
from ingestion.chunker import TextChunk
from vector_store.base import SearchResult, VectorStore
from vector_store.embeddings import get_embedding_function


class PgvectorVectorStore(VectorStore):
    """Vector store using PostgreSQL + pgvector. Free, self-hosted."""

    TABLE_NAME = "llm_rag_embeddings"

    def __init__(
        self,
        connection_string: str | None = None,
        table_name: str | None = None,
        dimension: int | None = None,
        create_extension: bool | None = None,
        tracker=None,
    ):
        super().__init__(tracker=tracker)
        self.connection_string = (
            connection_string
            or getattr(config, "PGVECTOR_CONNECTION_STRING", "")
            or getattr(config, "DATABASE_URL", "")
        )
        if not self.connection_string:
            raise ValueError(
                "PGVECTOR_CONNECTION_STRING or DATABASE_URL is required for pgvector."
            )
        self.table_name = table_name or getattr(
            config, "PGVECTOR_TABLE_NAME", self.TABLE_NAME
        )
        self.dimension = dimension or getattr(
            config, "PGVECTOR_DIMENSION", 384
        )
        self.create_extension = (
            create_extension
            if create_extension is not None
            else getattr(config, "PGVECTOR_CREATE_EXTENSION", True)
        )

        self.embedding_fn = get_embedding_function()
        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector
        except ImportError as exc:
            raise ImportError(
                "psycopg2-binary and pgvector are required for the pgvector provider."
            ) from exc

        self.psycopg2 = psycopg2
        self.register_vector = register_vector
        self._ensure_schema()

    def _conn(self):
        conn = self.psycopg2.connect(self.connection_string)
        self.register_vector(conn)
        return conn

    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                if self.create_extension:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        document_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}',
                        embedding vector({self.dimension})
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
                    ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                    """
                )
            conn.commit()

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

        with self._conn() as conn:
            with conn.cursor() as cur:
                for chunk, embedding in zip(chunks, embeddings):
                    meta = json.dumps({**chunk.metadata, "content": chunk.content})
                    cur.execute(
                        f"""
                        INSERT INTO {self.table_name}
                        (document_id, content, metadata, embedding)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (doc_id, chunk.content, meta, embedding),
                    )
            conn.commit()

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
        # pgvector cosine distance <=> returns 0 = identical, 2 = opposite
        # similarity = 1 - (distance/2) for cosine, or we use 1 - distance when normalized
        # vector_cosine_ops: smaller distance = more similar. similarity ≈ 1 - distance.
        embedding = self._embed([query])[0]

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT content, metadata, embedding <=> %s AS dist
                    FROM {self.table_name}
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (embedding, embedding, top_k),
                )
                rows = cur.fetchall()

        results: List[SearchResult] = []
        for content, metadata, dist in rows:
            dist_f = float(dist) if dist is not None else 2.0
            similarity = 1.0 - (dist_f / 2.0) if dist_f <= 2 else 0
            if similarity < threshold:
                continue
            meta = metadata if isinstance(metadata, dict) else json.loads(metadata or "{}")
            results.append(
                SearchResult(content=content, metadata=meta, distance=float(dist))
            )
        return results

    def delete_by_document_id(self, document_id: str) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self.table_name} WHERE document_id = %s",
                    (document_id,),
                )
            conn.commit()
        self._track_remove(document_id)

    def list_documents(self) -> List[str]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT DISTINCT document_id FROM {self.table_name} ORDER BY document_id"
                )
                return [row[0] for row in cur.fetchall()]
