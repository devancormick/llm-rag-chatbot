from __future__ import annotations

import logging

import config
from vector_store.base import VectorStore
from vector_store.document_tracker import DocumentTracker
from vector_store.providers import (
    ChromaVectorStore,
    FaissVectorStore,
    MilvusVectorStore,
    PineconeVectorStore,
    QdrantVectorStore,
)

logger = logging.getLogger(__name__)


def create_vector_store() -> VectorStore:
    """Instantiate the configured vector store provider."""
    provider = (config.VECTOR_PROVIDER or "chroma").lower()
    tracker = DocumentTracker()

    if provider == "pinecone":
        if not config.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is required for Pinecone provider.")
        logger.info("Using Pinecone vector store")
        return PineconeVectorStore(tracker=tracker)

    if provider == "qdrant":
        logger.info("Using Qdrant vector store at %s", config.QDRANT_URL)
        return QdrantVectorStore(tracker=tracker)

    if provider == "milvus":
        logger.info("Using Milvus vector store at %s", config.MILVUS_URI)
        return MilvusVectorStore(tracker=tracker)

    if provider == "faiss":
        logger.info("Using FAISS vector store (local)")
        return FaissVectorStore(tracker=tracker)

    # Default to Chroma
    logger.info("Using Chroma vector store (local)")
    return ChromaVectorStore(tracker=tracker)
