import unittest
from unittest import mock

import config
from vector_store import create_vector_store


class VectorFactoryTests(unittest.TestCase):
    def test_default_chroma_provider(self):
        with mock.patch.object(config, "VECTOR_PROVIDER", "chroma"), mock.patch(
            "vector_store.factory.ChromaVectorStore"
        ) as chroma_cls, mock.patch(
            "vector_store.factory.DocumentTracker"
        ) as tracker_cls:
            create_vector_store()
            tracker_cls.assert_called_once()
            chroma_cls.assert_called_once_with(tracker=mock.ANY)

    def test_pinecone_requires_api_key(self):
        with mock.patch.object(config, "VECTOR_PROVIDER", "pinecone"), mock.patch.object(
            config, "PINECONE_API_KEY", ""
        ):
            with self.assertRaises(ValueError):
                create_vector_store()

    def test_pinecone_creation(self):
        with mock.patch.object(config, "VECTOR_PROVIDER", "pinecone"), mock.patch.object(
            config, "PINECONE_API_KEY", "test-key"
        ), mock.patch(
            "vector_store.factory.PineconeVectorStore"
        ) as pinecone_cls, mock.patch(
            "vector_store.factory.DocumentTracker"
        ) as tracker_cls:
            create_vector_store()
            tracker_cls.assert_called_once()
            pinecone_cls.assert_called_once_with(tracker=mock.ANY)

    def test_qdrant_creation(self):
        with mock.patch.object(config, "VECTOR_PROVIDER", "qdrant"), mock.patch(
            "vector_store.factory.QdrantVectorStore"
        ) as qdrant_cls, mock.patch(
            "vector_store.factory.DocumentTracker"
        ) as tracker_cls:
            create_vector_store()
            tracker_cls.assert_called_once()
            qdrant_cls.assert_called_once_with(tracker=mock.ANY)

    def test_milvus_creation(self):
        with mock.patch.object(config, "VECTOR_PROVIDER", "milvus"), mock.patch(
            "vector_store.factory.MilvusVectorStore"
        ) as milvus_cls, mock.patch(
            "vector_store.factory.DocumentTracker"
        ) as tracker_cls:
            create_vector_store()
            tracker_cls.assert_called_once()
            milvus_cls.assert_called_once_with(tracker=mock.ANY)

    def test_faiss_creation(self):
        with mock.patch.object(config, "VECTOR_PROVIDER", "faiss"), mock.patch(
            "vector_store.factory.FaissVectorStore"
        ) as faiss_cls, mock.patch(
            "vector_store.factory.DocumentTracker"
        ) as tracker_cls:
            create_vector_store()
            tracker_cls.assert_called_once()
            faiss_cls.assert_called_once_with(tracker=mock.ANY)


if __name__ == "__main__":
    unittest.main()
