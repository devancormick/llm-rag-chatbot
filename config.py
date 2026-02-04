import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(".env.local")  # local overrides for local run (gitignored)
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent

# All uploads, DBs, and temp files go under DATA_DIR (default: project/data/) so the project root stays clean.
_data_dir = os.getenv("DATA_DIR", "").strip()
DATA_DIR = Path(_data_dir) if _data_dir else (BASE_DIR / "data")
_upload_dir = os.getenv("UPLOAD_DIR", "").strip()
UPLOAD_DIR = Path(_upload_dir) if _upload_dir else (DATA_DIR / "uploads")
_chroma_dir = os.getenv("CHROMA_DIR", "").strip()
CHROMA_DIR = Path(_chroma_dir) if _chroma_dir else (DATA_DIR / "chroma_db")
_leads_dir = os.getenv("LEADS_DIR", "").strip()
LEADS_DIR = Path(_leads_dir) if _leads_dir else (DATA_DIR / "leads")
TEMP_DIR = DATA_DIR / "tmp"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "8"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# When set, use Ollama Cloud (https://ollama.com) instead of local; reduces local resource usage
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_CLOUD_URL = "https://ollama.com"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

VECTOR_PROVIDER = os.getenv("VECTOR_PROVIDER", "chroma").lower()

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "llm-rag")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")
PINECONE_DIMENSION = int(
    os.getenv("PINECONE_DIMENSION", str(EMBEDDING_DIMENSION))
)
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "llm_rag_docs")
QDRANT_DIMENSION = int(os.getenv("QDRANT_DIMENSION", str(EMBEDDING_DIMENSION)))
QDRANT_GRPC = os.getenv("QDRANT_GRPC", "false").lower() == "true"

# pgvector (PostgreSQL) - free, self-hosted
DATABASE_URL = os.getenv("DATABASE_URL", "")
PGVECTOR_CONNECTION_STRING = os.getenv(
    "PGVECTOR_CONNECTION_STRING", DATABASE_URL
)
PGVECTOR_TABLE_NAME = os.getenv("PGVECTOR_TABLE_NAME", "llm_rag_embeddings")
PGVECTOR_DIMENSION = int(
    os.getenv("PGVECTOR_DIMENSION", str(EMBEDDING_DIMENSION))
)
PGVECTOR_CREATE_EXTENSION = (
    os.getenv("PGVECTOR_CREATE_EXTENSION", "true").lower() == "true"
)

# Weaviate - free self-hosted, hybrid search
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
WEAVIATE_COLLECTION_NAME = os.getenv(
    "WEAVIATE_COLLECTION_NAME", "LlmRagChunk"
)
WEAVIATE_DIMENSION = int(
    os.getenv("WEAVIATE_DIMENSION", str(EMBEDDING_DIMENSION))
)

# FAISS
FAISS_INDEX_PATH = os.getenv(
    "FAISS_INDEX_PATH", str(DATA_DIR / "faiss" / "index.faiss")
)
FAISS_METADATA_PATH = os.getenv(
    "FAISS_METADATA_PATH", str(DATA_DIR / "faiss" / "metadata.json")
)
FAISS_DIMENSION = int(os.getenv("FAISS_DIMENSION", str(EMBEDDING_DIMENSION)))
FAISS_NORMALIZE = os.getenv("FAISS_NORMALIZE", "true").lower() == "true"


def _json_env(var_name: str, default: dict) -> dict:
    raw = os.getenv(var_name)
    if not raw:
        return default
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    return default


# Milvus / Zilliz
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_USER = os.getenv("MILVUS_USER", "root")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "Milvus")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "llm_rag_docs")
MILVUS_DIMENSION = int(os.getenv("MILVUS_DIMENSION", str(EMBEDDING_DIMENSION)))
MILVUS_INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "IVF_FLAT")
MILVUS_INDEX_PARAMS = _json_env("MILVUS_INDEX_PARAMS", {"nlist": 1024})
MILVUS_SEARCH_PARAMS = _json_env("MILVUS_SEARCH_PARAMS", {"nprobe": 16})

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Public base URL for the frontend (API + static). Empty = same origin (e.g. localhost).
# Set to e.g. https://yourname.github.io/llm-rag-chatbot when hosting static on GitHub Pages.
BASE_URL = (os.getenv("BASE_URL") or "").rstrip("/")
