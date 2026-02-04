# LLM RAG Chatbot

A chatbot powered by a Large Language Model (LLM) using Retrieval-Augmented Generation (RAG) techniques. Features lead generation, document ingestion, and a web-based chat interface.

## Features

- **RAG Chatbot**: Retrieve and generate answers from uploaded documents
- **Lead Generation**: Capture visitor email, name, and company before chat
- **Document Ingestion**: Upload PDF and Markdown files
- **Streaming Responses**: Real-time answer streaming (API)
- **NLP**: Text chunking, embeddings, semantic search via pluggable vector stores (Chroma, Pinecone, Qdrant, Milvus, FAISS)
- **Web UI**: Responsive chat interface with lead modal

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running

## Installation

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and adjust if needed.

## Usage

1. Start Ollama:
   ```bash
   ollama serve
   ollama pull llama3:8b
   ```

2. Run the chatbot:
   ```bash
   python run.py
   ```

3. Open http://localhost:8000 in your browser.

4. Enter your details (lead capture) to start chatting.

5. Optional: Upload documents to expand knowledge:
   ```bash
   curl -X POST "http://localhost:8000/documents/upload" -F "file=@docs/guide.pdf"
   ```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Chat UI |
| GET | /health | Health check |
| GET | /vector/health | Vector store connectivity diagnostics |
| POST | /chat | Send message, get RAG response |
| POST | /chat/stream | Stream response |
| POST | /leads | Register lead |
| GET | /leads | List leads |
| GET | /leads/export | Export leads as CSV |
| POST | /documents/upload | Upload PDF or Markdown |
| GET | /documents | List documents |
| DELETE | /documents/{id} | Delete document |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| VECTOR_PROVIDER | chroma | Choose `chroma`, `pinecone`, `qdrant`, `milvus`, or `faiss` |
| OLLAMA_MODEL | llama3:8b | Ollama model name |
| OLLAMA_BASE_URL | http://localhost:11434 | Ollama API URL |
| EMBEDDING_PROVIDER | sentence_transformers | `sentence_transformers` or `openai` |
| EMBEDDING_MODEL | all-MiniLM-L6-v2 | SentenceTransformer model name |
| EMBEDDING_DIMENSION | 384 | Embedding vector size |
| CHUNK_SIZE | 1000 | Chunk size for documents |
| TOP_K_CHUNKS | 5 | Retrieval count |
| API_PORT | 8000 | Server port |

### Provider-specific settings

**Chroma (default)**
- `CHROMA_DIR`: override persistence directory (defaults to `data/chroma_db`)

**Pinecone**
- `PINECONE_API_KEY` *(required)*
- `PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE`, `PINECONE_DIMENSION`, `PINECONE_METRIC`
- `PINECONE_CLOUD` (aws/gcp) and `PINECONE_REGION`

**Qdrant**
- `QDRANT_URL`, `QDRANT_API_KEY`
- `QDRANT_COLLECTION_NAME`, `QDRANT_DIMENSION`, `QDRANT_GRPC`

**Milvus / Zilliz**
- `MILVUS_URI`, `MILVUS_USER`, `MILVUS_PASSWORD`
- `MILVUS_COLLECTION_NAME`, `MILVUS_DIMENSION`
- `MILVUS_INDEX_TYPE`, `MILVUS_INDEX_PARAMS` (JSON), `MILVUS_SEARCH_PARAMS` (JSON)

**FAISS**
- `FAISS_INDEX_PATH`, `FAISS_METADATA_PATH`
- `FAISS_DIMENSION`, `FAISS_NORMALIZE`

Switch providers by updating `.env` and restarting `run.py`. Use `/vector/health` to verify connectivity.

## Tech Stack

- Python, FastAPI, Chroma/Qdrant/Pinecone/Milvus/FAISS, sentence-transformers, Ollama
- Vanilla JavaScript, HTML5, CSS3

## License

Proprietary - Private. All rights reserved.
