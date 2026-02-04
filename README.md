# LLM RAG Chatbot

A chatbot powered by a Large Language Model (LLM) using Retrieval-Augmented Generation (RAG) techniques. Features lead generation, document ingestion, and a web-based chat interface.

## Features

- **RAG Chatbot**: Retrieve and generate answers from uploaded documents
- **Lead Generation**: Capture visitor email, name, and company before chat
- **Document Ingestion**: Upload PDF and Markdown files
- **Streaming Responses**: Real-time answer streaming (API)
- **NLP**: Text chunking, embeddings, semantic search via pluggable vector stores (Chroma, Pinecone, Qdrant, Milvus, FAISS, pgvector, Weaviate)
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
| BASE_URL | *(empty)* | Public base URL for frontend API calls. Empty = same origin (e.g. `http://localhost:8000`). Set e.g. `https://yourname.github.io/llm-rag-chatbot` when hosting static on GitHub Pages. Or use `?baseUrl=...` or `?api=...` in the page URL. |

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

**pgvector (PostgreSQL)** – free, self-hosted; good for existing Postgres apps
- `PGVECTOR_CONNECTION_STRING` or `DATABASE_URL` *(required)*
- `PGVECTOR_TABLE_NAME`, `PGVECTOR_DIMENSION`, `PGVECTOR_CREATE_EXTENSION`

**Weaviate** – free self-hosted; vector + keyword + filters
- `WEAVIATE_URL`, `WEAVIATE_GRPC_PORT`, `WEAVIATE_API_KEY`
- `WEAVIATE_COLLECTION_NAME`, `WEAVIATE_DIMENSION`

**FAISS**
- `FAISS_INDEX_PATH`, `FAISS_METADATA_PATH`
- `FAISS_DIMENSION`, `FAISS_NORMALIZE`

Switch providers by updating `.env` and restarting `run.py`. Use `/vector/health` to verify connectivity.

## Deployment

### GitHub Actions

- **CI** (`.github/workflows/ci.yml`): Runs on every push and PR to `main`. Tests on Python 3.10–3.12, Ruff lint, smoke test (app + `/config` with `BASE_URL` = [GitHub Pages URL](https://devancormick.github.io/llm-rag-chatbot)), and **smoke-ollama** (installs Ollama, pulls `tinyllama`, starts app, tests `/chat`).
- **Deploy** (`.github/workflows/deploy.yml`): On push to `main`, builds a Docker image and pushes it to [GitHub Container Registry](https://github.com/devancormick/llm-rag-chatbot/pkgs/container/llm-rag-chatbot). If `RENDER_DEPLOY_HOOK` is set in repo **Settings → Secrets and variables → Actions**, the workflow triggers a Render deploy.
- **GitHub Pages** (`.github/workflows/pages.yml`): Deploys the static UI to [https://devancormick.github.io/llm-rag-chatbot](https://devancormick.github.io/llm-rag-chatbot). To use a backend (e.g. Render), open the site with `?baseUrl=YOUR_API_URL` (e.g. `?baseUrl=https://llm-rag-chatbot.onrender.com`).

### Docker

```bash
docker build -t llm-rag-chatbot .
docker run -p 8000:8000 -e VECTOR_PROVIDER=chroma llm-rag-chatbot
```

For production, set env vars (e.g. `OLLAMA_BASE_URL` if using a remote LLM, or `PINECONE_API_KEY` for Pinecone).

### Render (native Python, no Docker)

1. Connect [Render](https://render.com) to this repo.
2. Use **Blueprint** and add `render.yaml` (native Python: `runtime: python`, `buildCommand`, `startCommand`). Or create a **Web Service** → **Python** and set build to `pip install -r requirements.txt`, start to `uvicorn api.main:app --host 0.0.0.0 --port $PORT`.
3. Set env vars in the Render dashboard (e.g. `VECTOR_PROVIDER`, `OLLAMA_BASE_URL` for a hosted Ollama).
4. Optional: In GitHub **Settings → Secrets**, add `RENDER_DEPLOY_HOOK` with your Render service’s deploy hook URL so every push to `main` triggers a deploy.

### Other platforms

- **Railway / Fly.io / Cloud Run**: Use the same Dockerfile; point the service at this repo or at the image `ghcr.io/devancormick/llm-rag-chatbot:latest` after the deploy workflow has run.

## Tech Stack

- Python, FastAPI, Chroma/Qdrant/Pinecone/Milvus/FAISS/pgvector/Weaviate, sentence-transformers, Ollama
- Vanilla JavaScript, HTML5, CSS3

## License

Proprietary - Private. All rights reserved.
