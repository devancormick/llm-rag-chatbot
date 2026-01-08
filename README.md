# LLM RAG Chatbot

A chatbot powered by a Large Language Model (LLM) using Retrieval-Augmented Generation (RAG) techniques. Features lead generation, document ingestion, and a web-based chat interface.

## Features

- **RAG Chatbot**: Retrieve and generate answers from uploaded documents
- **Lead Generation**: Capture visitor email, name, and company before chat
- **Document Ingestion**: Upload PDF and Markdown files
- **Streaming Responses**: Real-time answer streaming (API)
- **NLP**: Text chunking, embeddings, semantic search via ChromaDB
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
| OLLAMA_MODEL | llama3:8b | Ollama model |
| OLLAMA_BASE_URL | http://localhost:11434 | Ollama API URL |
| EMBEDDING_PROVIDER | sentence_transformers | Embedding backend |
| CHUNK_SIZE | 1000 | Chunk size for documents |
| TOP_K_CHUNKS | 5 | Retrieval count |
| API_PORT | 8000 | Server port |

## Tech Stack

- Python, FastAPI, ChromaDB, sentence-transformers, Ollama
- Vanilla JavaScript, HTML5, CSS3

## License

Proprietary - Private. All rights reserved.
