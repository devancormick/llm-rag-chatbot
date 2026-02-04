import uuid
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from ingestion.pipeline import IngestionPipeline
from vector_store import create_vector_store
from rag.chain import RAGChain
from leads.store import LeadStore

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
# Silence ChromaDB telemetry errors (posthog API mismatch in some versions)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

app = FastAPI(
    title="LLM RAG Chatbot",
    description="Chatbot powered by LLM with RAG and lead generation",
    version="1.0.0",
)

config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
config.DATA_DIR.mkdir(parents=True, exist_ok=True)
config.LEADS_DIR.mkdir(parents=True, exist_ok=True)

vector_store = create_vector_store()
ingestion_pipeline = IngestionPipeline()
rag_chain = RAGChain(vector_store)
lead_store = LeadStore()

static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list


class LeadRequest(BaseModel):
    email: str
    name: Optional[str] = None
    company: Optional[str] = None


@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    index_path = static_path / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "LLM RAG Chatbot API", "docs": "/docs"}


@app.get("/config")
async def get_config(request: Request):
    """Frontend config: base URL for API calls (from env). Empty = use same origin."""
    base_url = config.BASE_URL or str(request.base_url).rstrip("/")
    return {"baseUrl": base_url}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "vector_provider": config.VECTOR_PROVIDER,
    }


@app.get("/vector/health")
async def vector_health():
    try:
        doc_count = len(vector_store.list_documents())
        return {
            "status": "ok",
            "provider": config.VECTOR_PROVIDER,
            "tracked_documents": doc_count,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = rag_chain.query(request.question, top_k=request.top_k)
    return QueryResponse(answer=result.answer, sources=result.sources)


@app.post("/chat/stream")
async def chat_stream(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    def generate():
        for chunk in rag_chain.query_stream(request.question, top_k=request.top_k):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/leads")
async def register_lead(request: LeadRequest):
    if not request.email.strip():
        raise HTTPException(status_code=400, detail="Email is required")

    lead_id = lead_store.add(
        email=request.email,
        name=request.name,
        company=request.company,
    )
    return {"lead_id": lead_id, "status": "registered"}


@app.get("/leads")
async def list_leads():
    leads = lead_store.get_all()
    return {"leads": leads}


@app.get("/leads/export")
async def export_leads():
    csv = lead_store.export_csv()
    from fastapi.responses import Response
    return Response(content=csv, media_type="text/csv")


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".md", ".markdown"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported format. Use PDF or Markdown.",
        )

    _MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    doc_id = str(uuid.uuid4())
    upload_path = config.UPLOAD_DIR / f"{doc_id}{suffix}"
    upload_path.write_bytes(content)

    try:
        chunks = ingestion_pipeline.process_file(upload_path)
        count = vector_store.add_chunks(chunks, document_id=doc_id)
        return {
            "document_id": doc_id,
            "filename": file.filename,
            "chunks_indexed": count,
        }
    except Exception as e:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    doc_ids = vector_store.list_documents()
    return {"documents": doc_ids}


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    doc_ids = vector_store.list_documents()
    if document_id not in doc_ids:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        vector_store.delete_by_document_id(document_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    for path in config.UPLOAD_DIR.glob(f"{document_id}.*"):
        path.unlink(missing_ok=True)
    return {"status": "deleted", "document_id": document_id}
