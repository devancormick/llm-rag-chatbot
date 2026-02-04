import logging
from typing import List, Iterator
from dataclasses import dataclass

from ollama import Client

import config
from vector_store.base import VectorStore, SearchResult


def _ollama_client() -> Client:
    """Use Ollama Cloud when OLLAMA_API_KEY is set, otherwise local OLLAMA_BASE_URL."""
    if config.OLLAMA_API_KEY:
        return Client(
            host=config.OLLAMA_CLOUD_URL,
            headers={"Authorization": f"Bearer {config.OLLAMA_API_KEY}"},
        )
    return Client(host=config.OLLAMA_BASE_URL)

logger = logging.getLogger(__name__)

RAG_PROMPT = """You are a helpful AI chatbot assistant. Answer the user's question based on the following context. If the context contains relevant information, use it to formulate a helpful answer. If the context does not contain enough information, say so politely and offer to help with related topics. Be conversational and friendly. Keep responses concise but informative.

Context:
{context}

User question: {question}

Your response:"""


@dataclass
class RAGResponse:
    answer: str
    sources: List[dict]
    context_chunks: List[str]


class RAGChain:
    """Retrieval-Augmented Generation chain using local LLM (Ollama)."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.client = _ollama_client()

    def _build_context(self, results: List[SearchResult]) -> str:
        parts = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("source", "unknown")
            page = r.metadata.get("page", "")
            page_str = f" (page {page})" if page else ""
            parts.append(f"[Source {i}: {source}{page_str}]\n{r.content}")
        return "\n\n---\n\n".join(parts)

    def query(
        self,
        question: str,
        top_k: int = config.TOP_K_CHUNKS,
    ) -> RAGResponse:
        results = self.vector_store.search(query=question, top_k=top_k)
        context = self._build_context(results) if results else ""

        if not context:
            return RAGResponse(
                answer="I don't have specific information about that in my knowledge base yet. You can upload documents to expand my knowledge, or ask me something else!",
                sources=[],
                context_chunks=[],
            )

        prompt = RAG_PROMPT.format(context=context, question=question)
        try:
            response = self.client.generate(
                model=config.OLLAMA_MODEL,
                prompt=prompt,
                options={"temperature": 0.3, "num_predict": 1024},
            )
            answer = response["response"].strip()
        except Exception as e:
            status = getattr(e, "status_code", None)
            err_detail = str(e) if e else ""
            logger.error("LLM generation failed: %s (status=%s) %s", type(e).__name__, status, err_detail, exc_info=True)
            # When Ollama is down or cloud unreachable, still show relevant context
            snippet = (results[0].content[:500] + "…") if results and len(results[0].content) > 500 else (results[0].content if results else "")
            if config.OLLAMA_API_KEY:
                if status == 401:
                    hint = "Check **OLLAMA_API_KEY** in .env (get a key at https://ollama.com/settings/keys)."
                elif status == 404:
                    hint = "Use a **cloud model** name in OLLAMA_MODEL (e.g. `ministral-3:8b`, `gemini-3-flash-preview`). See https://ollama.com/search?c=cloud"
                else:
                    hint = "Check **OLLAMA_API_KEY** and network, and that **OLLAMA_MODEL** is a cloud model (https://ollama.com/search?c=cloud)."
                answer = (
                    f"Ollama Cloud error ({status or 'connection'}). {hint}\n\n"
                    "Here's relevant text from your documents:\n\n" + snippet
                )
            else:
                answer = (
                    "Ollama isn't running, so I can't generate a full answer. Start it with: **ollama serve** (and run **ollama pull "
                    + config.OLLAMA_MODEL
                    + "** if needed). Or set **OLLAMA_API_KEY** to use Ollama Cloud.\n\nHere's relevant text from your documents:\n\n"
                    + snippet
                )

        sources = [
            {"source": r.metadata.get("source"), "page": r.metadata.get("page")}
            for r in results
        ]

        return RAGResponse(
            answer=answer,
            sources=sources,
            context_chunks=[r.content for r in results],
        )

    def query_stream(
        self,
        question: str,
        top_k: int = config.TOP_K_CHUNKS,
    ) -> Iterator[str]:
        results = self.vector_store.search(query=question, top_k=top_k)
        context = self._build_context(results) if results else ""

        if not context:
            yield "I don't have specific information about that yet. Try uploading documents or ask something else!"
            return

        prompt = RAG_PROMPT.format(context=context, question=question)
        try:
            stream = self.client.generate(
                model=config.OLLAMA_MODEL,
                prompt=prompt,
                options={"temperature": 0.3, "num_predict": 1024},
                stream=True,
            )
            for chunk in stream:
                if "response" in chunk:
                    yield chunk["response"]
        except Exception as e:
            logger.error("LLM stream failed: %s (detail: %s)", type(e).__name__, getattr(e, "status_code", getattr(e, "message", str(e))), exc_info=True)
            yield "Sorry, I encountered an error."
