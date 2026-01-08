import logging
from typing import List, Iterator
from dataclasses import dataclass

from ollama import Client

import config
from vector_store.store import VectorStore, SearchResult

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
        self.client = Client(host=config.OLLAMA_BASE_URL)

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
            logger.error("LLM generation failed: %s", e)
            answer = "Sorry, I encountered an error. Please ensure Ollama is running (ollama serve)."

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
            logger.error("LLM stream failed: %s", e)
            yield "Sorry, I encountered an error."
