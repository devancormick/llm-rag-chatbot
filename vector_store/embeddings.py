"""Shared embedding utilities for vector stores."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, List

import config

EmbeddingFn = Callable[[List[str]], List[List[float]]]


@lru_cache(maxsize=1)
def get_embedding_function() -> EmbeddingFn:
    """Return a cached embedding function based on configuration."""
    if config.EMBEDDING_PROVIDER == "openai" and config.OPENAI_API_KEY:
        from openai import OpenAI

        client = OpenAI(api_key=config.OPENAI_API_KEY)
        model = config.OPENAI_EMBEDDING_MODEL

        def embed(texts: List[str]) -> List[List[float]]:
            resp = client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in resp.data]

        return embed

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.EMBEDDING_MODEL)

    def embed(texts: List[str]) -> List[List[float]]:
        return model.encode(texts).tolist()

    return embed
