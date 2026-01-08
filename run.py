#!/usr/bin/env python3
"""Run the LLM RAG Chatbot API."""

import config
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
    )
