#!/usr/bin/env python3
"""Run the LLM RAG Chatbot API."""

import os
import warnings
import config  # noqa: E402 - env must be set before config
import uvicorn  # noqa: E402

# Disable ChromaDB telemetry to avoid posthog errors (e.g. capture() argument mismatch)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Suppress noisy dependency warnings
warnings.filterwarnings("ignore", message=".*ARC4.*", category=DeprecationWarning, module=".*cryptography.*")
warnings.filterwarnings("ignore", message=".*OpenSSL.*", category=UserWarning, module="urllib3")

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
    )
