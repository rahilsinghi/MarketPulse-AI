""" 
Centralised configuration & environment handling.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --------------------------------------------------------------------------- #
# 🔑  API keys – loaded from environment                                      #
# --------------------------------------------------------------------------- #
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
FINNHUB_API_KEY: str | None = os.getenv("FINNHUB_API_KEY")        # optional

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY not found in environment. "
        "Set it in .env or your shell before running."
    )

# --------------------------------------------------------------------------- #
# ↔️  Ingestion mode                                                           #
# --------------------------------------------------------------------------- #
# Set MARKETPULSE_MODE=dummy   → synthetic news generator (default if unset)
#     MARKETPULSE_MODE=live    → real Finnhub feed (requires FINNHUB_API_KEY)
MODE = os.getenv("MARKETPULSE_MODE", "live").lower()
if MODE not in {"dummy", "live"}:
    raise ValueError("MARKETPULSE_MODE must be 'dummy' or 'live'")

# --------------------------------------------------------------------------- #
# 🔧  General constants                                                        #
# --------------------------------------------------------------------------- #
EMBED_MODEL = "text-embedding-ada-002"      # 1536-D embeddings
CHAT_MODEL  = "gpt-4"                       # main LLM
EMBED_CACHE_SIZE = 100                      # LRU size for query embeddings
TOP_K = 3                                   # docs fed to GPT-4