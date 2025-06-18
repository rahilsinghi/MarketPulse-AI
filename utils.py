"""
Utility helpers shared across the project.
"""

import os
from typing import List, Dict, Any

import openai
from dotenv import load_dotenv

# ─── ENV & OPENAI ──────────────────────────────────────────────────────────────
load_dotenv()                                    # read .env if present
openai.api_key = os.getenv("OPENAI_API_KEY")     # must be set before use

# ─── Embedding helper (OpenAI) ─────────────────────────────────────────────────
def embed_text(text: str) -> List[float]:
    """
    Call OpenAI text-embedding-3-small and return the 1536-D vector.
    A tiny wrapper so we can swap models later if needed.
    """
    resp = openai.Embedding.create(
        model="text-embedding-3-small",
        input=[text],
    )
    return resp["data"][0]["embedding"]


# ─── Pathway table accessor ────────────────────────────────────────────────────
def get_pathway_table() -> List[Dict[str, Any]]:
    """
    Returns the *latest* rows from the Pathway news table as a list[dict].
    Delegates to ingest_pipeline.get_latest_rows() so no module cycles.
    """
    from ingest_pipeline import get_latest_rows  # late import to avoid circular deps
    return get_latest_rows()
