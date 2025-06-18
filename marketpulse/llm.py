""" 
LLM-based query answering.

Provides a synchronous API for answering questions about news articles.
Uses OpenAI's GPT-4 to generate answers based on retrieved context.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import openai

from .config import CHAT_MODEL, EMBED_CACHE_SIZE, EMBED_MODEL, OPENAI_API_KEY, TOP_K
from .vector_store import search_vectors

# --------------------------------------------------------------------------- #
# 🔍  Query embedding                                                          #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=EMBED_CACHE_SIZE)
def _embed_query(query: str) -> np.ndarray:
    """Get query embedding (cached)."""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    res = client.embeddings.create(model=EMBED_MODEL, input=query)
    return np.asarray(res.data[0].embedding, dtype=np.float32)


# --------------------------------------------------------------------------- #
# 🤖  Answer generation                                                        #
# --------------------------------------------------------------------------- #
def answer_query_sync(query: str) -> str:
    """Answer a question about news articles."""
    # 1️⃣  Embed query & find relevant articles
    query_vec = _embed_query(query)
    hits = search_vectors(query_vec, k=TOP_K)

    # 2️⃣  Handle case where no articles are found
    if not hits:
        return "I don't have any relevant news articles to answer your question. The news ingestion pipeline may still be loading data."

    # 3️⃣  Build prompt
    context = "\n".join(f"- {headline}" for headline, _ in hits)
    prompt = (
        "You are a financial news analyst. Answer the following question based on "
        "the provided news headlines. Keep your answer concise and factual.\n\n"
        f"Headlines:\n{context}\n\nQuestion: {query}"
    )

    # 4️⃣  Generate answer
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    res = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=200,
    )
    
    # Handle case where OpenAI returns no choices
    if not res.choices or not res.choices[0].message.content:
        return "I'm sorry, I couldn't generate a response. Please try again."
    
    return res.choices[0].message.content.strip()