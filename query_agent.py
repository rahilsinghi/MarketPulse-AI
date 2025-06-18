"""
Retrieval-Augmented answer engine.
Given a user question it:
  1. Embeds the question
  2. Retrieves the K most similar news items from Pathway
  3. Calls GPT-4/o with a prompt citing those snippets
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict

import numpy as np
import openai

from utils import get_pathway_table, embed_text

logger = logging.getLogger(__name__)


# ─── Cosine similarity ─────────────────────────────────────────────────────────
def cosine_similarity(q: List[float], d: List[float]) -> float:
    qv, dv = np.array(q), np.array(d)
    denom = np.linalg.norm(qv) * np.linalg.norm(dv)
    if denom == 0:
        return 0.0
    return float(np.dot(qv, dv) / denom)


# ─── Async OpenAI call helper ──────────────────────────────────────────────────
async def _call_openai_async(messages: List[Dict[str, str]]) -> str:
    try:
        resp = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",        # use your plan’s model
            messages=messages,
            temperature=0.3,
            max_tokens=350,
            timeout=15,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return "⚠️ OpenAI error – please try again."


# ─── Public entrypoint ---------------------------------------------------------
def answer_query_sync(question: str, top_k: int = 4) -> str:
    """
    Synchronous wrapper so Streamlit can call it easily.
    """
    return asyncio.run(_answer_query(question, top_k))


# ─── Core coroutine ------------------------------------------------------------
async def _answer_query(question: str, top_k: int) -> str:
    rows = get_pathway_table()
    if not rows:
        return "⏳ No news ingested yet – please wait a moment."

    q_vec = embed_text(question)

    scored: List[Dict] = []
    for r in rows:
        if r.get("embedding"):
            sim = cosine_similarity(q_vec, r["embedding"])
            scored.append({**r, "similarity": sim})

    if not scored:
        return "🤷 I couldn’t find any relevant fresh news."

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    top_docs = scored[:top_k]

    context_lines = []
    for d in top_docs:
        ts = (
            datetime.fromisoformat(d["timestamp"])
            .strftime("%Y-%m-%d %H:%M")
            if isinstance(d["timestamp"], str)
            else str(d["timestamp"])
        )
        context_lines.append(f"• [{d['ticker']}] {d['headline']} – {ts}")

    context_block = "\n".join(context_lines)

    system_msg = (
        "You are MarketPulse AI, a real-time market analyst. "
        "Answer with the most relevant, *fresh* information first. "
        "Cite the bullet points if they explain the move. "
        "If nothing explains it, say 'No relevant updates yet.'"
    )

    user_msg = (
        f"RECENT NEWS CONTEXT:\n{context_block}\n\n"
        f"USER QUESTION: {question}\n\n"
        "Respond briefly (2-3 sentences) and mention the relevant bullet(s)."
    )

    reply = await _call_openai_async(
        [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    )
    return reply
