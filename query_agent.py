"""
Enhanced Retrieval-Augmented answer engine with better error handling,
logging, and more sophisticated retrieval strategies.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json

import numpy as np
import openai
from dataclasses import dataclass
from sui_logger import log_interaction 

from utils import get_pathway_table, embed_text, validate_embedding, validate_news_record

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Structure for retrieval results with metadata."""
    documents: List[Dict]
    query_embedding: List[float]
    total_docs: int
    retrieval_time: float

# ─── Enhanced Cosine similarity with validation ────────────────────────────────
def cosine_similarity(q: List[float], d: List[float]) -> float:
    """Calculate cosine similarity with validation."""
    try:
        if not validate_embedding(q) or not validate_embedding(d):
            logger.warning("Invalid embeddings provided for similarity calculation")
            return 0.0
            
        qv, dv = np.array(q), np.array(d)
        denom = np.linalg.norm(qv) * np.linalg.norm(dv)
        if denom == 0:
            return 0.0
        return float(np.dot(qv, dv) / denom)
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0

# ─── Enhanced retrieval with filtering and ranking ─────────────────────────────
def retrieve_relevant_docs(
    question: str, 
    top_k: int = 4, 
    min_similarity: float = 0.1,
    time_weight: float = 0.2,
    max_age_hours: int = 24
) -> RetrievalResult:
    """
    Enhanced retrieval with time-based weighting and filtering.
    """
    start_time = datetime.now()
    
    try:
        # Get documents
        rows = get_pathway_table()
        if not rows:
            logger.warning("No documents available for retrieval")
            return RetrievalResult([], [], 0, 0.0)
        
        # Embed query
        q_vec = embed_text(question)
        if not q_vec:
            logger.error("Failed to embed query")
            return RetrievalResult([], [], len(rows), 0.0)
        
        # Filter and score documents
        scored_docs = []
        current_time = datetime.now()
        
        for row in rows:
            if not validate_news_record(row) or not row.get("embedding"):
                continue
                
            # Calculate semantic similarity
            similarity = cosine_similarity(q_vec, row["embedding"])
            if similarity < min_similarity:
                continue
            
            # Calculate time decay factor
            try:
                doc_time = datetime.fromisoformat(row["timestamp"])
                age_hours = (current_time - doc_time).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    continue
                    
                # Recent docs get higher weight
                time_factor = max(0, 1 - (age_hours / max_age_hours))
                
                # Combined score
                final_score = similarity * (1 - time_weight) + time_factor * time_weight
                
                scored_docs.append({
                    **row,
                    "similarity": similarity,
                    "time_factor": time_factor,
                    "final_score": final_score,
                    "age_hours": age_hours
                })
                
            except Exception as e:
                logger.warning(f"Error processing document timestamp: {e}")
                # Fallback to similarity only
                scored_docs.append({
                    **row,
                    "similarity": similarity,
                    "time_factor": 0.5,
                    "final_score": similarity,
                    "age_hours": 0
                })
        
        # Sort by final score and take top-k
        scored_docs.sort(key=lambda x: x["final_score"], reverse=True)
        top_docs = scored_docs[:top_k]
        
        retrieval_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Retrieved {len(top_docs)} documents in {retrieval_time:.2f}s")
        top_scores = [f"{d['final_score']:.3f}" for d in top_docs[:3]]
        logger.debug(f"Top document scores: {top_scores}")
        
        
        return RetrievalResult(top_docs, q_vec, len(rows), retrieval_time)
        
    except Exception as e:
        logger.error(f"Error in document retrieval: {e}")
        return RetrievalResult([], [], 0, 0.0)

# ─── Enhanced prompt engineering ───────────────────────────────────────────────
def build_context_prompt(docs: List[Dict], question: str) -> Tuple[str, str]:
    """Build enhanced system and user prompts with better context."""
    
    # Group by ticker for better organization
    ticker_groups = {}
    for doc in docs:
        ticker = doc.get('ticker', 'UNKNOWN')
        if ticker not in ticker_groups:
            ticker_groups[ticker] = []
        ticker_groups[ticker].append(doc)
    
    # Build context with metadata
    context_sections = []
    for ticker, ticker_docs in ticker_groups.items():
        context_sections.append(f"\n**{ticker}:**")
        for doc in ticker_docs:
            timestamp = doc.get('timestamp', 'Unknown time')
            age_hours = doc.get('age_hours', 0)
            similarity = doc.get('similarity', 0)
            
            # Format timestamp nicely
            try:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime("%Y-%m-%d %H:%M UTC")
                if age_hours < 1:
                    age_str = f"({int(age_hours * 60)} minutes ago)"
                else:
                    age_str = f"({age_hours:.1f} hours ago)"
            except:
                time_str = timestamp
                age_str = ""
            
            context_sections.append(
                f"  • {doc['headline']} [{time_str} {age_str}] "
                f"[Relevance: {similarity:.2f}]"
            )
    
    context_block = "\n".join(context_sections)
    
    system_prompt = """You are MarketPulse AI, an expert financial analyst providing real-time market insights.

**Your capabilities:**
- Analyze recent market news and identify trends
- Explain stock price movements and market sentiment
- Provide context for financial events and their implications
- Cite specific news sources in your analysis

**Response guidelines:**
- Be concise but informative (2-4 sentences)
- Always cite the most relevant news sources
- If news doesn't explain the query, state this clearly
- Focus on the most recent and relevant information
- Use professional, accessible language
- Include specific tickers when relevant"""

    user_prompt = f"""**RECENT MARKET NEWS CONTEXT:**
{context_block}

**USER QUESTION:** {question}

**INSTRUCTIONS:** Analyze the above news context to answer the user's question. Cite the most relevant news items and explain their significance. If the available news doesn't adequately address the question, say so clearly."""

    return system_prompt, user_prompt

# ─── Enhanced async OpenAI call ────────────────────────────────────────────────
async def call_openai_with_retry(
    messages: List[Dict[str, str]], 
    model: str = "gpt-4o-mini",
    max_retries: int = 2
) -> str:
    """Enhanced OpenAI call with retry logic and better error handling."""
    
    for attempt in range(max_retries + 1):
        try:
            logger.debug(f"Calling OpenAI API (attempt {attempt + 1})")
            
            resp = await openai.ChatCompletion.acreate(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=400,
                timeout=20,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            content = resp.choices[0].message.content.strip()
            
            # Log usage statistics
            if hasattr(resp, 'usage'):
                logger.info(f"OpenAI usage - Prompt: {resp.usage.prompt_tokens}, "
                           f"Completion: {resp.usage.completion_tokens}")
            
            return content
            
        except openai.error.RateLimitError as e:
            wait_time = 2 ** attempt
            logger.warning(f"Rate limit hit, waiting {wait_time}s...")
            await asyncio.sleep(wait_time)
            
        except openai.error.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            if attempt == max_retries:
                return "⚠️ I'm experiencing technical difficulties. Please try again in a moment."
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI: {e}")
            if attempt == max_retries:
                return "⚠️ Something went wrong while generating the response. Please try again."
            await asyncio.sleep(0.5)
    
    return "⚠️ Unable to generate response after multiple attempts."

# ─── Enhanced main query function ──────────────────────────────────────────────
async def answer_query_async(
    question: str, 
    top_k: int = 4,
    include_metadata: bool = False
) -> str:
    """Enhanced async query answering with comprehensive error handling."""
    
    start_time = datetime.now()
    
    try:
        # Input validation
        if not question or not question.strip():
            return "⚠️ Please provide a valid question."
        
        question = question.strip()
        if len(question) > 500:
            question = question[:500]
            logger.warning("Question truncated due to length")
        
        # Retrieve relevant documents
        retrieval_result = retrieve_relevant_docs(question, top_k)
        
        if not retrieval_result.documents:
            if retrieval_result.total_docs == 0:
                return "⏳ No news has been ingested yet. Please wait a moment for the system to gather market data."
            else:
                return "🤷 I couldn't find any recent news relevant to your question. Try asking about specific stocks or market events."
        
        # Build enhanced prompts
        system_prompt, user_prompt = build_context_prompt(retrieval_result.documents, question)
        
        # Generate response
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = await call_openai_with_retry(messages)
        
        # Add metadata if requested
        if include_metadata:
            total_time = (datetime.now() - start_time).total_seconds()
            metadata = (f"\n\n*[Retrieved {len(retrieval_result.documents)} sources "
                       f"from {retrieval_result.total_docs} total documents in {total_time:.1f}s]*")
            response += metadata
        
        logger.info(f"Successfully answered query in {(datetime.now() - start_time).total_seconds():.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error in answer_query_async: {e}")
        return "⚠️ I encountered an unexpected error. Please try again."

# ─── Synchronous wrapper ──────────────────────────────────────────────────────
def answer_query_sync(question: str, top_k: int = 4, include_metadata: bool = False) -> str:
    """Synchronous wrapper for Streamlit compatibility."""
  
    # -- after answer is generated, before return --
    try:
        # naive ticker extraction: first [XYZ] bullet or None
        first_bullet = next((d["ticker"] for d in top_docs if d.get("ticker")), None)
        log_interaction(question, reply, first_bullet)
    except Exception:
        pass  # never block on logging
        return "⚠️ System error. Please try again."