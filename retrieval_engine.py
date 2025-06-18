"""
MarketPulse AI - Legacy Retrieval Engine (Updated to use marketpulse_ml package)
This file maintains backward compatibility while using the new modular package.
"""

import os
import time
import asyncio
import logging
from typing import List, Dict
from datetime import datetime

# Import from our new package
from marketpulse_ml import (
    embed_text,
    retrieve_top_k as ml_retrieve_top_k,
    cosine_similarity,
    DEFAULT_SIMILARITY_THRESHOLD
)

# Set up logging
logger = logging.getLogger(__name__)

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MOCK_EMBEDDING = os.getenv("MOCK_EMBEDDING", "false").lower() == "true"

# Updated similarity threshold from package
SIMILARITY_THRESHOLD = DEFAULT_SIMILARITY_THRESHOLD  # Now 0.30 instead of 0.25

def retrieve_top_k(query_vec: List[float], k: int = 4) -> List[dict]:
    """
    Wrapper function for backward compatibility.
    Uses the new marketpulse_ml package underneath.
    """
    return ml_retrieve_top_k(query_vec, k, SIMILARITY_THRESHOLD)

def build_prompt(question: str, docs: List[dict]) -> List[dict]:
    """
    Build structured prompt for OpenAI chat completion.
    Enhanced to work better with the improved similarity threshold.
    """
    if not docs:
        return [
            {"role": "system", "content": "You are a financial analyst. No recent news is available."},
            {"role": "user", "content": question}
        ]
    
    # Enhanced system message
    system_msg = """You are a concise, real-time market analyst. Answer questions about financial news using ONLY the provided recent articles. 

Guidelines:
- Cite specific companies using [TICKER] format
- Include relevant numbers and percentages when available
- Mention timeframes when provided
- If multiple companies are relevant, mention the most relevant ones
- Keep responses under 3 sentences
- Focus on factual information from the articles"""

    # Build context with improved formatting
    context_parts = []
    for doc in docs:
        ticker = doc.get('ticker', 'Unknown')
        headline = doc.get('headline', 'No headline')
        similarity = doc.get('similarity', 0)
        timestamp = doc.get('timestamp', '')
        source = doc.get('source', '')
        
        # Include similarity score for context quality
        if similarity >= 0.8:
            quality_indicator = "🔥"  # High relevance
        elif similarity >= 0.6:
            quality_indicator = "✅"  # Good relevance
        elif similarity >= 0.4:
            quality_indicator = "📊"  # Moderate relevance
        else:
            quality_indicator = "💭"  # Lower relevance
        
        # Format timestamp
        timestamp_str = ""
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamp_str = f" — {dt.strftime('%Y-%m-%d %H:%M')}"
            except:
                timestamp_str = f" — {timestamp}"
        
        context_parts.append(f"{quality_indicator} [{ticker}] {headline}{timestamp_str}")
        if source:
            context_parts[-1] += f" (Source: {source})"
    
    context = "\n".join(context_parts)
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Recent financial news:\n{context}\n\nQuestion: {question}"}
    ]
    
    return messages

def answer_query_sync(question: str, top_k: int = 4) -> str:
    """
    Complete Q&A pipeline using the new marketpulse_ml package.
    Enhanced with improved similarity threshold and better error handling.
    """
    t0 = time.time()
    
    try:
        # Get query embedding using package function
        query_vec = embed_text(question)
        
        # Retrieve relevant documents using package function with improved threshold
        docs = retrieve_top_k(query_vec, top_k)
        
        # Enhanced logging with similarity scores
        logger.info(f"Query: '{question}' found {len(docs)} documents above threshold {SIMILARITY_THRESHOLD}")
        for i, doc in enumerate(docs[:3]):
            ticker = doc.get('ticker', 'N/A')
            similarity = doc.get('similarity', 0)
            headline = doc.get('headline', 'N/A')[:60]
            logger.info(f"  {i+1}. [{ticker}] {similarity:.3f} - {headline}...")
        
        # Filter by improved similarity threshold
        relevant_docs = [doc for doc in docs if doc.get('similarity', 0) >= SIMILARITY_THRESHOLD]
        
        if not relevant_docs:
            t1 = time.time()
            return f"(🕒 {t1-t0:.1f}s) No highly relevant recent news found for that query. Try rephrasing or asking about specific companies."
        
        # Build enhanced prompt
        messages = build_prompt(question, relevant_docs)
        
        # Debug logging
        logger.info(f"MOCK_EMBEDDING: {MOCK_EMBEDDING}, OPENAI_API_KEY: {'SET' if OPENAI_API_KEY else 'NOT SET'}")
        logger.info(f"Using similarity threshold: {SIMILARITY_THRESHOLD}")
        
        # Get AI response with better fallback
        if OPENAI_API_KEY and not MOCK_EMBEDDING:
            try:
                logger.info("🤖 Attempting real OpenAI chat completion...")
                answer = asyncio.run(_answer_query_async(messages))
                logger.info("✅ Real OpenAI chat response generated successfully")
            except Exception as e:
                logger.warning(f"❌ OpenAI chat failed: {e}")
                logger.info("🔄 Falling back to smart mock response...")
                answer = _generate_smart_mock_response(messages, relevant_docs, question)
        else:
            logger.info("🧪 Using smart mock response (demo mode)")
            answer = _generate_smart_mock_response(messages, relevant_docs, question)
        
        t1 = time.time()
        return f"(🕒 {t1-t0:.1f}s) {answer}"
        
    except Exception as e:
        t1 = time.time()
        logger.error(f"Error in answer_query_sync: {e}")
        return f"(🕒 {t1-t0:.1f}s) Error processing query: {str(e)}"

async def _answer_query_async(messages: List[dict]) -> str:
    """
    Enhanced async OpenAI chat completion with better error handling.
    """
    if MOCK_EMBEDDING or not OPENAI_API_KEY:
        return _generate_smart_mock_response(messages, [], "")
    
    try:
        from openai import AsyncOpenAI
        
        async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        try:
            response = await async_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=300,
                temperature=0.1,
                top_p=0.9  # Added for more focused responses
            )
            
            result = response.choices[0].message.content.strip()
            await async_client.close()
            return result
            
        except Exception as rate_error:
            if "rate" in str(rate_error).lower():
                logger.warning("Rate limit hit, waiting 2s and retrying...")
                await asyncio.sleep(2)
                
                response = await async_client.chat.completions.create(
                    model="gpt-4o-mini", 
                    messages=messages,
                    max_tokens=300,
                    temperature=0.1,
                    top_p=0.9
                )
                
                result = response.choices[0].message.content.strip()
                await async_client.close()
                return result
            else:
                await async_client.close()
                raise rate_error
                
    except Exception as e:
        logger.warning(f"OpenAI chat API failed: {e}")
        raise e

def _generate_smart_mock_response(messages: List[dict], relevant_docs: List[dict], question: str) -> str:
    """
    Enhanced smart mock response using actual retrieved documents.
    Now works with improved similarity threshold.
    """
    if not relevant_docs:
        return "I don't have any highly relevant recent news for that query. The similarity threshold has been increased for better quality results."
    
    # Get the most relevant document
    top_doc = relevant_docs[0]
    ticker = top_doc.get('ticker', 'Unknown')
    headline = top_doc.get('headline', 'No headline available')
    similarity = top_doc.get('similarity', 0)
    source = top_doc.get('source', 'Unknown source')
    
    logger.info(f"📰 Generating smart response based on: [{ticker}] {similarity:.3f} - {headline}")
    
    # Enhanced response generation based on similarity score
    if similarity > 0.8:
        confidence_phrase = "Based on highly relevant recent news"
    elif similarity > 0.6:
        confidence_phrase = "According to relevant market news"
    elif similarity > 0.4:
        confidence_phrase = "From recent financial news"
    else:
        confidence_phrase = "Based on available market information"
    
    # Create contextual response using actual content
    base_response = f"{confidence_phrase}, here's what I found: [{ticker}] {headline}"
    
    # Add additional context from other relevant docs
    if len(relevant_docs) > 1:
        other_high_relevance = [doc for doc in relevant_docs[1:3] if doc.get('similarity', 0) > 0.5]
        if other_high_relevance:
            additional_tickers = [doc.get('ticker') for doc in other_high_relevance if doc.get('ticker') != ticker]
            if additional_tickers:
                base_response += f" Also relevant: {', '.join(set(additional_tickers))}"
    
    # Add source and confidence indicator
    base_response += f" (Source: {source}, Relevance: {similarity:.2f})"
    
    return base_response

# Export main functions for backward compatibility
__all__ = [
    "embed_text",           # From marketpulse_ml
    "retrieve_top_k",       # Wrapper around ml_retrieve_top_k
    "build_prompt",         # Enhanced version
    "answer_query_sync",    # Enhanced version
    "cosine_similarity",    # From marketpulse_ml
    "SIMILARITY_THRESHOLD"  # Updated threshold
]