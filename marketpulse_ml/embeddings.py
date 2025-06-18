"""
MarketPulse ML - Embedding Module
Handles text-to-vector conversion with OpenAI integration.
"""

import os
import logging
from typing import List
from functools import lru_cache
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MOCK_EMBEDDING = os.getenv("MOCK_EMBEDDING", "false").lower() == "true"

@lru_cache(maxsize=512)  # Increased cache size for package use
def embed_text(text: str) -> List[float]:
    """
    Convert text to 1536-dimensional semantic embedding vector.
    
    Uses OpenAI's text-embedding-3-small model for production,
    or deterministic mock embeddings for testing/development.
    
    Args:
        text (str): Input text to embed (max 8000 chars)
        
    Returns:
        List[float]: 1536-dimensional embedding vector
        
    Example:
        >>> embedding = embed_text("Apple quarterly earnings report")
        >>> len(embedding)
        1536
        >>> type(embedding[0])
        <class 'float'>
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for embedding")
        return [0.0] * 1536
    
    # Use mock embeddings for development/testing
    if MOCK_EMBEDDING or not OPENAI_API_KEY:
        return _generate_mock_embedding(text)
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[text.strip()[:8000]]  # Truncate if too long
        )
        
        embedding = response.data[0].embedding
        logger.debug(f"Generated OpenAI embedding for: {text[:50]}...")
        return embedding
        
    except Exception as e:
        logger.warning(f"OpenAI embedding failed, using mock: {e}")
        return _generate_mock_embedding(text)

def _generate_mock_embedding(text: str) -> List[float]:
    """
    Generate deterministic mock embedding for testing.
    
    Args:
        text (str): Input text
        
    Returns:
        List[float]: Deterministic 1536-dimensional vector
    """
    # Deterministic mock embedding based on text hash
    np.random.seed(hash(text) % 2**32)
    embedding = np.random.rand(1536).astype(float).tolist()
    logger.debug(f"Generated mock embedding for: {text[:50]}...")
    return embedding

def batch_embed_texts(texts: List[str], batch_size: int = 10) -> List[List[float]]:
    """
    Generate embeddings for multiple texts efficiently.
    
    Args:
        texts (List[str]): List of texts to embed
        batch_size (int): Number of texts to process in each batch
        
    Returns:
        List[List[float]]: List of embedding vectors
        
    Example:
        >>> texts = ["Apple news", "Tesla updates", "Microsoft earnings"]
        >>> embeddings = batch_embed_texts(texts)
        >>> len(embeddings)
        3
        >>> len(embeddings[0])
        1536
    """
    embeddings = []
    for text in texts:
        embedding = embed_text(text)
        embeddings.append(embedding)
    return embeddings

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two embedding vectors.
    
    Args:
        vec1 (List[float]): First embedding vector
        vec2 (List[float]): Second embedding vector
        
    Returns:
        float: Cosine similarity score (0.0 to 1.0)
        
    Example:
        >>> vec1 = embed_text("Apple earnings")
        >>> vec2 = embed_text("Apple quarterly results")
        >>> similarity = cosine_similarity(vec1, vec2)
        >>> 0.0 <= similarity <= 1.0
        True
    """
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Cache statistics for monitoring
def get_embedding_cache_stats() -> dict:
    """
    Get statistics about the embedding cache.
    
    Returns:
        dict: Cache statistics including hits, misses, size
    """
    cache_info = embed_text.cache_info()
    return {
        "hits": cache_info.hits,
        "misses": cache_info.misses,
        "maxsize": cache_info.maxsize,
        "currsize": cache_info.currsize,
        "hit_rate": cache_info.hits / (cache_info.hits + cache_info.misses) if (cache_info.hits + cache_info.misses) > 0 else 0.0
    }

def clear_embedding_cache():
    """Clear the embedding cache."""
    embed_text.cache_clear()
    logger.info("Embedding cache cleared")