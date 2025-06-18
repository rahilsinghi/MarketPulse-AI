"""
MarketPulse ML - Retrieval Module
Handles semantic document search and ranking.
"""

import logging
from typing import List, Dict, Optional
import numpy as np

from .embeddings import embed_text, cosine_similarity

# Set up logging
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_SIMILARITY_THRESHOLD = 0.30  # Increased from 0.25 for better quality
DEFAULT_TOP_K = 4

def retrieve_top_k(
    query_vec: List[float], 
    k: int = DEFAULT_TOP_K,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> List[dict]:
    """
    Find most semantically similar documents to query vector.
    
    Args:
        query_vec (List[float]): Query embedding vector (1536 dimensions)
        k (int): Number of top results to return
        similarity_threshold (float): Minimum similarity score to include
        
    Returns:
        List[dict]: Ranked list of documents with similarity scores
        
    Example:
        >>> query_vec = embed_text("Tesla delivery numbers")
        >>> results = retrieve_top_k(query_vec, k=3)
        >>> len(results) <= 3
        True
        >>> all('similarity' in doc for doc in results)
        True
    """
    try:
        # Import here to avoid circular imports
        from ingest_pipeline import get_latest_rows
        rows = get_latest_rows()
    except ImportError:
        logger.warning("ingest_pipeline not available, using mock data")
        rows = _get_mock_rows()
    
    if not rows:
        logger.warning("No rows available for retrieval")
        return []
    
    # Filter rows that have embeddings
    valid_rows = [row for row in rows if row.get('embedding')]
    
    if len(valid_rows) > 1000:
        # Use FAISS for large datasets
        return _faiss_search(query_vec, valid_rows, k, similarity_threshold)
    else:
        # Use linear search for smaller datasets
        return _linear_search(query_vec, valid_rows, k, similarity_threshold)

def retrieve_by_query(
    query: str, 
    k: int = DEFAULT_TOP_K,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> List[dict]:
    """
    Retrieve documents using natural language query.
    
    Args:
        query (str): Natural language search query
        k (int): Number of top results to return
        similarity_threshold (float): Minimum similarity score to include
        
    Returns:
        List[dict]: Ranked list of relevant documents
        
    Example:
        >>> results = retrieve_by_query("Microsoft cloud earnings", k=3)
        >>> any('MSFT' in doc.get('ticker', '') for doc in results)
        True
    """
    query_vec = embed_text(query)
    return retrieve_top_k(query_vec, k, similarity_threshold)

def _linear_search(
    query_vec: List[float], 
    rows: List[dict], 
    k: int,
    similarity_threshold: float
) -> List[dict]:
    """
    Linear search with cosine similarity.
    
    Args:
        query_vec: Query embedding vector
        rows: List of document rows with embeddings
        k: Number of results to return
        similarity_threshold: Minimum similarity to include
        
    Returns:
        List of ranked documents with similarity scores
    """
    scored_docs = []
    query_vec_np = np.array(query_vec)
    
    for row in rows:
        try:
            doc_vec = np.array(row['embedding'])
            similarity = cosine_similarity(query_vec, doc_vec)
            
            # Only include documents above threshold
            if similarity >= similarity_threshold:
                result = row.copy()
                result['similarity'] = float(similarity)
                scored_docs.append(result)
            
        except Exception as e:
            logger.warning(f"Error calculating similarity for row: {e}")
            continue
    
    # Sort by similarity and return top-k
    scored_docs.sort(key=lambda x: x['similarity'], reverse=True)
    results = scored_docs[:k]
    
    logger.info(f"Linear search returned {len(results)} results (threshold: {similarity_threshold})")
    return results

def _faiss_search(
    query_vec: List[float], 
    rows: List[dict], 
    k: int,
    similarity_threshold: float
) -> List[dict]:
    """
    FAISS-based search for large document collections.
    
    Args:
        query_vec: Query embedding vector
        rows: List of document rows with embeddings
        k: Number of results to return
        similarity_threshold: Minimum similarity to include
        
    Returns:
        List of ranked documents with similarity scores
    """
    try:
        import faiss
    except ImportError:
        logger.warning("FAISS not available, falling back to linear search")
        return _linear_search(query_vec, rows, k, similarity_threshold)
    
    try:
        # Build FAISS index
        embeddings = np.array([row['embedding'] for row in rows], dtype=np.float32)
        
        # Normalize vectors for inner product similarity
        faiss.normalize_L2(embeddings)
        query_norm = np.array([query_vec], dtype=np.float32)
        faiss.normalize_L2(query_norm)
        
        # Create index
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        # Search for more results to filter by threshold
        search_k = min(k * 3, len(rows))  # Search more to filter by threshold
        similarities, indices = index.search(query_norm, search_k)
        
        # Build results and filter by threshold
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < len(rows) and similarity >= similarity_threshold:
                result = rows[idx].copy()
                result['similarity'] = float(similarity)
                results.append(result)
                
                if len(results) >= k:
                    break
        
        logger.info(f"FAISS search returned {len(results)} results (threshold: {similarity_threshold})")
        return results
        
    except Exception as e:
        logger.error(f"FAISS search failed: {e}, falling back to linear search")
        return _linear_search(query_vec, rows, k, similarity_threshold)

def _get_mock_rows() -> List[dict]:
    """Fallback mock data when ingest_pipeline is not available."""
    from datetime import datetime, timedelta
    
    base_time = datetime.now()
    mock_data = [
        {
            "ticker": "AAPL",
            "headline": "Apple reports record quarterly earnings, beating analyst expectations",
            "timestamp": (base_time - timedelta(hours=2)).isoformat(),
            "source": "Reuters",
            "category": "earnings",
            "embedding": embed_text("Apple reports record quarterly earnings, beating analyst expectations")
        },
        {
            "ticker": "TSLA", 
            "headline": "Tesla delivers 500,000 vehicles in Q4, stock surges 8%",
            "timestamp": (base_time - timedelta(hours=1)).isoformat(),
            "source": "Bloomberg",
            "category": "delivery",
            "embedding": embed_text("Tesla delivers 500,000 vehicles in Q4, stock surges 8%")
        },
        {
            "ticker": "MSFT",
            "headline": "Microsoft Azure revenue grows 35% YoY, cloud dominance continues",
            "timestamp": (base_time - timedelta(minutes=45)).isoformat(),
            "source": "CNBC",
            "category": "earnings",
            "embedding": embed_text("Microsoft Azure revenue grows 35% YoY, cloud dominance continues")
        },
        {
            "ticker": "GOOGL",
            "headline": "Google announces breakthrough in quantum computing research",
            "timestamp": (base_time - timedelta(minutes=30)).isoformat(),
            "source": "TechCrunch",
            "category": "innovation",
            "embedding": embed_text("Google announces breakthrough in quantum computing research")
        }
    ]
    
    return mock_data

# Performance monitoring
def get_retrieval_stats() -> dict:
    """
    Get retrieval performance statistics.
    
    Returns:
        dict: Statistics about retrieval performance
    """
    try:
        from ingest_pipeline import get_pipeline_stats
        stats = get_pipeline_stats()
        return {
            "total_documents": stats.get('total_articles', 0),
            "documents_with_embeddings": stats.get('articles_with_embeddings', 0),
            "unique_tickers": stats.get('unique_tickers', 0),
            "default_threshold": DEFAULT_SIMILARITY_THRESHOLD,
            "default_top_k": DEFAULT_TOP_K
        }
    except Exception as e:
        logger.warning(f"Could not get retrieval stats: {e}")
        return {
            "total_documents": 0,
            "documents_with_embeddings": 0,
            "unique_tickers": 0,
            "default_threshold": DEFAULT_SIMILARITY_THRESHOLD,
            "default_top_k": DEFAULT_TOP_K
        }