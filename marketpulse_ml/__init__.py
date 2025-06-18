"""
MarketPulse ML - AI-Powered Financial News Analysis Package

This package provides semantic search and AI-powered analysis capabilities
for financial news and market data.

Main Functions:
    embed_text: Convert text to semantic embeddings
    retrieve_top_k: Find similar documents using vector search
    retrieve_by_query: Natural language document search
    answer_query_sync: Complete Q&A pipeline
"""

from .embeddings import (
    embed_text,
    batch_embed_texts,
    cosine_similarity,
    get_embedding_cache_stats,
    clear_embedding_cache
)

from .retrieval import (
    retrieve_top_k,
    retrieve_by_query,
    get_retrieval_stats,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K
)

# Import answer_query_sync from the original retrieval_engine for now
# This maintains backward compatibility while we transition
try:
    from retrieval_engine import answer_query_sync, build_prompt
except ImportError:
    # Fallback if retrieval_engine is not available
    def answer_query_sync(question: str, top_k: int = 4) -> str:
        """Fallback implementation when retrieval_engine is not available."""
        return f"MarketPulse ML package loaded. Query: {question} (fallback mode)"
    
    def build_prompt(question: str, docs: list) -> list:
        """Fallback implementation when retrieval_engine is not available."""
        return [
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": question}
        ]

# Package metadata
__version__ = "1.0.0"
__author__ = "MarketPulse AI Team"
__description__ = "AI-Powered Financial News Analysis with Semantic Search"

# Export main functions
__all__ = [
    # Core embedding functions
    "embed_text",
    "batch_embed_texts", 
    "cosine_similarity",
    
    # Retrieval functions
    "retrieve_top_k",
    "retrieve_by_query",
    
    # Q&A pipeline
    "answer_query_sync",
    "build_prompt",
    
    # Utility functions
    "get_embedding_cache_stats",
    "clear_embedding_cache",
    "get_retrieval_stats",
    
    # Constants
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DEFAULT_TOP_K",
]

# Package-level configuration
def configure_package(
    openai_api_key: str = None,
    mock_embedding: bool = False,
    similarity_threshold: float = 0.30,
    cache_size: int = 512
):
    """
    Configure MarketPulse ML package settings.
    
    Args:
        openai_api_key: OpenAI API key for embeddings
        mock_embedding: Use mock embeddings instead of OpenAI
        similarity_threshold: Default similarity threshold for retrieval
        cache_size: Size of embedding cache
    """
    import os
    
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    
    if mock_embedding:
        os.environ["MOCK_EMBEDDING"] = "true"
    else:
        os.environ["MOCK_EMBEDDING"] = "false"
    
    # Update module-level defaults
    global DEFAULT_SIMILARITY_THRESHOLD
    DEFAULT_SIMILARITY_THRESHOLD = similarity_threshold
    
    print(f"✅ MarketPulse ML configured:")
    print(f"   OpenAI API: {'Mock' if mock_embedding else 'Real'}")
    print(f"   Similarity threshold: {similarity_threshold}")
    print(f"   Cache size: {cache_size}")