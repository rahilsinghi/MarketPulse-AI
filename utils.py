"""
Utility helpers with updated OpenAI API.
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional
from functools import wraps, lru_cache

from openai import OpenAI
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/marketpulse.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# ENV & OPENAI
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY is required")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Rate limiting and retry decorator
def retry_with_backoff(max_retries=3, backoff_factor=1.0):
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise e
                    
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

# Enhanced Embedding helper with new API
@retry_with_backoff(max_retries=3, backoff_factor=0.5)
@lru_cache(maxsize=1000)
def embed_text(text: str, model: str = "text-embedding-3-small") -> Optional[List[float]]:
    """
    Call OpenAI embedding API using new client and return the vector.
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for embedding")
        return None
    
    try:
        # Truncate text if too long
        if len(text) > 8000:
            text = text[:8000]
            logger.warning("Text truncated for embedding due to length")
        
        logger.debug(f"Generating embedding for text: {text[:100]}...")
        
        response = client.embeddings.create(
            model=model,
            input=[text.strip()]
        )
        
        embedding = response.data[0].embedding
        logger.debug(f"Successfully generated embedding of dimension {len(embedding)}")
        return embedding
        
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise

# Mock pathway functions for now
def get_pathway_table() -> List[Dict[str, Any]]:
    """Mock pathway table data."""
    from datetime import datetime, timedelta
    
    mock_data = [
        {
            "id": "AAPL_1",
            "ticker": "AAPL", 
            "headline": "Apple reports record quarterly earnings, beating expectations",
            "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
            "embedding": embed_text("Apple reports record quarterly earnings")
        },
        {
            "id": "GOOGL_1",
            "ticker": "GOOGL",
            "headline": "Google announces breakthrough in quantum computing research", 
            "timestamp": (datetime.now() - timedelta(hours=1)).isoformat(),
            "embedding": embed_text("Google announces breakthrough in quantum computing")
        },
        {
            "id": "TSLA_1", 
            "ticker": "TSLA",
            "headline": "Tesla stock surges 8% on record vehicle deliveries",
            "timestamp": (datetime.now() - timedelta(minutes=30)).isoformat(),
            "embedding": embed_text("Tesla stock surges on record deliveries")
        }
    ]
    
    return mock_data

def validate_embedding(embedding: List[float]) -> bool:
    """Validate that embedding is properly formatted."""
    if not embedding or not isinstance(embedding, list):
        return False
    if len(embedding) != 1536:  # text-embedding-3-small dimension
        logger.warning(f"Unexpected embedding dimension: {len(embedding)}")
        return False
    return all(isinstance(x, (int, float)) for x in embedding)

def validate_news_record(record: Dict[str, Any]) -> bool:
    """Validate news record has required fields."""
    required_fields = ['id', 'ticker', 'headline', 'timestamp']
    return all(field in record and record[field] for field in required_fields)