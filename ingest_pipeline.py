"""
MarketPulse AI - Enhanced Ingest Pipeline with Pathway Integration
Updated to work with both real-time Pathway data and mock data fallback.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import threading
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import from marketpulse_ml package
from marketpulse_ml import embed_text

# Check if Pathway integration is available
try:
    from pathway_pipeline import get_latest_rows as pathway_get_latest_rows, start_pathway
    PATHWAY_AVAILABLE = True
    print("✅ Pathway integration available")
except ImportError:
    PATHWAY_AVAILABLE = False
    print("⚠️ Pathway not available, using mock data only")

# Set up logging
logger = logging.getLogger(__name__)

# Data source configuration
DATA_SOURCE_CONFIG = {
    "pathway": {
        "enabled": os.getenv("USE_PATHWAY", "false").lower() == "true",
        "live_feed": os.getenv("USE_LIVE_FEED", "false").lower() == "true",
        "finnhub_token": os.getenv("FINNHUB_TOKEN", ""),
        "symbols": os.getenv("SYMBOLS", "AAPL,TSLA,MSFT,GOOGL,NVDA,META,AMZN").split(","),
    },
    "mock": {
        "enabled": True,  # Always available as fallback
        "refresh_interval": 900,  # 15 minutes
        "num_articles": 10,
    }
}

# OpenAI configuration
OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "mock_mode": os.getenv("MOCK_EMBEDDING", "false").lower() == "true",
    "embedding_model": "text-embedding-3-small",
    "chat_model": "gpt-4o-mini",
}

# System configuration
SYSTEM_CONFIG = {
    "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.30")),
    "cache_size": int(os.getenv("CACHE_SIZE", "512")),
    "log_level": os.getenv("LOG_LEVEL", "INFO"),
}

def get_active_data_source() -> str:
    """Determine which data source is currently active."""
    if DATA_SOURCE_CONFIG["pathway"]["enabled"] and PATHWAY_AVAILABLE:
        return "pathway"
    else:
        return "mock"

def is_pathway_configured() -> bool:
    """Check if Pathway is properly configured."""
    if not PATHWAY_AVAILABLE:
        return False
        
    config = DATA_SOURCE_CONFIG["pathway"]
    if not config["enabled"]:
        return False
    
    if config["live_feed"] and not config["finnhub_token"]:
        return False
    
    return True

# Cache for mock data
_cache_data = None
_cache_time = 0
CACHE_DURATION = DATA_SOURCE_CONFIG["mock"]["refresh_interval"]  # 15 minutes

# Lock for thread safety
_cache_lock = threading.Lock()

def get_latest_rows(limit: int = 200) -> List[dict]:
    """
    Get latest news rows from either Pathway (real-time) or mock data.
    
    Args:
        limit: Maximum number of rows to return
        
    Returns:
        List of news articles with embeddings
    """
    global _cache_data, _cache_time
    
    active_source = get_active_data_source()
    logger.info(f"📡 Using data source: {active_source}")
    
    # Check if we should use Pathway for real-time data
    if active_source == "pathway":
        try:
            # Start Pathway if not already started
            start_pathway()
            
            # Get real-time data from Pathway
            pathway_data = pathway_get_latest_rows(limit)
            
            if pathway_data and len(pathway_data) > 0:
                logger.info(f"📡 Retrieved {len(pathway_data)} rows from Pathway (real-time)")
                
                # Ensure embeddings are present
                pathway_data = _ensure_embeddings(pathway_data)
                return pathway_data[:limit]
            else:
                logger.warning("🔄 Pathway returned no data, falling back to mock data")
        except Exception as e:
            logger.error(f"❌ Pathway error: {e}, falling back to mock data")
    
    # Fallback to existing mock data logic
    with _cache_lock:
        current_time = time.time()
        
        if _cache_data is None or (current_time - _cache_time) > CACHE_DURATION:
            logger.info("📰 Generating fresh mock news data...")
            _cache_data = _generate_fresh_news()
            _cache_time = current_time
            logger.info(f"📊 Updated cache with {len(_cache_data)} fresh rows")
    
    return _cache_data[:limit]

def _ensure_embeddings(rows: List[dict]) -> List[dict]:
    """
    Ensure all rows have embeddings, generate if missing.
    
    Args:
        rows: List of news articles
        
    Returns:
        List of articles with embeddings guaranteed
    """
    rows_with_embeddings = []
    
    for row in rows:
        # Check if embedding already exists
        if 'embedding' not in row or not row['embedding']:
            # Generate embedding from headline
            headline = row.get('headline', '')
            ticker = row.get('ticker', '')
            text_to_embed = f"{ticker} {headline}".strip()
            
            try:
                row['embedding'] = embed_text(text_to_embed)
                logger.debug(f"Generated embedding for: {text_to_embed[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to generate embedding for {text_to_embed}: {e}")
                continue
        
        rows_with_embeddings.append(row)
    
    return rows_with_embeddings

def _generate_fresh_news() -> List[dict]:
    """
    Generate fresh mock news data with embeddings.
    Enhanced with more realistic financial news scenarios.
    """
    base_time = datetime.now()
    
    # Enhanced mock news with more realistic scenarios
    mock_news = [
        {
            "ticker": "AAPL",
            "headline": "Apple reports record quarterly earnings, iPhone sales exceed expectations by 12%",
            "timestamp": (base_time - timedelta(hours=2)).isoformat(),
            "source": "Reuters",
            "category": "earnings",
            "sentiment": "positive"
        },
        {
            "ticker": "TSLA", 
            "headline": "Tesla delivers 500,000 vehicles in Q4, stock surges 8% in after-hours trading",
            "timestamp": (base_time - timedelta(hours=1)).isoformat(),
            "source": "Bloomberg",
            "category": "delivery",
            "sentiment": "positive"
        },
        {
            "ticker": "MSFT",
            "headline": "Microsoft Azure revenue grows 35% YoY, cloud dominance continues with enterprise deals",
            "timestamp": (base_time - timedelta(minutes=45)).isoformat(),
            "source": "CNBC",
            "category": "earnings",
            "sentiment": "positive"
        },
        {
            "ticker": "GOOGL",
            "headline": "Google announces breakthrough in quantum computing, partnerships with research institutions",
            "timestamp": (base_time - timedelta(minutes=30)).isoformat(),
            "source": "TechCrunch",
            "category": "innovation",
            "sentiment": "positive"
        },
        {
            "ticker": "NVDA",
            "headline": "NVIDIA partners with major automakers for next-gen AI chips, $2B deal announced",
            "timestamp": (base_time - timedelta(minutes=25)).isoformat(),
            "source": "MarketWatch",
            "category": "partnership", 
            "sentiment": "positive"
        },
        {
            "ticker": "META",
            "headline": "Meta's VR division shows promising growth, metaverse investments paying off",
            "timestamp": (base_time - timedelta(minutes=20)).isoformat(),
            "source": "The Verge",
            "category": "innovation",
            "sentiment": "positive"
        },
        {
            "ticker": "AMZN",
            "headline": "Amazon Web Services signs major cloud deal with government agencies, $3B contract",
            "timestamp": (base_time - timedelta(minutes=15)).isoformat(),
            "source": "Reuters",
            "category": "partnership",
            "sentiment": "positive"
        },
        {
            "ticker": "NFLX",
            "headline": "Netflix subscriber growth accelerates with new content strategy, international expansion",
            "timestamp": (base_time - timedelta(minutes=10)).isoformat(),
            "source": "Variety",
            "category": "growth",
            "sentiment": "positive"
        },
        {
            "ticker": "SPY",
            "headline": "S&P 500 reaches new all-time high amid strong earnings season and economic optimism",
            "timestamp": (base_time - timedelta(minutes=5)).isoformat(),
            "source": "MarketWatch",
            "category": "market",
            "sentiment": "positive"
        },
        {
            "ticker": "BTC",
            "headline": "Bitcoin surges past $45,000 as institutional adoption accelerates globally",
            "timestamp": (base_time - timedelta(minutes=2)).isoformat(),
            "source": "CoinDesk",
            "category": "crypto",
            "sentiment": "positive"
        }
    ]
    
    # Generate embeddings for all mock news
    logger.info("🧠 Generating embeddings for mock news...")
    rows_with_embeddings = []
    
    for news in mock_news:
        try:
            # Create text for embedding
            text_to_embed = f"{news['ticker']} {news['headline']}"
            news['embedding'] = embed_text(text_to_embed)
            rows_with_embeddings.append(news)
        except Exception as e:
            logger.warning(f"Failed to generate embedding for {news['headline']}: {e}")
    
    logger.info(f"✅ Generated embeddings for {len(rows_with_embeddings)}/{len(mock_news)} articles")
    return rows_with_embeddings

def get_news_by_ticker(ticker: str, limit: int = 50) -> List[dict]:
    """Get news filtered by ticker symbol."""
    all_rows = get_latest_rows()
    filtered = [row for row in all_rows if row.get('ticker', '').upper() == ticker.upper()]
    return filtered[:limit]

def get_news_by_category(category: str, limit: int = 50) -> List[dict]:
    """Get news filtered by category."""
    all_rows = get_latest_rows()
    filtered = [row for row in all_rows if row.get('category', '').lower() == category.lower()]
    return filtered[:limit]

def get_recent_news(hours: int = 24, limit: int = 100) -> List[dict]:
    """Get news from the last N hours."""
    cutoff_time = datetime.now() - timedelta(hours=hours)
    all_rows = get_latest_rows()
    
    recent = []
    for row in all_rows:
        try:
            timestamp_str = row.get('timestamp', '')
            if timestamp_str:
                # Parse timestamp
                if 'T' in timestamp_str:
                    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if dt.replace(tzinfo=None) > cutoff_time:
                        recent.append(row)
        except Exception as e:
            logger.warning(f"Error parsing timestamp {timestamp_str}: {e}")
            continue
    
    return recent[:limit]

def get_pipeline_stats() -> dict:
    """Get comprehensive pipeline statistics."""
    try:
        all_rows = get_latest_rows()
        
        # Basic counts
        total_articles = len(all_rows)
        articles_with_embeddings = len([r for r in all_rows if r.get('embedding')])
        
        # Ticker distribution
        tickers = [r.get('ticker', 'Unknown') for r in all_rows]
        unique_tickers = list(set(tickers))
        
        # Category distribution
        categories = [r.get('category', 'general') for r in all_rows]
        unique_categories = list(set(categories))
        
        # Source distribution
        sources = [r.get('source', 'Unknown') for r in all_rows]
        unique_sources = list(set(sources))
        
        # Data source info
        active_source = get_active_data_source()
        
        stats = {
            "total_articles": total_articles,
            "articles_with_embeddings": articles_with_embeddings,
            "embedding_coverage": articles_with_embeddings / total_articles if total_articles > 0 else 0,
            "unique_tickers": len(unique_tickers),
            "unique_categories": len(unique_categories),
            "unique_sources": len(unique_sources),
            "tickers": unique_tickers,
            "categories": unique_categories,
            "sources": unique_sources,
            "active_data_source": active_source,
            "pathway_available": PATHWAY_AVAILABLE,
            "pathway_configured": is_pathway_configured(),
            "cache_age_minutes": (time.time() - _cache_time) / 60 if _cache_time > 0 else 0
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting pipeline stats: {e}")
        return {"error": str(e)}

def refresh_cache():
    """Force refresh of the data cache."""
    global _cache_data, _cache_time
    
    with _cache_lock:
        logger.info("🔄 Forcing cache refresh...")
        _cache_data = None
        _cache_time = 0
        
        # Trigger fresh data load
        get_latest_rows()
        
    logger.info("✅ Cache refresh completed")

# Export main functions
__all__ = [
    "get_latest_rows",
    "get_news_by_ticker", 
    "get_news_by_category",
    "get_recent_news",
    "get_pipeline_stats",
    "refresh_cache",
    "get_active_data_source",
    "is_pathway_configured",
    "DATA_SOURCE_CONFIG",
    "OPENAI_CONFIG",
    "SYSTEM_CONFIG"
]