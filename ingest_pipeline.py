"""
MarketPulse AI - Enhanced Ingest Pipeline
Updated to work with the new retrieval engine.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import threading
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread safety
_data_lock = threading.Lock()
_cached_rows = []
_last_update = None
CACHE_DURATION = 60  # Cache for 60 seconds

def get_latest_rows() -> List[Dict]:
    """
    Enhanced implementation of get_latest_rows().
    Returns fresh financial news data with embeddings.
    
    In production, this would:
    - Connect to real RSS feeds (Reuters, Bloomberg, Yahoo Finance)
    - Parse and normalize news data
    - Generate embeddings for each article
    - Cache results for performance
    """
    global _cached_rows, _last_update
    
    with _data_lock:
        # Check if we have fresh cached data
        now = time.time()
        if (_last_update and 
            _cached_rows and 
            now - _last_update < CACHE_DURATION):
            logger.debug(f"Returning cached data ({len(_cached_rows)} rows)")
            return _cached_rows.copy()
        
        logger.info("Fetching fresh news data...")
        
        # Generate fresh mock data (in production, this would fetch real data)
        fresh_rows = _generate_mock_news_data()
        
        # Add embeddings to each row
        fresh_rows = _add_embeddings_to_rows(fresh_rows)
        
        # Update cache
        _cached_rows = fresh_rows
        _last_update = now
        
        logger.info(f"Updated cache with {len(fresh_rows)} fresh rows")
        return fresh_rows.copy()

def _generate_mock_news_data() -> List[Dict]:
    """
    Generate realistic mock financial news data.
    In production, this would be replaced with real RSS/API feeds.
    """
    base_time = datetime.now()
    
    # Expanded mock news data with more variety
    mock_news = [
        {
            "id": "AAPL_001",
            "ticker": "AAPL",
            "headline": "Apple reports record quarterly earnings, beating analyst expectations by 15%",
            "timestamp": (base_time - timedelta(hours=2)).isoformat(),
            "source": "Reuters",
            "category": "earnings",
            "sentiment": "positive"
        },
        {
            "id": "TSLA_001",
            "ticker": "TSLA", 
            "headline": "Tesla delivers 500,000 vehicles in Q4, stock surges 8% in after-hours trading",
            "timestamp": (base_time - timedelta(hours=1)).isoformat(),
            "source": "Bloomberg",
            "category": "delivery",
            "sentiment": "positive"
        },
        {
            "id": "GOOGL_001",
            "ticker": "GOOGL",
            "headline": "Google announces breakthrough in quantum computing, Alphabet shares jump 12%",
            "timestamp": (base_time - timedelta(minutes=45)).isoformat(),
            "source": "TechCrunch",
            "category": "innovation",
            "sentiment": "positive"
        },
        {
            "id": "MSFT_001",
            "ticker": "MSFT",
            "headline": "Microsoft Azure revenue grows 35% YoY, cloud dominance continues",
            "timestamp": (base_time - timedelta(hours=3)).isoformat(),
            "source": "CNBC",
            "category": "earnings",
            "sentiment": "positive"
        },
        {
            "id": "NVDA_001",
            "ticker": "NVDA",
            "headline": "NVIDIA partners with major automakers for next-gen AI chips",
            "timestamp": (base_time - timedelta(minutes=20)).isoformat(),
            "source": "MarketWatch",
            "category": "partnership",
            "sentiment": "positive"
        },
        {
            "id": "META_001",
            "ticker": "META",
            "headline": "Meta's VR division shows promising growth, metaverse investments paying off",
            "timestamp": (base_time - timedelta(hours=4)).isoformat(),
            "source": "The Verge",
            "category": "earnings",
            "sentiment": "positive"
        },
        {
            "id": "AMZN_001",
            "ticker": "AMZN",
            "headline": "Amazon Web Services signs major cloud deal with government agency",
            "timestamp": (base_time - timedelta(minutes=10)).isoformat(),
            "source": "Reuters",
            "category": "contract",
            "sentiment": "positive"
        },
        {
            "id": "NFLX_001",
            "ticker": "NFLX",
            "headline": "Netflix subscriber growth accelerates with new content strategy",
            "timestamp": (base_time - timedelta(hours=5)).isoformat(),
            "source": "Variety",
            "category": "growth",
            "sentiment": "positive"
        },
        {
            "id": "SPY_001",
            "ticker": "SPY",
            "headline": "S&P 500 reaches new all-time high amid strong earnings season",
            "timestamp": (base_time - timedelta(minutes=30)).isoformat(),
            "source": "MarketWatch",
            "category": "market",
            "sentiment": "positive"
        },
        {
            "id": "BTC_001",
            "ticker": "BTC",
            "headline": "Bitcoin approaches $50,000 as institutional adoption increases",
            "timestamp": (base_time - timedelta(hours=6)).isoformat(),
            "source": "CoinDesk",
            "category": "crypto",
            "sentiment": "positive"
        }
    ]
    
    return mock_news

def _add_embeddings_to_rows(rows: List[Dict]) -> List[Dict]:
    """
    Add embeddings to news rows using the retrieval engine.
    """
    try:
        from retrieval_engine import embed_text
    except ImportError:
        logger.warning("retrieval_engine not available, using fallback embeddings")
        return _add_fallback_embeddings(rows)
    
    enhanced_rows = []
    for row in rows:
        try:
            # Create embedding text from headline and ticker
            embed_text_content = f"{row['ticker']} {row['headline']}"
            
            # Generate embedding
            embedding = embed_text(embed_text_content)
            
            # Add embedding to row
            enhanced_row = row.copy()
            enhanced_row['embedding'] = embedding
            enhanced_row['embed_text'] = embed_text_content
            
            enhanced_rows.append(enhanced_row)
            
        except Exception as e:
            logger.warning(f"Failed to generate embedding for {row.get('id', 'unknown')}: {e}")
            # Add row without embedding
            enhanced_rows.append(row)
    
    logger.info(f"Added embeddings to {len([r for r in enhanced_rows if 'embedding' in r])}/{len(rows)} rows")
    return enhanced_rows

def _add_fallback_embeddings(rows: List[Dict]) -> List[Dict]:
    """
    Fallback embedding generation when retrieval_engine is not available.
    """
    import numpy as np
    
    enhanced_rows = []
    for row in rows:
        embed_text_content = f"{row['ticker']} {row['headline']}"
        
        # Generate deterministic fallback embedding
        np.random.seed(hash(embed_text_content) % 2**32)
        embedding = np.random.rand(1536).astype(float).tolist()
        
        enhanced_row = row.copy()
        enhanced_row['embedding'] = embedding
        enhanced_row['embed_text'] = embed_text_content
        
        enhanced_rows.append(enhanced_row)
    
    return enhanced_rows

def get_news_by_ticker(ticker: str) -> List[Dict]:
    """
    Get news filtered by specific ticker symbol.
    """
    all_rows = get_latest_rows()
    filtered_rows = [row for row in all_rows if row.get('ticker') == ticker.upper()]
    
    logger.info(f"Found {len(filtered_rows)} articles for ticker {ticker}")
    return filtered_rows

def get_news_by_category(category: str) -> List[Dict]:
    """
    Get news filtered by category (earnings, partnership, innovation, etc.).
    """
    all_rows = get_latest_rows()
    filtered_rows = [row for row in all_rows if row.get('category') == category.lower()]
    
    logger.info(f"Found {len(filtered_rows)} articles for category {category}")
    return filtered_rows

def get_recent_news(hours: int = 24) -> List[Dict]:
    """
    Get news from the last N hours.
    """
    all_rows = get_latest_rows()
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    recent_rows = []
    for row in all_rows:
        try:
            timestamp = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
            if timestamp.replace(tzinfo=None) > cutoff_time:
                recent_rows.append(row)
        except Exception as e:
            logger.warning(f"Invalid timestamp in row {row.get('id', 'unknown')}: {e}")
    
    logger.info(f"Found {len(recent_rows)} articles from last {hours} hours")
    return recent_rows

def refresh_cache():
    """
    Force refresh of the news cache.
    """
    global _last_update
    with _data_lock:
        _last_update = None  # Force refresh on next call
    
    logger.info("Cache refresh forced")
    return get_latest_rows()

def get_pipeline_stats() -> Dict:
    """
    Get statistics about the current pipeline state.
    """
    rows = get_latest_rows()
    
    # Count by ticker
    ticker_counts = {}
    category_counts = {}
    sentiment_counts = {}
    
    for row in rows:
        ticker = row.get('ticker', 'Unknown')
        category = row.get('category', 'unknown')
        sentiment = row.get('sentiment', 'neutral')
        
        ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    
    # Calculate timing stats
    now = datetime.now()
    times_ago = []
    
    for row in rows:
        try:
            timestamp = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
            hours_ago = (now - timestamp.replace(tzinfo=None)).total_seconds() / 3600
            times_ago.append(hours_ago)
        except:
            continue
    
    stats = {
        'total_articles': len(rows),
        'articles_with_embeddings': len([r for r in rows if 'embedding' in r]),
        'unique_tickers': len(ticker_counts),
        'ticker_distribution': ticker_counts,
        'category_distribution': category_counts,
        'sentiment_distribution': sentiment_counts,
        'avg_hours_ago': sum(times_ago) / len(times_ago) if times_ago else 0,
        'cache_age_seconds': time.time() - _last_update if _last_update else 0
    }
    
    return stats

# For backward compatibility
def get_mock_news_data():
    """Legacy function name - redirects to get_latest_rows()"""
    logger.warning("get_mock_news_data() is deprecated, use get_latest_rows() instead")
    return get_latest_rows()

# Production-ready functions (stubs for future implementation)
def setup_rss_feeds(feeds: List[str]):
    """
    Setup RSS feeds for real-time news ingestion.
    TODO: Implement in production version.
    """
    logger.info(f"RSS feeds setup requested: {feeds}")
    raise NotImplementedError("RSS feeds not implemented in demo version")

def setup_news_apis(api_keys: Dict[str, str]):
    """
    Setup news API connections (NewsAPI, Alpha Vantage, etc.).
    TODO: Implement in production version.
    """
    logger.info(f"News APIs setup requested: {list(api_keys.keys())}")
    raise NotImplementedError("News APIs not implemented in demo version")

if __name__ == "__main__":
    # Test the enhanced pipeline
    print("🧪 Testing Enhanced Ingest Pipeline")
    print("=" * 40)
    
    # Test basic functionality
    print("\n📰 Fetching latest rows...")
    rows = get_latest_rows()
    print(f"✅ Got {len(rows)} articles")
    
    # Test embeddings
    print(f"✅ {len([r for r in rows if 'embedding' in r])} articles have embeddings")
    
    # Test filtering
    print("\n🔍 Testing filters...")
    aapl_news = get_news_by_ticker("AAPL")
    print(f"✅ AAPL news: {len(aapl_news)} articles")
    
    earnings_news = get_news_by_category("earnings")
    print(f"✅ Earnings news: {len(earnings_news)} articles")
    
    recent_news = get_recent_news(hours=2)
    print(f"✅ Recent news (2h): {len(recent_news)} articles")
    
    # Test stats
    print("\n📊 Pipeline statistics...")
    stats = get_pipeline_stats()
    print(f"✅ Total articles: {stats['total_articles']}")
    print(f"✅ Unique tickers: {stats['unique_tickers']}")
    print(f"✅ Average age: {stats['avg_hours_ago']:.1f} hours")
    
    print("\n🎉 Enhanced pipeline test completed!")