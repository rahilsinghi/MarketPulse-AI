"""
Enhanced Pathway streaming pipeline with monitoring, health checks,
and better error handling.
"""

import json
import os
import threading
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Iterator, List, Optional
from dataclasses import dataclass, asdict
import pickle

import pathway as pw

from utils import embed_text, validate_news_record

logger = logging.getLogger(__name__)

# ─── Configuration ─────────────────────────────────────────────────────────────
NEWS_FILE = Path(os.getenv("NEWS_SAMPLE_FILE", "data/sample_news.jsonl")).expanduser()
CHECKPOINT_FILE = Path("data/pipeline_checkpoint.pkl")
HEALTH_CHECK_INTERVAL = 30  # seconds

@dataclass
class PipelineStats:
    """Pipeline monitoring statistics."""
    total_processed: int = 0
    successful_embeddings: int = 0
    failed_embeddings: int = 0
    last_processed: Optional[datetime] = None
    pipeline_start_time: Optional[datetime] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

# Global pipeline statistics
pipeline_stats = PipelineStats()

# ─── Enhanced news generator with error handling ───────────────────────────────
def news_generator() -> Iterator[Dict[str, Any]]:
    """
    Enhanced news generator with checkpoint recovery and error handling.
    """
    global pipeline_stats
    
    if not NEWS_FILE.exists():
        logger.error(f"Sample news file not found: {NEWS_FILE}")
        # Create a dummy file for demo purposes
        NEWS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with NEWS_FILE.open('w') as f:
            f.write('{"id": "demo_1", "ticker": "DEMO", "headline": "Demo news for testing"}\n')
        logger.info(f"Created demo news file: {NEWS_FILE}")
    
    # Load checkpoint if exists
    start_line = 0
    if CHECKPOINT_FILE.exists():
        try:
            with CHECKPOINT_FILE.open('rb') as f:
                checkpoint_data = pickle.load(f)
                start_line = checkpoint_data.get('last_line', 0)
                logger.info(f"Resuming from line {start_line}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    
    pipeline_stats.pipeline_start_time = datetime.now()
    
    try:
        with NEWS_FILE.open() as f:
            # Skip to checkpoint
            for _ in range(start_line):
                next(f, None)
            
            line_num = start_line
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    
                    # Validate and enrich
                    obj.setdefault("timestamp", datetime.utcnow().isoformat())
                    obj.setdefault("id", f"{obj.get('ticker', 'UNK')}_{int(time.time()*1e6)}")
                    
                    if validate_news_record(obj):
                        pipeline_stats.total_processed += 1
                        pipeline_stats.last_processed = datetime.now()
                        
                        # Save checkpoint periodically
                        if line_num % 10 == 0:
                            save_checkpoint(line_num)
                        
                        yield obj
                        time.sleep(3)  # Simulate live feed
                        
                    else:
                        logger.warning(f"Invalid news record on line {line_num}: {obj}")
                        pipeline_stats.errors.append(f"Invalid record at line {line_num}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error on line {line_num}: {e}")
                    pipeline_stats.errors.append(f"JSON error at line {line_num}: {str(e)}")
                except Exception as e:
                    logger.error(f"Unexpected error processing line {line_num}: {e}")
                    pipeline_stats.errors.append(f"Processing error at line {line_num}: {str(e)}")
                
                line_num += 1
                
    except Exception as e:
        logger.error(f"Critical error in news generator: {e}")
        pipeline_stats.errors.append(f"Critical generator error: {str(e)}")

def save_checkpoint(line_num: int):
    """Save processing checkpoint."""
    try:
        CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with CHECKPOINT_FILE.open('wb') as f:
            pickle.dump({'last_line': line_num, 'timestamp': datetime.now()}, f)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

# ─── Enhanced embedding UDF with error handling ────────────────────────────────
@pw.udf
def safe_embed(headline: str) -> List[float]:
    """Pathway UDF with error handling and monitoring."""
    global pipeline_stats
    
    try:
        if not headline or not headline.strip():
            logger.warning("Empty headline provided for embedding")
            pipeline_stats.failed_embeddings += 1
            return [0.0] * 1536  # Return zero vector
        
        embedding = embed_text(headline)
        if embedding:
            pipeline_stats.successful_embeddings += 1
            logger.debug(f"Successfully embedded: {headline[:50]}...")
            return embedding
        else:
            pipeline_stats.failed_embeddings += 1
            logger.error(f"Failed to embed: {headline[:50]}...")
            return [0.0] * 1536
            
    except Exception as e:
        pipeline_stats.failed_embeddings += 1
        pipeline_stats.errors.append(f"Embedding error: {str(e)}")
        logger.error(f"Error in safe_embed: {e}")
        return [0.0] * 1536

# ─── Enhanced Pathway pipeline ─────────────────────────────────────────────────
try:
    # Define the streaming source
    news_src = pw.io.python.read(
        news_generator,
        schema=pw.schema_builder(
            id=str,
            ticker=str,
            headline=str,
            timestamp=str,
        ),
        mode="streaming"
    )

    # Add embeddings with error handling
    news_table = news_src.select(
        id=pw.this.id,
        ticker=pw.this.ticker,
        headline=pw.this.headline,
        timestamp=pw.this.timestamp,
        embedding=safe_embed(pw.this.headline),
    )
    
    logger.info("Pathway pipeline initialized successfully")
    
except Exception as e:
    logger.error(f"Failed to initialize Pathway pipeline: {e}")
    news_table = None

# ─── Enhanced data access with caching ─────────────────────────────────────────
_cache = {"data": [], "last_update": None, "cache_duration": 5}  # 5 second cache

def get_latest_rows(limit: int = 200, force_refresh: bool = False) -> List[Dict[str, Any]]:
    """
    Enhanced data access with caching and error handling.
    """
    global _cache
    
    try:
        # Check cache
        now = datetime.now()
        if (not force_refresh and 
            _cache["last_update"] and 
            (now - _cache["last_update"]).seconds < _cache["cache_duration"]):
            logger.debug("Returning cached data")
            return _cache["data"]
        
        if news_table is None:
            logger.warning("News table not initialized")
            return []
        
        # Convert Pathway table to pandas
        import pandas as pd
        df = pw.debug.table_to_pandas(news_table)
        
        if df.empty:
            logger.info("No data in news table yet")
            return []
        
        # Filter out zero embeddings (failed embeddings)
        df = df[df['embedding'].apply(lambda x: sum(x) != 0 if isinstance(x, list) else True)]
        
        # Sort by timestamp (newest first)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp", ascending=False).head(limit)
        
        # Update cache
        result = df.to_dict("records")
        _cache["data"] = result
        _cache["last_update"] = now
        
        logger.info(f"Retrieved {len(result)} rows from Pathway table")
        return result
        
    except Exception as e:
        logger.error(f"Error in get_latest_rows: {e}")
        pipeline_stats.errors.append(f"Data access error: {str(e)}")
        return []

# ─── Pipeline health monitoring ────────────────────────────────────────────────
def get_pipeline_health() -> Dict[str, Any]:
    """Get comprehensive pipeline health information."""
    global pipeline_stats
    
    health_data = asdict(pipeline_stats)
    
    # Add computed metrics
    if pipeline_stats.pipeline_start_time:
        uptime = datetime.now() - pipeline_stats.pipeline_start_time
        health_data["uptime_seconds"] = uptime.total_seconds()
        health_data["uptime_readable"] = str(uptime)
    
    # Embedding success rate
    total_attempts = pipeline_stats.successful_embeddings + pipeline_stats.failed_embeddings
    if total_attempts > 0:
        health_data["embedding_success_rate"] = pipeline_stats.successful_embeddings / total_attempts
    else:
        health_data["embedding_success_rate"] = 0.0
    
    # Recent activity check
    if pipeline_stats.last_processed:
        time_since_last = datetime.now() - pipeline_stats.last_processed
        health_data["minutes_since_last_processed"] = time_since_last.total_seconds() / 60
        health_data["is_active"] = time_since_last.total_seconds() < 300  # 5 minutes threshold
    else:
        health_data["is_active"] = False
        health_data["minutes_since_last_processed"] = None
    
    # Data availability
    try:
        current_rows = len(get_latest_rows(limit=1000))
        health_data["total_available_rows"] = current_rows
    except:
        health_data["total_available_rows"] = 0
    
    return health_data

def start_health_monitor():
    """Start health monitoring in background thread."""
    def monitor():
        while True:
            try:
                health = get_pipeline_health()
                if not health["is_active"] and health["total_processed"] > 0:
                    logger.warning("Pipeline appears inactive!")
                time.sleep(HEALTH_CHECK_INTERVAL)
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                time.sleep(HEALTH_CHECK_INTERVAL)
    
    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    return t

# ─── Enhanced pipeline startup ─────────────────────────────────────────────────
def _run_pipeline_blocking():
    """Run Pathway pipeline with enhanced error handling."""
    try:
        logger.info("Starting Pathway pipeline...")
        pw.run()
    except Exception as e:
        logger.error(f"Pipeline crashed: {e}")
        pipeline_stats.errors.append(f"Pipeline crash: {str(e)}")

def start_ingest():
    """Start enhanced pipeline with monitoring."""
    if news_table is None:
        logger.error("Cannot start pipeline - initialization failed")
        return None
    
    # Start pipeline
    pipeline_thread = threading.Thread(target=_run_pipeline_blocking, daemon=True)
    pipeline_thread.start()
    
    # Start health monitor
    health_thread = start_health_monitor()
    
    logger.info("Pipeline and health monitor started")
    return pipeline_thread, health_thread