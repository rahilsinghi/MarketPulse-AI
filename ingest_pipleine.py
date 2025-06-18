"""
Pathway streaming pipeline.
Tails `data/sample_news.jsonl` (one JSON line every few seconds) to simulate
a live news feed.  Each headline is embedded in real-time and stored in a
Pathway table that other modules can query.
"""

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Iterator, List

import pathway as pw

from utils import embed_text

# ─── Config ────────────────────────────────────────────────────────────────────
NEWS_FILE = Path(os.getenv("NEWS_SAMPLE_FILE", "data/sample_news.jsonl")).expanduser()

# ─── Python generator that yields dicts ────────────────────────────────────────
def news_generator() -> Iterator[Dict[str, Any]]:
    """
    Tail the JSONL file forever, yielding one record at a time.
    Every 3 seconds we emit the next line – perfect for demo latency.
    """
    if not NEWS_FILE.exists():
        raise FileNotFoundError(f"Sample news file not found: {NEWS_FILE!s}")

    with NEWS_FILE.open() as f:
        for line in f:
            obj = json.loads(line.strip())
            # enrich with timestamp if missing
            obj.setdefault("timestamp", datetime.utcnow().isoformat())
            # ensure a unique ID
            obj.setdefault("id", f"{obj['ticker']}_{int(time.time()*1e6)}")
            yield obj
            time.sleep(3)  # simulate live feed


# ─── Pathway pipeline ──────────────────────────────────────────────────────────
@pw.udf  # Pathway user-defined function so it runs inside the pipeline
def _emb(headline: str) -> List[float]:
    return embed_text(headline)


# Define the streaming source
news_src = pw.io.python.read(
    news_generator,
    schema=pw.schema_builder(
        id=str,
        ticker=str,
        headline=str,
        timestamp=str,      # keep as ISO string for simplicity
    ),
)

# Add embeddings
news_table = news_src.select(
    id=pw.this.id,
    ticker=pw.this.ticker,
    headline=pw.this.headline,
    timestamp=pw.this.timestamp,
    embedding=_emb(pw.this.headline),
)

# ─── Helper exposed to other modules ───────────────────────────────────────────
def get_latest_rows(limit: int = 200):
    """
    Convert Pathway table -> pandas -> list[dict] ≤ `limit` newest rows.
    Used by utils.get_pathway_table().
    """
    import pandas as pd
    df = pw.debug.table_to_pandas(news_table)
    if df.empty:
        return []
    # newest first
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", ascending=False).head(limit)
    return df.to_dict("records")


# ─── Run pipeline in a background thread so Streamlit can keep its loop ────────
def _run_pipeline_blocking():
    pw.run()


def start_ingest():
    """Start Pathway pipeline in a daemon thread (called once at app start)."""
    t = threading.Thread(target=_run_pipeline_blocking, daemon=True)
    t.start()
    return t
