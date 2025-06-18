import pathway as pw
import uuid
import json
import time
import threading
import os
import datetime
from websocket import create_connection

# Environment variables
USE_LIVE_FEED = os.getenv("USE_LIVE_FEED", "false").lower() == "true"
FINNHUB_TOKEN = os.getenv("FINNHUB_TOKEN", "")
SYMBOLS = os.getenv("SYMBOLS", "AAPL,TSLA,MSFT").split(",")

# Embedding function
@pw.udf
def _emb(text: str) -> list[float]:
    from marketpulse_ml import embed_text
    return embed_text(text)

# JSONL generator for demo mode
def _jsonl_generator():
    with open("data/sample_news.jsonl", "r") as f:
        for line in f:
            yield json.loads(line)
            time.sleep(3)

# WebSocket generator for live mode
def _ws_generator():
    while True:
        try:
            ws = create_connection(f"wss://ws.finnhub.io?token={FINNHUB_TOKEN}")
            for symbol in SYMBOLS:
                ws.send(json.dumps({"type": "subscribe", "symbol": symbol}))
            while True:
                msg = json.loads(ws.recv())
                if "data" in msg:
                    for event in msg["data"]:
                        yield {
                            "id": str(uuid.uuid4()),
                            "ticker": event["s"],
                            "headline": event["d"],
                            "timestamp": datetime.datetime.utcnow().isoformat(),
                        }
        except Exception as e:
            print(f"WebSocket error: {e}, reconnecting...")
            time.sleep(5)

# Select data source
generator = _ws_generator if USE_LIVE_FEED else _jsonl_generator

# Pathway table schema and ingestion
news_src = pw.io.python.read(
    generator,
    schema=pw.Schema(
        id=pw.ColumnType.STRING,
        ticker=pw.ColumnType.STRING,
        headline=pw.ColumnType.STRING,
        timestamp=pw.ColumnType.STRING,
    ),
)
news_table = news_src.select(
    id=pw.this.id,
    ticker=pw.this.ticker,
    headline=pw.this.headline,
    timestamp=pw.this.timestamp,
    embedding=_emb(pw.this.headline),
)

# Public API: Get latest rows
def get_latest_rows(limit=200):
    df = pw.debug.table_to_pandas(news_table)
    return df.sort_values("timestamp", ascending=False).head(limit).to_dict("records")

# Public API: Manual append
def manual_append(ticker, headline, source="Manual"):
    news_table += pw.Table.row(
        {
            "id": str(uuid.uuid4()),
            "ticker": ticker,
            "headline": headline,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "embedding": _emb(headline),
        }
    )

# Public API: Start Pathway
_pathway_started = False
def start_pathway():
    global _pathway_started
    if not _pathway_started:
        _pathway_started = True
        threading.Thread(target=pw.run, daemon=True).start()
