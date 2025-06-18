""" 
Real-time news ingestion pipeline using Pathway.

Creates a continuously updating stream of:
    id, ticker, headline, source, timestamp, embedding
and populates a vector store for retrieval.

Public helpers
--------------
start_pipeline():  kicks off ingestion in a daemon thread.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import openai
import pathway as pw
from pathway.io.python import ConnectorSubject

from .config import (EMBED_MODEL, FINNHUB_API_KEY, MODE, OPENAI_API_KEY)
from .vector_store import add_vector

# OpenAI client will be initialized in each function that needs it

# --------------------------------------------------------------------------- #
# 📰  Data model                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class NewsItem:
    """Canonical representation of a news article we care about."""
    id: int
    ticker: str
    headline: str
    source: str
    timestamp: float
    embedding: Sequence[float] | None = None

# --------------------------------------------------------------------------- #
# 🛠️  Custom connector (dummy & live)                                          #
# --------------------------------------------------------------------------- #
class NewsFeedSubject(ConnectorSubject):
    """
    Pathway connector feeding NewsItem rows.

    * dummy mode: emits synthetic headlines every 10 s.
    * live  mode: polls Finnhub `general` news every 60 s.
    """

    def __init__(self) -> None:
        super().__init__()
        self._seen_ids: set[int] = set()
        self._dummy_counter = 1
        self._last_poll = 0.0

    # --- Finnhub helpers ---------------------------------------------------- #
    def _poll_finnhub(self) -> list[dict]:
        import requests

        url = (
            "https://finnhub.io/api/v1/news"
            "?category=general"
            f"&token={FINNHUB_API_KEY}"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()        # list[dict]

    # --- Dummy generator ---------------------------------------------------- #
    def _make_dummy(self) -> dict:
        headline = (
            f"DummyCorp announces innovative product line #{self._dummy_counter}"
        )
        data = {
            "id": 10_000_000 + self._dummy_counter,
            "headline": headline,
            "source": "DemoWire",
            "datetime": int(time.time()),
            "symbol": "DUM"
        }
        self._dummy_counter += 1
        return data

    # --- Embedding helper --------------------------------------------------- #
    @staticmethod
    def _embed(text: str) -> list[float] | None:
        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            res = client.embeddings.create(model=EMBED_MODEL, input=text)
            return res.data[0].embedding          # list[float]
        except Exception as exc:                        # noqa: BLE001
            print("Embedding error:", exc)
            return None

    # ----------------------------------------------------------------------- #
    # 🚀  Main run loop                                                       #
    # ----------------------------------------------------------------------- #
    def run(self) -> None:                # noqa: D401 – Pathway required
        while True:
            if MODE == "live":
                if FINNHUB_API_KEY is None:
                    raise RuntimeError(
                        "MARKETPULSE_MODE=live but FINNHUB_API_KEY not set."
                    )
                if time.time() - self._last_poll >= 60:
                    try:
                        for art in self._poll_finnhub():
                            if art["id"] in self._seen_ids:
                                continue
                            self._seen_ids.add(art["id"])
                            self._send_row(art)
                        self._last_poll = time.time()
                    except Exception as exc:            # noqa: BLE001
                        print("Finnhub fetch error:", exc)
            else:  # dummy
                self._send_row(self._make_dummy())

            time.sleep(10 if MODE == "dummy" else 5)

    # ----------------------------------------------------------------------- #
    # 📤  Push a single row into Pathway & vector store                       #
    # ----------------------------------------------------------------------- #
    def _send_row(self, raw: dict) -> None:
        emb = self._embed(raw["headline"])
        row = NewsItem(
            id=raw["id"],
            ticker=raw.get("symbol", ""),
            headline=raw["headline"],
            source=raw.get("source", ""),
            timestamp=raw.get("datetime", time.time()),
            embedding=emb,
        )
        # 1️⃣  Inject into Pathway (for potential downstream analytics)
        self.next(**asdict(row))          # type: ignore[arg-type]

        # 2️⃣  Inject into global vector store for retrieval
        if emb is not None:
            add_vector(vector=np.asarray(emb, dtype=np.float32),
                       metadata=row.headline)

# --------------------------------------------------------------------------- #
# 🏗️  Build Pathway table & schema                                            #
# --------------------------------------------------------------------------- #
class NewsSchema(pw.Schema):
    id: int = pw.column_definition(primary_key=True)
    ticker: str
    headline: str
    source: str
    timestamp: float
    embedding: tuple | None


def _build_pipeline() -> None:
    """Create Pathway table (side-effect)."""
    pw.io.python.read(
        NewsFeedSubject(),
        schema=NewsSchema,
    )
    # No further processing for now – indexing is handled by vector_store.


def start_pipeline() -> None:
    """Launch Pathway engine in a daemon thread (idempotent)."""
    if getattr(start_pipeline, "_started", False):      # type: ignore[attr-defined]
        return
    start_pipeline._started = True                     # type: ignore[attr-defined]

    def _run() -> None:
        _build_pipeline()
        pw.run()

    threading.Thread(target=_run, daemon=True).start()