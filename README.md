# 💹 MarketPulse AI  
Real-time financial news Q&A powered by **Pathway** streaming RAG + GPT-4o

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org) 
[![Pathway](https://img.shields.io/badge/Streaming-Pathway-brightgreen)](https://github.com/pathwaycom/pathway) 
[![OpenAI](https://img.shields.io/badge/LLM-GPT-4o-lightgrey)](https://platform.openai.com)  

> **Pitch (20 sec)**  
> Markets move in seconds; static chatbots miss the story.  
> **MarketPulse AI** streams headlines into a live vector index via **Pathway**, so you can ask  
> “Why is TSLA spiking right now?” and get an answer citing the bulletin that landed moments ago.

---

## ✨ Features

| Category | Details |
|----------|---------|
| **Real-time ingest** | Pathway pipeline tails a JSONL feed *(offline demo)* or a Finnhub WebSocket *(online)*. |
| **Adaptive RAG** | New headlines are embedded (OpenAI `text-embedding-3-small`) and instantly queryable. |
| **Chat UI** | Streamlit chat with latency badge & “NEW” highlight for < 5 min headlines. |
| **Semantic search** | Cosine similarity ≥ 0.30 filters noise; FAISS ready for > 1 k docs. |
| **Smart mock mode** | Runs fully offline – deterministic embeddings, sample feed. |
| **Tests** | `pytest` suite covers embeddings, retrieval, end-to-end Q&A. |

---

## 🚀 Quick Start (local)

```bash
git clone https://github.com/<your-org>/MarketPulse-AI.git
cd MarketPulse-AI
python3 -m venv venv && source venv/bin/activate   # Win: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # edit keys as needed
streamlit run app.py            # open http://localhost:8501
```

### `.env` keys

| Var | Example | Notes |
|-----|---------|-------|
| `OPENAI_API_KEY` | `sk-…` | required for real embeddings & answers |
| `USE_LIVE_FEED` | `false` | `true` → Finnhub WS, `false` → demo JSONL |
| `FINNHUB_TOKEN` | `YOUR_TOKEN` | only needed if live feed |
| `SYMBOLS` | `AAPL,TSLA,MSFT` | comma list for WS subscribe |
| `LOG_LEVEL` | `INFO` | `DEBUG` prints Pathway deltas |

---

## 🖥️ How to Demo

1. Launch app (JSONL mode).  
2. **Sidebar** shows a new headline every ~3 s.  
3. Ask **“Why is ACME up?”** → bot replies “No recent news.”  
4. Click **“Inject Test Headline”** (or run `manual_append()` in a shell).  
5. Re-ask question → answer cites the fresh bullet with a **NEW** badge.  
6. Latency badge should read < 5 s for a cold query, < 1 s when cached.

---

## 🏗️ Architecture

```text
┌──────── Demo JSONL ───────┐
│ or Finnhub WebSocket      │
└─────────────┬─────────────┘
              │ ① ingest ② embed
       pathway_pipeline.py
              │ (live table + vector index)
              ▼
     marketpulse_ml.retrieval
              │ ③ top-k docs
              ▼
    GPT-4o mini completion
              │
      ┌──── Streamlit UI ────┐
      │ chat + sidebar feed  │
      └──────────────────────┘
```

*Technical notes*  
* Embedding UDF runs inside Pathway; cached via LRU to spare tokens.  
* Retrieval falls back to linear cosine; flips to FAISS automatically > 1 k docs.  
* Prompt design & threshold rationale in [`docs/prompt_design.md`](docs/prompt_design.md).

---

## 🧪 Tests

```bash
pytest -q            # all green
```

| File | Scope |
|------|-------|
| `tests/test_query_agent.py` | prompt + answer path |
| `tests/test_retrieval_engine.py` | embeddings, similarity, FAISS fallback |
| `tests/test_complete_pipeline.py` | Pathway ingest ➜ chat answer |

---

## ⚙️ Configuration Flags

| Flag | Location | Effect |
|------|----------|--------|
| `SIMILARITY_THRESHOLD` | `.env` (default `0.30`) | tighten/loosen retrieval filter |
| `MAX_QPS` | `.env` | simple rate-limit guard for OpenAI calls |
| `MOCK_EMBEDDING` | `.env` (`true/false`) | force deterministic numpy vector (no API) |

---

## 📈 Performance Benchmarks

| Path | P95 latency | Note |
|------|-------------|------|
| Cached query | **0.1 – 0.3 s** | embedding hits LRU |
| Cold query (JSONL) | **1.5 – 3 s** | embedding + GPT |
| Cold query (WS) | **2 – 5 s** | network variance |
| Retrieval 1 k docs (FAISS) | **< 20 ms** | cosine search |

---

## ➕ Planned Enhancements

* Live price-tick feed with delta explanations.  
* Prometheus metrics panel (ingest → index lag).  
* SSE / WebSocket streaming answers.  
* Auto-tune similarity threshold with feedback loop.

---

## 📝 License

MIT — see `LICENSE`.

*Made with ❤️ by the MarketPulse AI team for HackWithNewYork 2025.*
