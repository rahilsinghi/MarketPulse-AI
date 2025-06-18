# 💹 MarketPulse AI  
Real-Time Market Insight Chatbot powered by **Pathway** streaming RAG & GPT-4o

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Pathway](https://img.shields.io/badge/Streaming-Pathway-brightgreen)](https://github.com/pathwaycom/pathway)
[![OpenAI](https://img.shields.io/badge/LLM-GPT-4o-lightgrey)](https://platform.openai.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](#license)

> **Pitch (30 sec)**  
> Traders bleed alpha when breaking news hits Twitter before their dashboards.  
> **MarketPulse AI** ingests live headlines & price alerts, re-indexes them in-memory with  
> **Pathway**, and lets you ask *“Why is TSLA spiking right now?”* — the answer cites the update that landed seconds ago.

---

## ✨ Features
| Real-time | What it does |
|-----------|--------------|
| **Streaming ingest** | Tails a WebSocket or demo JSON-L file, embedding each headline on arrival. |
| **Live RAG** | Pathway’s vector index updates **instantly** – no rebuilds. |
| **Chat UI** | Streamlit frontend with latency badge, “NEW” highlight, and live headline sidebar. |
| **One-click demo** | Works offline with sample data *or* flips to Finnhub API in prod. |
| **REST API** (optional) | `/ask` endpoint for other front-ends or Slack bots. |

---

## 📦 Quick Start (3 min)

```bash
# 1. Clone & enter repo
git clone https://github.com/YOUR-ORG/MarketPulse-AI.git
cd MarketPulse-AI

# 2. Create .env from template and add keys
cp .env.example .env
#   └── OPENAI_API_KEY=sk-...
#   └── FINNHUB_TOKEN=<optional>

# 3. Install deps & launch
pip install -r requirements.txt
streamlit run app.py          # open http://localhost:8501
