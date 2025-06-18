# MarketPulse AI

Real-time financial news analysis with GPT-4. This backend service provides:

- Real-time news ingestion (synthetic or Finnhub feed)
- Vector-based semantic search (FAISS)
- Question answering with GPT-4
- REST API with FastAPI

## 🚀 Quick Start

1. Copy `.env.example` to `.env` and set your OpenAI API key:
   ```bash
   cp .env.example .env
   # Edit .env and set OPENAI_API_KEY
   ```

2. Run with Docker Compose:
   ```bash
   docker compose up --build
   ```

3. Visit `http://localhost:8000/docs` for interactive API documentation.

## 💡 Usage

The service exposes a single endpoint:

- `POST /ask`
  ```json
  {
    "text": "What are the latest developments in tech?"
  }
  ```

## ⚙️ Configuration

Environment variables:

- `OPENAI_API_KEY` (required): Your OpenAI API key
- `FINNHUB_API_KEY` (optional): For live news feed
- `MARKETPULSE_MODE`: `dummy` (default) or `live`

## 🏗️ Architecture

- `ingestion.py`: Real-time news pipeline with Pathway
- `vector_store.py`: FAISS-backed embedding store
- `llm.py`: GPT-4 answer generation
- `api.py`: FastAPI service

## 📝 License

MIT