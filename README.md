# 💹 MarketPulse AI

**Real-time Financial News Analysis with AI-Powered Semantic Search**

A production-ready financial news analysis platform that combines OpenAI's latest AI models with intelligent document retrieval to provide real-time insights on market news.

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/fastapi-v0.100+-green.svg)
![OpenAI](https://img.shields.io/badge/openai-v1.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🚀 Features

### 🤖 **AI-Powered Intelligence**
- **Real OpenAI Integration**: Uses `text-embedding-3-small` for semantic embeddings and `gpt-4o-mini` for intelligent responses
- **Semantic Search**: Understands context and intent, not just keyword matching
- **Natural Language Q&A**: Ask questions in plain English about market trends, earnings, partnerships, and more
- **Context-Aware Responses**: AI provides relevant answers with proper source attribution

### 📰 **Real-Time News Processing**
- **Live Financial Data**: Processes news from Finnhub API or synthetic data for testing
- **Multi-Company Coverage**: Tracks major stocks and market indicators
- **Category Intelligence**: Automatically categorizes news by type and relevance
- **Sentiment Analysis**: Tracks market sentiment trends

### 🔍 **Advanced Retrieval System**
- **FAISS Vector Store**: High-performance similarity search with semantic embeddings
- **Intelligent Filtering**: Returns only relevant results with configurable similarity thresholds
- **Performance Optimization**: Efficient caching and indexing for fast responses
- **Scalable Architecture**: Ready for production deployment

### 💼 **Professional API Interface**
- **FastAPI Backend**: High-performance async API with automatic documentation
- **Real-Time Pipeline**: Continuous news ingestion with Pathway framework
- **RESTful Endpoints**: Clean API design for easy integration
- **Interactive Documentation**: Built-in Swagger UI for API exploration

## 🏗️ Architecture

### Backend Components
MarketPulse AI backend is built with the following core components:
- **Ingestion Pipeline**: Real-time news ingestion and processing with Pathway
- **Vector Store**: FAISS-backed embedding store for semantic search
- **LLM Integration**: GPT-4 answer generation with context awareness
- **API Service**: FastAPI-based REST API with async support

## 📁 File Structure

```
marketpulse/
├── __init__.py
├── api.py                   # 🚀 FastAPI service and endpoints
├── config.py               # ⚙️ Configuration management
├── ingestion.py            # 📰 Real-time news pipeline
├── llm.py                  # 🤖 GPT-4 integration
└── vector_store.py         # 🔍 FAISS vector store
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key
- (Optional) Finnhub API key for live news

### Installation

1. Copy `.env.example` to `.env` and set your API keys:
   ```bash
   cp .env.example .env
   # Edit .env and set OPENAI_API_KEY and optionally FINNHUB_API_KEY
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
- `MARKETPULSE_MODE`: `live` (default) or `dummy`

## 🔄 Data Pipeline

### News Ingestion (`ingestion.py`)
- Fetches financial news from Finnhub API or generates synthetic data
- Processes and filters news by relevance and recency
- Generates embeddings for semantic search

### Vector Storage (`vector_store.py`)
- FAISS-backed vector database for efficient similarity search
- Automatic indexing and retrieval optimization
- Configurable similarity thresholds

### AI Response Generation (`llm.py`)
- GPT-4 integration for intelligent question answering
- Context-aware prompt engineering
- Source attribution and relevance scoring

### API Service (`api.py`)
- FastAPI-based REST API with async support
- Automatic startup of ingestion pipeline
- Health checks and monitoring endpoints

## 📝 License

MIT
