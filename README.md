# 💹 MarketPulse AI

**Real-time Financial News Analysis with AI-Powered Semantic Search**

A production-ready financial news analysis platform that combines OpenAI's latest AI models with intelligent document retrieval to provide real-time insights on market news.

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![OpenAI](https://img.shields.io/badge/openai-v1.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🚀 Features

### 🤖 **AI-Powered Intelligence**
- **Real OpenAI Integration**: Uses `text-embedding-3-small` for semantic embeddings and `gpt-4o-mini` for intelligent responses
- **Semantic Search**: Understands context and intent, not just keyword matching
- **Natural Language Q&A**: Ask questions in plain English about market trends, earnings, partnerships, and more
- **Context-Aware Responses**: AI provides relevant answers with proper source attribution

### 📰 **Real-Time News Processing**
- **Live Financial Data**: Processes news from major sources (Reuters, Bloomberg, CNBC, MarketWatch)
- **Multi-Company Coverage**: Tracks AAPL, TSLA, GOOGL, MSFT, NVDA, META, AMZN, NFLX, SPY, BTC, and more
- **Category Intelligence**: Automatically categorizes news (earnings, partnerships, innovation, deliveries, market trends)
- **Sentiment Analysis**: Tracks positive, negative, and neutral sentiment trends

### 🔍 **Advanced Retrieval System**
- **Similarity Scoring**: Ranks documents by semantic relevance (0.0-1.0 scale)
- **Intelligent Filtering**: Returns only relevant results (similarity > 0.25 threshold)
- **Performance Optimization**: LRU caching prevents duplicate API calls
- **Scalable Architecture**: Ready for FAISS integration for large-scale deployment

### 💼 **Professional Web Interface**
- **Interactive Dashboard**: Professional Streamlit-based interface
- **Real-Time Chat**: AI-powered conversational interface for market queries
- **Analytics Dashboard**: Comprehensive news distribution and sentiment analytics
- **Debug Tools**: Built-in testing and debugging capabilities for system monitoring

## 🏗️ Architecture

### Core Components
MarketPulse AI is built with the following core components:
- **Retrieval Engine**: AI-powered semantic search and document ranking.
- **Ingestion Pipeline**: Real-time news ingestion and processing.
- **User Interface**: Interactive Streamlit-based dashboard for analytics and chat.

## 📁 File Structure

```
MarketPulse-AI/
├── retrieval_engine.py      # 🧠 Core AI retrieval system
├── ingest_pipeline.py       # 📰 News data ingestion & processing
├── app.py                   # 💼 Streamlit-based user interface
├── requirements.txt         # 📦 Python dependencies
├── .env                     # 🔒 Environment variables
└── README.md                # 📖 Project documentation
```

## 🔧 Key Functions

### `embed_text(text: str) -> List[float]`
- Converts text to 1536-dimensional semantic vectors.
- Uses OpenAI's `text-embedding-3-small` model.
- **Features**:
  - LRU cache (256 entries) prevents duplicate API calls.
  - Deterministic fallback for testing.

### `retrieve_top_k(query_vec: List[float], k: int) -> List[dict]`
- Performs semantic search using cosine similarity.
- Returns ranked results with similarity scores.
- **Features**:
  - Scalable architecture (linear search → FAISS for 1000+ docs).
  - Thread-safe operation.

### `answer_query_sync(question: str, top_k: int) -> str`
- Complete Q&A pipeline with timing.
- **Pipeline**:
  - Semantic retrieval → Context building → AI response.
- **Features**:
  - Intelligent fallbacks and error handling.
  - Performance monitoring.

## 🔄 Data Pipeline

### News Ingestion (`ingest_pipeline.py`)
- Fetches financial news from multiple sources.
- Generates embeddings for each article.
- **Features**:
  - Thread-safe caching with automatic refresh.
  - Filtering by ticker, category, and time.

### Semantic Processing (`retrieval_engine.py`)
- Converts user queries to embedding vectors.
- Performs similarity search across the news database.
- Filters and ranks results by relevance.

### AI Response Generation
- Builds structured prompts with context.
- Calls OpenAI `GPT-4o-mini` for intelligent responses.
- Handles rate limiting and error recovery.

### User Interface (`app.py`)
- **Features**:
  - Real-time chat interface.
  - News browsing and filtering.
  - Analytics and system monitoring.
  - Debug tools and testing interface.

## 📊 Demo Capabilities

### 💬 AI Chat Examples
- Ask questions about market trends, earnings, partnerships, and more.

### 📈 Analytics Dashboard
- **Company Coverage**: Visual distribution of news across different tickers.
- **Category Analysis**: Breakdown by earnings, partnerships, innovation, etc.
- **Sentiment Tracking**: Positive/negative/neutral sentiment trends.
- **Performance Metrics**: System performance and cache statistics.

### 🔧 Debug Interface
- **Embedding Testing**: Test AI embedding generation with custom text.
- **Search Testing**: Test semantic document retrieval with similarity scores.
- **System Information**: Environment status, API connectivity, performance metrics.
- **Quick AI Test**: One-click system functionality verification.

## 🎯 Production Features

### ⚡ Performance Optimizations
- **Caching Strategy**: LRU cache for embeddings, time-based cache for news.
- **Async Operations**: Non-blocking API calls for better responsiveness.
- **Connection Management**: Proper client lifecycle management.
- **Memory Efficiency**: Optimized data structures and cleanup.

### 🛡️ Reliability & Error Handling
- **Graceful Fallbacks**: Mock responses when APIs are unavailable.
- **Rate Limit Handling**: Intelligent backoff and retry mechanisms.
- **Connection Recovery**: Automatic retry with exponential backoff.
- **Input Validation**: Robust handling of edge cases and malformed data.

## 🔒 Security & Configuration
- **Environment Variables**: Secure API key management.
- **Input Sanitization**: Safe handling of user inputs.
- **Error Logging**: Comprehensive logging without exposing sensitive data.
- **Configurable Modes**: Easy switching between production and development.

## ⚡ Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/MarketPulse-AI.git
cd MarketPulse-AI

# Create virtual environment
python3.11 -m venv venv_py311
source venv_py311/bin/activate  # On Windows: venv_py311\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "OPENAI_API_KEY=your-api-key-here" > .env
echo "MOCK_EMBEDDING=false" >> .env

# Start the Streamlit app
streamlit run app.py

# The app will open in your browser at http://localhost:8501
```

## 🤝 Contributing

We welcome contributions to MarketPulse AI! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m "Add feature-name"`.
4. Push to your branch: `git push origin feature-name`.
5. Open a pull request on GitHub.

Please ensure your code follows the project's coding standards and includes appropriate tests.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🛠️ Support

If you encounter any issues or have questions, feel free to:

- Open an issue on the [GitHub repository](https://github.com/your-username/MarketPulse-AI/issues).
- Contact us via email at support@marketpulseai.com.

We appreciate your feedback and contributions!