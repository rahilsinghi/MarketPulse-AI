"""
MarketPulse AI - Main Streamlit Application
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configuration
st.set_page_config(
    page_title="MarketPulse AI",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mock embedding function (deterministic)
@st.cache_data
def mock_embed_text(text: str):
    """Create deterministic mock embeddings for testing."""
    np.random.seed(hash(text) % 2**32)
    return np.random.rand(1536).tolist()

# Mock news data
@st.cache_data
def get_mock_news_data():
    """Get mock news data for testing."""
    base_time = datetime.now()
    
    news_data = [
        {
            "id": "AAPL_001",
            "ticker": "AAPL",
            "headline": "Apple reports record Q4 earnings, beating analyst expectations by 15%",
            "timestamp": (base_time - timedelta(hours=2)).isoformat(),
            "source": "Reuters",
            "embedding": mock_embed_text("Apple reports record Q4 earnings, beating analyst expectations by 15%")
        },
        {
            "id": "TSLA_001", 
            "ticker": "TSLA",
            "headline": "Tesla delivers 500,000 vehicles in Q4, stock surges 8% in after-hours trading",
            "timestamp": (base_time - timedelta(hours=1)).isoformat(),
            "source": "Bloomberg",
            "embedding": mock_embed_text("Tesla delivers 500,000 vehicles in Q4, stock surges 8% in after-hours trading")
        },
        {
            "id": "GOOGL_001",
            "ticker": "GOOGL", 
            "headline": "Google announces breakthrough in quantum computing, Alphabet shares jump 12%",
            "timestamp": (base_time - timedelta(minutes=45)).isoformat(),
            "source": "TechCrunch",
            "embedding": mock_embed_text("Google announces breakthrough in quantum computing, Alphabet shares jump 12%")
        },
        {
            "id": "MSFT_001",
            "ticker": "MSFT",
            "headline": "Microsoft Azure revenue grows 35% YoY, cloud dominance continues",
            "timestamp": (base_time - timedelta(hours=3)).isoformat(),
            "source": "CNBC",
            "embedding": mock_embed_text("Microsoft Azure revenue grows 35% YoY, cloud dominance continues")
        },
        {
            "id": "NVDA_001",
            "ticker": "NVDA",
            "headline": "NVIDIA partners with major automakers for next-gen AI chips",
            "timestamp": (base_time - timedelta(minutes=20)).isoformat(),
            "source": "MarketWatch", 
            "embedding": mock_embed_text("NVIDIA partners with major automakers for next-gen AI chips")
        },
        {
            "id": "META_001",
            "ticker": "META",
            "headline": "Meta's VR division shows promising growth, metaverse investments paying off",
            "timestamp": (base_time - timedelta(hours=4)).isoformat(),
            "source": "The Verge",
            "embedding": mock_embed_text("Meta's VR division shows promising growth, metaverse investments paying off")
        }
    ]
    
    return news_data

# Similarity calculation
def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Retrieval function
def retrieve_relevant_docs(question: str, data: list, top_k: int = 3):
    """Retrieve most relevant documents for a question."""
    question_embedding = mock_embed_text(question)
    
    # Calculate similarities
    scored_docs = []
    for doc in data:
        similarity = cosine_similarity(question_embedding, doc['embedding'])
        scored_docs.append({
            **doc,
            'similarity': similarity
        })
    
    # Sort by similarity and return top-k
    scored_docs.sort(key=lambda x: x['similarity'], reverse=True)
    return scored_docs[:top_k]

# Simple Q&A function
def answer_question(question: str, relevant_docs: list) -> str:
    """Generate answer based on relevant documents."""
    if not relevant_docs:
        return "I couldn't find any relevant news for your question."
    
    # Create context from relevant documents
    context_parts = []
    for doc in relevant_docs:
        timestamp = datetime.fromisoformat(doc['timestamp'])
        time_ago = datetime.now() - timestamp
        
        if time_ago.total_seconds() < 3600:  # Less than 1 hour
            time_str = f"{int(time_ago.total_seconds() / 60)} minutes ago"
        else:
            time_str = f"{int(time_ago.total_seconds() / 3600)} hours ago"
        
        context_parts.append(
            f"**[{doc['ticker']}]** {doc['headline']} "
            f"*({time_str}, {doc['source']}, relevance: {doc['similarity']:.2f})*"
        )
    
    context = "\n\n".join(context_parts)
    
    # Simple response generation
    tickers = [doc['ticker'] for doc in relevant_docs]
    main_ticker = max(set(tickers), key=tickers.count)
    
    response = f"Based on recent market news about **{main_ticker}** and related stocks:\n\n{context}\n\n"
    
    # Add simple analysis
    if "earnings" in question.lower():
        response += "💰 **Analysis**: This appears to be related to earnings reports and financial performance."
    elif "stock" in question.lower() or "price" in question.lower():
        response += "📈 **Analysis**: This involves stock price movements and market reactions."
    elif any(word in question.lower() for word in ["news", "what", "happening"]):
        response += "📰 **Analysis**: Here's the latest news that might be relevant to your query."
    else:
        response += "🔍 **Analysis**: This information should help answer your question about recent market activity."
    
    return response

# Main app
def main():
    # Get news data
    news_data = get_mock_news_data()
    
    # Sidebar
    with st.sidebar:
        st.title("🔧 Configuration")
        
        # Data source info
        st.subheader("📊 Data Source")
        st.info("Currently using mock data for demonstration. In production, this would connect to real-time news feeds.")
        
        # System status
        st.subheader("⚡ System Status")
        st.success("✅ Core system operational")
        st.success("✅ Mock data loaded")
        
        # Stats
        st.metric("Total News Items", len(news_data))
        st.metric("Unique Tickers", len(set(item['ticker'] for item in news_data)))
        
        # API Configuration
        st.subheader("🔑 API Configuration")
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            st.success("✅ OpenAI API key configured")
        else:
            st.warning("⚠️ OpenAI API key not found")

    # Main content
    st.title("💹 MarketPulse AI")
    st.caption("Real-time Financial News Analysis & Q&A")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📰 Latest News", "📊 Analytics"])

    with tab1:
        st.subheader("Ask about market news")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant", 
                    "content": "Hi! I'm MarketPulse AI. Ask me about recent market news, earnings, stock movements, or specific companies like Apple, Tesla, Google, Microsoft, NVIDIA, or Meta."
                }
            ]
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about stocks, earnings, or market news..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing market news..."):
                    # Retrieve relevant documents
                    relevant_docs = retrieve_relevant_docs(prompt, news_data, top_k=3)
                    
                    # Generate answer
                    response = answer_question(prompt, relevant_docs)
                    
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

    with tab2:
        st.subheader("📰 Latest Market News")
        
        # Show news data
        for doc in news_data:
            timestamp = datetime.fromisoformat(doc['timestamp'])
            time_ago = datetime.now() - timestamp
            
            if time_ago.total_seconds() < 3600:
                time_str = f"{int(time_ago.total_seconds() / 60)}m ago"
            else:
                time_str = f"{int(time_ago.total_seconds() / 3600)}h ago"
            
            with st.container():
                col1, col2, col3 = st.columns([1, 6, 2])
                
                with col1:
                    st.markdown(f"**{doc['ticker']}**")
                
                with col2:
                    st.markdown(doc['headline'])
                
                with col3:
                    st.caption(f"{time_str} • {doc['source']}")
                
                st.divider()

    with tab3:
        st.subheader("📊 Market Analytics")
        
        # Ticker distribution
        tickers = [item['ticker'] for item in news_data]
        ticker_counts = pd.Series(tickers).value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.bar_chart(ticker_counts)
            st.caption("News count by ticker")
        
        with col2:
            # Time distribution
            timestamps = [datetime.fromisoformat(item['timestamp']) for item in news_data]
            hours_ago = [(datetime.now() - ts).total_seconds() / 3600 for ts in timestamps]
            
            time_df = pd.DataFrame({
                'Hours Ago': hours_ago,
                'Count': [1] * len(hours_ago)
            })
            
            st.scatter_chart(time_df.set_index('Hours Ago'))
            st.caption("News timing distribution")

    # Footer
    st.markdown("---")
    st.markdown("**MarketPulse AI** • Built with Streamlit • Demo Version")

if __name__ == "__main__":
    main()