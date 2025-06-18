"""
MarketPulse AI - Updated to use real OpenAI retrieval engine
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment
load_dotenv()
from marketpulse_ml import (
    embed_text,
    retrieve_by_query,
    answer_query_sync,
    get_embedding_cache_stats,
    get_retrieval_stats,
    configure_package,
    DEFAULT_SIMILARITY_THRESHOLD
)

# Legacy imports for backward compatibility
from retrieval_engine import build_prompt
from ingest_pipeline import (
    get_latest_rows, 
    get_news_by_ticker, 
    get_news_by_category,
    get_recent_news,
    get_pipeline_stats,
    refresh_cache
)


# Import the real retrieval engine
from retrieval_engine import answer_query_sync, retrieve_top_k, embed_text
from ingest_pipeline import (
    get_latest_rows, 
    get_news_by_ticker, 
    get_news_by_category,
    get_recent_news,
    get_pipeline_stats,
    refresh_cache
)

# Configuration
st.set_page_config(
    page_title="MarketPulse AI",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Sidebar
    with st.sidebar:
        st.title("🔧 Configuration")
        
        # System status
        st.subheader("📦 Package Info")
        try:
            cache_stats = get_embedding_cache_stats()
            retrieval_stats = get_retrieval_stats()
            
            st.metric("Cache Hit Rate", f"{cache_stats['hit_rate']:.1%}")
            st.metric("Cache Size", f"{cache_stats['currsize']}/{cache_stats['maxsize']}")
            st.metric("Similarity Threshold", f"{DEFAULT_SIMILARITY_THRESHOLD:.2f}")
            
            if st.button("🧹 Clear Cache"):
                from marketpulse_ml import clear_embedding_cache
                clear_embedding_cache()
                st.success("Cache cleared!")
                st.rerun()
                
        except Exception as e:
            st.error(f"Package stats error: {e}")
        
        # Check environment
        openai_key = os.getenv("OPENAI_API_KEY")
        mock_mode = os.getenv("MOCK_EMBEDDING", "false").lower() == "true"
        
        if openai_key and not mock_mode:
            st.success("✅ OpenAI API Active")
            st.info("🔄 Using real AI embeddings & responses")
        elif openai_key and mock_mode:
            st.warning("⚠️ Mock mode enabled")
            st.info("🧪 Using mock embeddings (set MOCK_EMBEDDING=false for real AI)")
        else:
            st.error("❌ OpenAI API not configured")
            st.info("🔑 Set OPENAI_API_KEY for real AI")
        
        # Pipeline stats
        try:
            stats = get_pipeline_stats()
            st.metric("Total Articles", stats['total_articles'])
            st.metric("With Embeddings", stats['articles_with_embeddings'])
            st.metric("Unique Tickers", stats['unique_tickers'])
            st.metric("Avg Age (hours)", f"{stats['avg_hours_ago']:.1f}")
            
            if st.button("🔄 Refresh Data"):
                with st.spinner("Refreshing..."):
                    refresh_cache()
                st.success("Data refreshed!")
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Pipeline error: {e}")
        
        # Settings
        st.subheader("⚙️ Query Settings")
        top_k = st.slider("Results per query", 1, 10, 4)
        
        # Toggle modes
        if st.checkbox("Force mock mode", value=mock_mode):
            os.environ["MOCK_EMBEDDING"] = "true"
        else:
            os.environ["MOCK_EMBEDDING"] = "false"

    # Main content
    st.title("💹 MarketPulse AI")
    st.caption("Real-time Financial News Analysis with AI-Powered Retrieval")
    
    # Show current mode
    if openai_key and not mock_mode:
        st.success("🤖 **AI Mode Active** - Using OpenAI for embeddings and responses")
    else:
        st.info("🧪 **Demo Mode** - Using mock data for testing")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 AI Chat", "📰 Live News", "📊 Analytics", "🔧 Debug"])
    
    with tab1:
        st.subheader("Ask AI about market news")
        
        # Initialize chat history
        if "ai_messages" not in st.session_state:
            st.session_state.ai_messages = [
                {
                    "role": "assistant",
                    "content": "Hi! I'm MarketPulse AI. I use advanced semantic search and AI to analyze financial news. Ask me about specific companies, earnings, market trends, or any financial topic!"
                }
            ]
        
        # Display chat messages
        for message in st.session_state.ai_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about stocks, earnings, market news, or financial trends..."):
            # Add user message
            st.session_state.ai_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("🤖 AI analyzing news and generating response..."):
                    # Use the real retrieval engine
                    response = answer_query_sync(prompt, top_k=top_k)
                    st.markdown(response)
            
            # Add assistant response
            st.session_state.ai_messages.append({"role": "assistant", "content": response})
    
    with tab2:
        st.subheader("📰 Live Financial News")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_type = st.selectbox("Filter by", ["All Recent", "Ticker", "Category"])
        
        with col2:
            if filter_type == "Ticker":
                available_tickers = ["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA", "META", "AMZN", "NFLX", "SPY", "BTC"]
                selected_ticker = st.selectbox("Select Ticker", available_tickers)
            elif filter_type == "Category":
                available_categories = ["earnings", "partnership", "innovation", "delivery", "market", "crypto", "growth", "contract"]
                selected_category = st.selectbox("Select Category", available_categories)
        
        with col3:
            hours_filter = st.selectbox("Time Range", [6, 12, 24, 48], index=2)
            st.caption(f"Last {hours_filter} hours")
        
        # Get filtered news
        try:
            if filter_type == "Ticker":
                news_data = get_news_by_ticker(selected_ticker)
            elif filter_type == "Category":
                news_data = get_news_by_category(selected_category)
            else:
                news_data = get_recent_news(hours=hours_filter)
            
            # Display news with enhanced formatting
            for article in news_data:
                with st.container():
                    col1, col2, col3, col4 = st.columns([1, 5, 1, 2])
                    
                    with col1:
                        ticker = article.get('ticker', 'N/A')
                        st.markdown(f"**{ticker}**")
                        
                        # Add sentiment indicator if available
                        sentiment = article.get('sentiment', 'neutral')
                        if sentiment == 'positive':
                            st.markdown("🟢")
                        elif sentiment == 'negative':
                            st.markdown("🔴")
                        else:
                            st.markdown("⚪")
                    
                    with col2:
                        headline = article.get('headline', 'No headline')
                        st.markdown(headline)
                        
                        # Show similarity score if searching
                        if 'similarity' in article:
                            st.caption(f"Relevance: {article['similarity']:.3f}")
                    
                    with col3:
                        category = article.get('category', 'general')
                        # Create colored badges for categories
                        if category == 'earnings':
                            st.markdown("🟡 **Earnings**")
                        elif category == 'partnership':
                            st.markdown("🔵 **Partnership**")
                        elif category == 'innovation':
                            st.markdown("🟣 **Innovation**")
                        else:
                            st.markdown(f"⚫ **{category.title()}**")
                    
                    with col4:
                        timestamp = article.get('timestamp', '')
                        if timestamp:
                            try:
                                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                time_ago = datetime.now() - dt.replace(tzinfo=None)
                                if time_ago.total_seconds() < 3600:
                                    time_str = f"{int(time_ago.total_seconds() / 60)}m ago"
                                else:
                                    time_str = f"{int(time_ago.total_seconds() / 3600)}h ago"
                                st.caption(f"🕒 {time_str}")
                                st.caption(f"📰 {article.get('source', 'Unknown')}")
                            except:
                                st.caption(f"📰 {article.get('source', 'Unknown')}")
                    
                    st.divider()
            
            if not news_data:
                st.info("No news found for the selected filters.")
                    
        except Exception as e:
            st.error(f"Error loading news: {e}")
    
    with tab3:
        st.subheader("📊 Market Analytics")
        
        try:
            stats = get_pipeline_stats()
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📰 Total Articles", stats['total_articles'])
            with col2:
                st.metric("🧠 With AI Embeddings", stats['articles_with_embeddings'])
            with col3:
                st.metric("🏢 Unique Companies", stats['unique_tickers'])
            with col4:
                cache_age = stats.get('cache_age_seconds', 0)
                st.metric("⚡ Cache Age", f"{cache_age:.1f}s")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Company Coverage")
                ticker_df = pd.DataFrame(
                    list(stats['ticker_distribution'].items()),
                    columns=['Company', 'Articles']
                )
                st.bar_chart(ticker_df.set_index('Company'))
            
            with col2:
                st.subheader("📋 News Categories")
                category_df = pd.DataFrame(
                    list(stats['category_distribution'].items()),
                    columns=['Category', 'Count']
                )
                st.bar_chart(category_df.set_index('Category'))
            
            # Sentiment analysis
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("😊 Sentiment Analysis")
                sentiment_df = pd.DataFrame(
                    list(stats['sentiment_distribution'].items()),
                    columns=['Sentiment', 'Count']
                )
                st.bar_chart(sentiment_df.set_index('Sentiment'))
            
            with col2:
                st.subheader("📊 Key Metrics")
                avg_age = stats.get('avg_hours_ago', 0)
                st.metric("⏱️ Average News Age", f"{avg_age:.1f} hours")
                
                embedding_rate = (stats['articles_with_embeddings'] / stats['total_articles']) * 100
                st.metric("🧠 AI Processing Rate", f"{embedding_rate:.1f}%")
                
                recent_count = len(get_recent_news(hours=6))
                st.metric("🔥 Recent News (6h)", recent_count)
            
        except Exception as e:
            st.error(f"Error loading analytics: {e}")
    
    with tab4:
        st.subheader("🔧 Debug & Testing")
        
        # Test embedding
        with st.expander("🧠 Test AI Embedding"):
            test_text = st.text_input("Text to embed:", "Apple quarterly earnings report")
            if st.button("Generate Embedding"):
                with st.spinner("Generating AI embedding..."):
                    try:
                        start_time = datetime.now()
                        embedding = embed_text(test_text)
                        end_time = datetime.now()
                        duration = (end_time - start_time).total_seconds()
                        
                        st.success(f"✅ Generated {len(embedding)}-dimensional embedding in {duration:.2f}s")
                        st.code(f"First 10 values: {embedding[:10]}")
                        
                        # Show if using real AI or mock
                        if os.getenv("MOCK_EMBEDDING", "false").lower() == "true":
                            st.info("🧪 Using mock embedding (deterministic)")
                        else:
                            st.success("🤖 Using real OpenAI embedding")
                            
                    except Exception as e:
                        st.error(f"❌ Failed: {e}")
        
        # Test retrieval
        with st.expander("🔍 Test Document Retrieval"):
            query = st.text_input("Search query:", "Tesla delivery numbers")
            if st.button("Test AI Search"):
                with st.spinner("AI searching documents..."):
                    try:
                        start_time = datetime.now()
                        query_vec = embed_text(query)
                        results = retrieve_top_k(query_vec, k=5)
                        end_time = datetime.now()
                        duration = (end_time - start_time).total_seconds()
                        
                        st.success(f"✅ Found {len(results)} results in {duration:.2f}s")
                        
                        for i, result in enumerate(results):
                            with st.container():
                                col1, col2, col3 = st.columns([1, 5, 1])
                                
                                with col1:
                                    st.write(f"**#{i+1}**")
                                    similarity = result.get('similarity', 0)
                                    st.metric("Similarity", f"{similarity:.3f}")
                                
                                with col2:
                                    ticker = result.get('ticker', 'N/A')
                                    headline = result.get('headline', 'No headline')
                                    st.write(f"**[{ticker}]** {headline}")
                                    
                                    source = result.get('source', 'Unknown')
                                    category = result.get('category', 'general')
                                    st.caption(f"📰 {source} • 📋 {category}")
                                
                                with col3:
                                    timestamp = result.get('timestamp', '')
                                    if timestamp:
                                        try:
                                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                            time_ago = datetime.now() - dt.replace(tzinfo=None)
                                            if time_ago.total_seconds() < 3600:
                                                time_str = f"{int(time_ago.total_seconds() / 60)}m"
                                            else:
                                                time_str = f"{int(time_ago.total_seconds() / 3600)}h"
                                            st.caption(f"🕒 {time_str} ago")
                                        except:
                                            st.caption("🕒 Recent")
                                
                                st.divider()
                                
                    except Exception as e:
                        st.error(f"❌ Search failed: {e}")
        
        # Environment info
        with st.expander("🔧 System Information"):
            st.code(f"""
Environment Variables:
OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Not set'}
MOCK_EMBEDDING: {os.getenv('MOCK_EMBEDDING', 'false')}

System Info:
Streamlit Version: {st.__version__}
Cache Status: Active
AI Mode: {'Mock' if os.getenv('MOCK_EMBEDDING', 'false').lower() == 'true' else 'Real OpenAI'}
            """)
        
        # Quick test button
        if st.button("🚀 Quick AI Test"):
            with st.spinner("Running quick AI test..."):
                try:
                    test_result = answer_query_sync("Test AI functionality", top_k=2)
                    st.success("✅ AI test completed!")
                    st.write("**Test Result:**")
                    st.write(test_result)
                except Exception as e:
                    st.error(f"❌ AI test failed: {e}")

if __name__ == "__main__":
    main()