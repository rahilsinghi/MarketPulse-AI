"""
<<<<<<< HEAD
<<<<<<< HEAD
MarketPulse AI - Updated to use real OpenAI retrieval engine
=======
MarketPulse AI - Main Streamlit Application with Persistent Dark/Light Mode
>>>>>>> 8f8c747 (UI update)
=======
MarketPulse AI - Main Streamlit Application with Persistent Dark/Light Mode
>>>>>>> 122955e98d6b94dee209e6a2c97700fac9d392af
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh
import time

# Load environment
load_dotenv()

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
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

<<<<<<< HEAD
<<<<<<< HEAD
def main():
    # Sidebar
    with st.sidebar:
        st.title("🔧 Configuration")
        
        # System status
        st.subheader("⚡ System Status")
        
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
=======
=======
>>>>>>> 122955e98d6b94dee209e6a2c97700fac9d392af
# Dark/Light Mode Management with persistence
def get_initial_theme():
    query_params = st.experimental_get_query_params()
    return query_params.get("theme", ["light"])[0]

if "theme" not in st.session_state:
    st.session_state.theme = get_initial_theme()

def set_theme(new_theme):
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.experimental_set_query_params(theme=new_theme)
        st.experimental_rerun()

# Updated color palettes for a more professional look
COLOR_PALETTES = {
    "light": {
        "primary": "#0B3D91",           # Dark blue
        "background": "#FFFFFF",        # White
        "text": "#222222",              # Dark gray
        "secondary_text": "#555555",    # Medium gray
        "sidebar_bg": "#F7F9FC",        # Very light gray-blue
        "border_color": "#DDDDDD",      # Light gray
        "card_bg": "#FAFAFA"            # Slightly off-white
    },
    "dark": {
        "primary": "#5699D2",           # Soft blue
        "background": "#121212",        # Very dark gray
        "text": "#E0E0E0",              # Light gray
        "secondary_text": "#AAAAAA",    # Medium-light gray
        "sidebar_bg": "#1F1F1F",        # Dark gray
        "border_color": "#333333",      # Dark gray
        "card_bg": "#1E1E1E"            # Dark gray
    }
}

colors = COLOR_PALETTES[st.session_state.theme]

# Inject CSS for colors and typography
def inject_css():
    st.markdown(
        f"""
        <style>
            /* Global background and text colors */
            .css-1d391kg {{
                background-color: {colors['background']} !important;
                color: {colors['text']} !important;
            }}
            /* Sidebar background */
            .css-1d391kg .css-18e3th9 {{
                background-color: {colors['sidebar_bg']} !important;
            }}
            /* Card/container backgrounds */
            .css-1d391kg .css-12oz5g7, .stButton>button {{
                background-color: {colors['card_bg']} !important;
                border: 1px solid {colors['border_color']} !important;
                color: {colors['text']} !important;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            /* Headings typography */
            h1, h2, h3, h4, h5, h6 {{
                font-family: 'Montserrat', sans-serif;
                font-weight: 700;
                color: {colors['primary']};
            }}
            /* Paragraph text */
            p, span, div {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                color: {colors['text']};
            }}
            /* Sidebar titles */
            .css-1d391kg .css-1v0mbdj h1, .css-1d391kg .css-1v0mbdj h2 {{
                color: {colors['primary']};
            }}
            /* Links */
            a {{
                color: {colors['primary']};
            }}
            /* Chat messages style */
            .stChatMessage div[role="button"] {{
                color: {colors['text']} !important;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            /* Streamlit divider color */
            .stDivider {{
                border-color: {colors['border_color']} !important;
            }}
            /* Streamlit metric label */
            .stMetric label {{
                color: {colors['secondary_text']} !important;
            }}
            /* Scrollbar for sidebar */
            ::-webkit-scrollbar {{
                width: 8px;
                height: 8px;
            }}
            ::-webkit-scrollbar-thumb {{
                background: {colors['border_color']};
                border-radius: 10px;
            }}
            ::-webkit-scrollbar-thumb:hover {{
                background: {colors['primary']};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()

# Mock embedding function (deterministic)
@st.cache_data
def mock_embed_text(text: str):
    np.random.seed(hash(text) % 2**32)
    return np.random.rand(1536).tolist()

# Mock news data
def get_mock_news_data():
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

# Initialize news_data in session state once
if "news_data" not in st.session_state:
    st.session_state.news_data = get_mock_news_data()

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_relevant_docs(question: str, data: list, top_k: int = 3):
    question_embedding = mock_embed_text(question)
    scored_docs = []
    for doc in data:
        similarity = cosine_similarity(question_embedding, doc['embedding'])
        scored_docs.append({**doc, 'similarity': similarity})
    scored_docs.sort(key=lambda x: x['similarity'], reverse=True)
    return scored_docs[:top_k]

def get_pathway_table():
    df = pd.DataFrame([{
        "Ticker": item['ticker'],
        "Headline": item['headline'],
        "Timestamp": datetime.fromisoformat(item['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
    } for item in st.session_state.news_data])
    return df

# Assume query_agent is imported or defined somewhere in your codebase
# from query_agent_module import query_agent

def main():
    with st.sidebar:
        st.title("Configuration")

        theme_choice = st.radio(
            "Choose Theme:",
            options=["light", "dark"],
            index=0 if st.session_state.theme == "light" else 1,
            help="Toggle between Light and Dark mode"
        )
        if theme_choice != st.session_state.theme:
            set_theme(theme_choice)

        st.markdown("---")

        st.subheader("Data Source")
        st.info("Currently using mock data for demonstration. In production, this would connect to real-time news feeds.")

        st.subheader("System Status")
        st.success("Core system operational")
        st.success("Mock data loaded")

        news_data = st.session_state.news_data
        st.metric("Total News Items", len(news_data))
        st.metric("Unique Tickers", len(set(item['ticker'] for item in news_data)))

        st.subheader("API Configuration")
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            st.success("OpenAI API key configured")
        else:
            st.warning("OpenAI API key not found")

        # === Live feed auto-refresh every 5 seconds ===
        count = st_autorefresh(interval=5 * 1000, limit=None, key="livefeed_autorefresh")
        st.markdown("Live Market Feed (updates every 5 seconds)")

        # Generate live table data
        df = get_pathway_table()
        st.dataframe(df, use_container_width=True)

    # Main content
    st.title("MarketPulse AI")
    st.caption("Real-time Financial News Analysis & Q&A")

    tab1, tab2, tab3 = st.tabs(["Chat", "Latest News", "Analytics"])

    with tab1:
        st.subheader("Ask about market news")

        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hi! I'm MarketPulse AI. Ask me about recent market news, earnings, stock movements, or specific companies like Apple, Tesla, Google, Microsoft, NVIDIA, or Meta."
                }
            ]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about stocks, earnings, or market news..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()  # Placeholder for typing indicator / final answer

                # Typing indicator animation (dots)
                for i in range(3):
                    message_placeholder.markdown(f"Analyzing market news{'.' * (i+1)}")
                    time.sleep(0.5)

                # Now do the actual processing
                with st.spinner("Analyzing market news..."):
                    relevant_docs = retrieve_relevant_docs(prompt, st.session_state.news_data, top_k=3)
                    result = query_agent.answer_query_sync(prompt, relevant_docs)

                if isinstance(result, tuple) and len(result) == 2:
                    answer, latency = result
                else:
                    answer = result
                    latency = None

                # Show the final answer (replacing typing indicator)
                message_placeholder.markdown(answer)
                if latency is not None:
                    st.markdown(f'<span style="font-size:12px;color:gray;">Latency: {latency:.2f} seconds</span>', unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": answer})

    with tab2:
        st.subheader("Latest Market News")
        for doc in st.session_state.news_data:
            timestamp = datetime.fromisoformat(doc['timestamp'])
            time_ago = datetime.now() - timestamp
            time_str = (
                f"{int(time_ago.total_seconds() / 60)}m ago"
                if time_ago.total_seconds() < 3600
                else f"{int(time_ago.total_seconds() / 3600)}h ago"
            )
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
        st.subheader("Market Analytics")

        tickers = [item['ticker'] for item in st.session_state.news_data]
        ticker_counts = pd.Series(tickers).value_counts()

        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(ticker_counts)
            st.caption("News count by ticker")
        with col2:
            timestamps = [datetime.fromisoformat(item['timestamp']) for item in st.session_state.news_data]
            hours_ago = [(datetime.now() - ts).total_seconds() / 3600 for ts in timestamps]
            time_df = pd.DataFrame({'Hours Ago': hours_ago, 'Count': [1] * len(hours_ago)})
            st.scatter_chart(time_df.set_index('Hours Ago'))
            st.caption("News timing distribution")

    st.markdown("---")
    st.markdown("MarketPulse AI • Built with Streamlit • Demo Version")
<<<<<<< HEAD
>>>>>>> 8f8c747 (UI update)
=======
>>>>>>> 122955e98d6b94dee209e6a2c97700fac9d392af

if __name__ == "__main__":
    main()
