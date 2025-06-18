import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh
import time
from typing import List, Dict, Tuple, Union

# ========== Load environment variables ==========
load_dotenv()

# ========== Imports ==========
from retrieval_engine import answer_query_sync, retrieve_top_k, embed_text
from ingest_pipeline import (
    get_news_by_ticker, 
    get_news_by_category,
    get_recent_news,
    get_pipeline_stats,
    refresh_cache
)

# ========== Page Configuration ==========
st.set_page_config(
    page_title="MarketPulse AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Styling ==========
st.markdown(
    """
    <style>
        .centered-title {
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            font-family: 'Segoe UI', sans-serif;
            margin-bottom: 0.2em;
        }
        .streamlit-expanderHeader {
            font-size: 1.2em !important;
        }
        .headline-new {
            color: green;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ========== Utilities ==========
@st.cache_data
def mock_embed_text(text: str) -> List[float]:
    np.random.seed(hash(text) % 2**32)
    return np.random.rand(1536).tolist()

def get_mock_news_data() -> List[Dict[str, Union[str, List[float]]]]:
    base_time = datetime.now()
    headlines = [
        ("AAPL", "📊 Apple reports record Q4 earnings, beating analyst expectations by 15%", "Reuters"),
        ("TSLA", "🚗 Tesla delivers 500,000 vehicles in Q4, stock surges 8% in after-hours trading", "Bloomberg"),
        ("GOOGL", "🧠 Google announces breakthrough in quantum computing, Alphabet shares jump 12%", "TechCrunch"),
        ("MSFT", "☁️ Microsoft Azure revenue grows 35% YoY, cloud dominance continues", "CNBC"),
        ("NVDA", "🤖 NVIDIA partners with major automakers for next-gen AI chips", "MarketWatch"),
        ("META", "🕶️ Meta's VR division shows promising growth, metaverse investments paying off", "The Verge")
    ]
    return [{
        "id": f"{ticker}_001",
        "ticker": ticker,
        "headline": headline,
        "timestamp": (base_time - timedelta(minutes=i*20)).isoformat(),
        "source": source,
        "embedding": mock_embed_text(headline)
    } for i, (ticker, headline, source) in enumerate(headlines)]

def get_pathway_table() -> pd.DataFrame:
    return pd.DataFrame([{
        "Ticker": item['ticker'],
        "Headline": item['headline'],
        "Timestamp": datetime.fromisoformat(item['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
    } for item in st.session_state.news_data])

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a_vec, b_vec = np.array(a), np.array(b)
    return np.dot(a_vec, b_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(b_vec))

def retrieve_relevant_docs(question: str, data: List[Dict], top_k: int = 3) -> List[Dict]:
    question_embedding = mock_embed_text(question)
    scored_docs = []
    for doc in data:
        similarity = cosine_similarity(question_embedding, doc['embedding'])
        scored_docs.append({**doc, 'similarity': similarity})
    scored_docs.sort(key=lambda x: x['similarity'], reverse=True)
    return scored_docs[:top_k]

# ========== Session State Init ==========
if "news_data" not in st.session_state:
    st.session_state.news_data = get_mock_news_data()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm MarketPulse AI. Ask me about recent market news, earnings, stock movements, or specific companies like Apple, Tesla, or Google."
        }
    ]

# ========== Tabs ==========
def render_sidebar():
    with st.sidebar:
        st.title("Configuration")
        theme = st.radio("Theme", ["light", "dark"])
        openai_key = os.getenv("OPENAI_API_KEY")
        mock_mode = os.getenv("MOCK_EMBEDDING", "false").lower() == "true"
        if openai_key and not mock_mode:
            st.success("OpenAI API Active")
        else:
            st.warning("Mock mode or API not configured")

        st_autorefresh(interval=5000, limit=None, key="sidebar_livefeed_refresh")
        st.markdown("### Live-feed: Latest News Table")
        st.dataframe(get_pathway_table(), use_container_width=True)

def render_chat_tab():
    st.subheader("Ask about market news")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about stocks, earnings, or market news..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            for i in range(3):
                placeholder.markdown(f"Analyzing market news{'.' * (i+1)}")
                time.sleep(0.5)

            with st.spinner("Analyzing market news..."):
                relevant_docs = retrieve_relevant_docs(prompt, st.session_state.news_data)
                result = answer_query_sync(prompt, relevant_docs)
                latency = None
                if isinstance(result, tuple):
                    answer, latency = result
                else:
                    answer = result

            placeholder.markdown(answer)
            if latency:
                st.markdown(f'<span style="font-size:12px;color:gray;">⏱️ Latency: {latency:.2f} seconds</span>', unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": answer})

def render_news_tab():
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
                is_new = time_ago.total_seconds() < 300
                st.markdown(f"{doc['headline']} {'🟢 NEW' if is_new else ''}")
            with col3:
                st.caption(f"{time_str} • {doc['source']}")
            st.divider()

def render_analytics_tab():
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

# ========== Main Entry ==========
def main():
    render_sidebar()
    st.markdown("<div class='centered-title'>MarketPulse AI</div>", unsafe_allow_html=True)
    st.caption("Real-time Financial News Analysis & Q&A")
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📰 Latest News", "📊 Analytics"])
    with tab1:
        render_chat_tab()
    with tab2:
        render_news_tab()
    with tab3:
        render_analytics_tab()
    st.markdown("---")
    st.markdown("MarketPulse AI • Built with Streamlit • Demo Version")

if __name__ == "__main__":
    main()
