"""
Streamlit front-end.  Launch with:
    streamlit run app.py
"""

import streamlit as st

import ingest_pipeline
from query_agent import answer_query_sync

# ─── kick off ingestion once ───────────────────────────────────────────────────
if "ingest_started" not in st.session_state:
    ingest_pipeline.start_ingest()
    st.session_state.ingest_started = True

# ─── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MarketPulse AI", layout="centered")
st.title("💹 MarketPulse AI")
st.caption("Real-time market Q&A powered by Pathway streaming RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

# replay chat history
for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

# new input
if prompt := st.chat_input("Ask about a stock, company, or market move…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.spinner("Crunching fresh data…"):
        response = answer_query_sync(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").markdown(response)
