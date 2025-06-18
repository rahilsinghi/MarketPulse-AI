import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import time
import pandas as pd

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set page config
st.set_page_config(page_title="Real-Time Finance Q&A Bot", layout="wide")
st.title("💬 Real-Time Finance Q&A Bot\n(OpenAI)")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "questions_log" not in st.session_state:
    st.session_state.questions_log = []

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Ask OpenAI
def ask_openai(question):
    prompt = f"""
You are a financial assistant. Based on current stock market and company news, provide helpful insights.

Question: {question}
"""
    start = time.time()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    latency = time.time() - start
    reply = response.choices[0].message.content.strip()
    return reply, latency

# Sidebar live feed
st.sidebar.subheader("🗞️ Live Feed")
recent_questions = st.session_state.questions_log[-6:][::-1]  # last 6, most recent first
if recent_questions:
    df = pd.DataFrame(recent_questions, columns=["Headline", "Timestamp"])
    st.sidebar.table(df)
else:
    st.sidebar.info("No questions asked yet.")

# Toggle theme
if st.sidebar.button("Toggle Theme"):
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

# Chat input
user_input = st.chat_input("Ask me something about a company or stock...")
if user_input:
    with st.spinner("Getting answer..."):
        reply, latency = ask_openai(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.questions_log.append([user_input, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

# Display message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])