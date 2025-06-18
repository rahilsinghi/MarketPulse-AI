# app.py (Gemini version)

import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load Gemini key
load_dotenv()

print("Loaded GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="Live Finance Chatbot", layout="centered")
st.title("💬 Real-Time Finance Q&A Bot (Gemini)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Generate response from Gemini
def ask_gemini(question, mock_data):
    model = genai.GenerativeModel("models/gemini-pro")
    prompt = f"""You are a financial assistant. Use this real-time data to answer:

    {mock_data}

    Question: {question}
    Answer:"""
    response = model.generate_content(prompt)
    return response.text

# Handle user input
user_input = st.chat_input("Ask me something about a company or stock...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        mock_data = """
        📉 Tesla stock is down 2.4% today.
        📰 News: Tesla factory operations delayed due to new labor regulations.
        💬 Analyst sentiment: Mixed. Caution over international expansion.
        """
        with st.spinner("Thinking..."):
            reply = ask_gemini(user_input, mock_data)
        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
