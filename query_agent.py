from openai import OpenAI

# Directly set your key here
client = OpenAI(api_key="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")

def answer_query_sync(question):
    prompt = f"""
You are a financial assistant. Based on current stock market and company news, provide helpful insights.

Whenever you mention a company, include its stock ticker in square brackets. For example: Tesla [TSLA], Apple [AAPL], etc.

Question: {question}
"""


    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    answer = response.choices[0].message.content.strip()
    return answer, 0  # skip latency if not needed
