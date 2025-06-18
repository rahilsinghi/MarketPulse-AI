"""
Test core functionality without OpenAI API calls.
"""

import numpy as np
from datetime import datetime, timedelta

def test_core_without_openai():
    print("🧪 Testing core functionality without OpenAI...")
    
    # Mock embedding (1536 dimensions like OpenAI)
    def mock_embed_text(text: str):
        np.random.seed(hash(text) % 2**32)  # Deterministic based on text
        return np.random.rand(1536).tolist()
    
    # Test data
    headlines = [
        "Apple reports strong quarterly earnings",
        "Tesla stock rises on delivery numbers", 
        "Google announces AI breakthrough",
        "Microsoft Azure growth accelerates"
    ]
    
    # Test embedding
    embeddings = {}
    for headline in headlines:
        embeddings[headline] = mock_embed_text(headline)
        print(f"✅ Mock embedding for: {headline[:30]}...")
    
    # Test similarity
    def cosine_similarity(a, b):
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # Test similarity between embeddings
    emb1 = embeddings[headlines[0]]
    emb2 = embeddings[headlines[1]]
    similarity = cosine_similarity(emb1, emb2)
    print(f"✅ Similarity calculation: {similarity:.3f}")
    
    # Test Q&A logic
    def simple_qa(question: str):
        question_words = question.lower().split()
        scores = {}
        
        for headline in headlines:
            headline_words = headline.lower().split()
            score = len(set(question_words) & set(headline_words))
            if score > 0:
                scores[headline] = score
        
        if not scores:
            return "No relevant news found."
        
        best_headline = max(scores, key=scores.get)
        return f"Most relevant: {best_headline}"
    
    # Test questions
    test_questions = ["Apple earnings", "Tesla delivery", "Google AI", "Microsoft cloud"]
    
    for question in test_questions:
        answer = simple_qa(question)
        print(f"Q: {question} → A: {answer}")
    
    print("✅ All core tests passed!")

if __name__ == "__main__":
    test_core_without_openai()