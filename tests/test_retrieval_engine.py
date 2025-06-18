"""
Test script for the new retrieval engine implementation.
"""

import os
import asyncio
from retrieval_engine import (
    embed_text,
    retrieve_top_k, 
    build_prompt,
    answer_query_sync,
    _answer_query_async
)

def test_retrieval_engine():
    """Test all retrieval engine functions."""
    print("🧪 Testing MarketPulse AI Retrieval Engine")
    print("=" * 50)
    
    # Set mock mode for testing
    os.environ["MOCK_EMBEDDING"] = "true"
    
    # Test 1: Embedding function
    print("\n1️⃣ Testing embed_text()...")
    try:
        embedding = embed_text("Apple stock news")
        print(f"✅ Embedding generated: {len(embedding)} dimensions")
        assert len(embedding) == 1536, "Embedding should be 1536 dimensions"
        print(f"   Sample values: {embedding[:5]}")
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return
    
    # Test 2: Retrieval function
    print("\n2️⃣ Testing retrieve_top_k()...")
    try:
        query_vec = embed_text("Tesla delivery numbers")
        results = retrieve_top_k(query_vec, k=3)
        print(f"✅ Retrieved {len(results)} documents")
        
        for i, result in enumerate(results):
            ticker = result.get('ticker', 'N/A')
            similarity = result.get('similarity', 0)
            headline = result.get('headline', 'N/A')[:60]
            print(f"   {i+1}. [{ticker}] {similarity:.3f} - {headline}...")
            
    except Exception as e:
        print(f"❌ Retrieval test failed: {e}")
        return
    
    # Test 3: Prompt building
    print("\n3️⃣ Testing build_prompt()...")
    try:
        question = "What's happening with Tesla?"
        messages = build_prompt(question, results)
        print(f"✅ Built prompt with {len(messages)} messages")
        print(f"   System message: {messages[0]['content'][:80]}...")
        print(f"   User message preview: {messages[1]['content'][:100]}...")
    except Exception as e:
        print(f"❌ Prompt building test failed: {e}")
        return
    
    # Test 4: Async answer function
    print("\n4️⃣ Testing _answer_query_async()...")
    try:
        async def test_async():
            answer = await _answer_query_async(messages)
            return answer
        
        answer = asyncio.run(test_async())
        print(f"✅ Generated answer: {answer[:100]}...")
    except Exception as e:
        print(f"❌ Async answer test failed: {e}")
        return
    
    # Test 5: Full synchronous query
    print("\n5️⃣ Testing answer_query_sync()...")
    try:
        test_questions = [
            "What's happening with Apple?",
            "Tell me about Tesla deliveries",
            "Any Google news?",
            "Microsoft cloud growth"
        ]
        
        for question in test_questions:
            answer = answer_query_sync(question, top_k=3)
            print(f"   Q: {question}")
            print(f"   A: {answer}")
            print()
            
    except Exception as e:
        print(f"❌ Sync query test failed: {e}")
        return
    
    print("🎉 All retrieval engine tests passed!")
    print("\n💡 Next steps:")
    print("   1. Set OPENAI_API_KEY for real embeddings")
    print("   2. Install FAISS for faster search: pip install faiss-cpu")
    print("   3. Integrate with Streamlit app")

if __name__ == "__main__":
    test_retrieval_engine()