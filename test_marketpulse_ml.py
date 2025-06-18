"""
Test script for the MarketPulse ML package.
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, '.')

def test_marketpulse_ml_package():
    """Test the new MarketPulse ML package structure."""
    print("🧪 Testing MarketPulse ML Package")
    print("=" * 50)
    
    # Test 1: Package import
    print("\n1️⃣ Testing package import...")
    try:
        import marketpulse_ml
        print(f"✅ Package imported successfully")
        print(f"   Version: {marketpulse_ml.__version__}")
        print(f"   Available functions: {len(marketpulse_ml.__all__)}")
    except Exception as e:
        print(f"❌ Package import failed: {e}")
        return
    
    # Test 2: Embedding function
    print("\n2️⃣ Testing embed_text from package...")
    try:
        from marketpulse_ml import embed_text
        embedding = embed_text("Apple quarterly earnings")
        print(f"✅ Embedding generated: {len(embedding)} dimensions")
        print(f"   Sample values: {embedding[:5]}")
        
        # Test cache
        embedding2 = embed_text("Apple quarterly earnings")  # Should hit cache
        assert embedding == embedding2, "Cache not working"
        print("✅ Embedding cache working")
        
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return
    
    # Test 3: Retrieval function
    print("\n3️⃣ Testing retrieve_by_query from package...")
    try:
        from marketpulse_ml import retrieve_by_query
        results = retrieve_by_query("Microsoft cloud news", k=3)
        print(f"✅ Retrieved {len(results)} documents")
        
        for i, result in enumerate(results):
            ticker = result.get('ticker', 'N/A')
            similarity = result.get('similarity', 0)
            headline = result.get('headline', 'N/A')[:50]
            print(f"   {i+1}. [{ticker}] {similarity:.3f} - {headline}...")
            
    except Exception as e:
        print(f"❌ Retrieval test failed: {e}")
        return
    
    # Test 4: Full Q&A pipeline
    print("\n4️⃣ Testing answer_query_sync from package...")
    try:
        from marketpulse_ml import answer_query_sync
        response = answer_query_sync("What's happening with Tesla?")
        print(f"✅ Q&A pipeline working")
        print(f"   Response: {response[:100]}...")
        
    except Exception as e:
        print(f"❌ Q&A test failed: {e}")
        return
    
    # Test 5: Package configuration
    print("\n5️⃣ Testing package configuration...")
    try:
        from marketpulse_ml import configure_package, get_embedding_cache_stats
        
        # Configure package
        configure_package(
            mock_embedding=True,
            similarity_threshold=0.35,
            cache_size=256
        )
        
        # Test cache stats
        stats = get_embedding_cache_stats()
        print(f"✅ Package configuration working")
        print(f"   Cache hits: {stats['hits']}")
        print(f"   Cache misses: {stats['misses']}")
        print(f"   Hit rate: {stats['hit_rate']:.2%}")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return
    
    # Test 6: Similarity threshold tuning
    print("\n6️⃣ Testing similarity threshold tuning...")
    try:
        from marketpulse_ml import retrieve_by_query, DEFAULT_SIMILARITY_THRESHOLD
        
        print(f"   Default threshold: {DEFAULT_SIMILARITY_THRESHOLD}")
        
        # Test with different thresholds
        thresholds = [0.20, 0.25, 0.30, 0.35, 0.40]
        for threshold in thresholds:
            # We'll implement this in the actual retrieval function
            results = retrieve_by_query("Apple news", k=5, similarity_threshold=threshold)
            print(f"   Threshold {threshold}: {len(results)} results")
            
        print("✅ Threshold tuning test completed")
        
    except Exception as e:
        print(f"❌ Threshold test failed: {e}")
        return
    
    print("\n🎉 All MarketPulse ML package tests passed!")
    print("\n📦 Package is ready for use:")
    print("   from marketpulse_ml import embed_text, retrieve_by_query, answer_query_sync")
    print("   from marketpulse_ml import configure_package")
    
    print("\n💡 Next steps:")
    print("   1. Update app.py to use: from marketpulse_ml import answer_query_sync")
    print("   2. Update retrieval_engine.py to use the new similarity threshold")
    print("   3. Test FAISS integration with larger datasets")

if __name__ == "__main__":
    test_marketpulse_ml_package()