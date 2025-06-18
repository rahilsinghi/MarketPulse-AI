"""
Comprehensive tests for the query agent module.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from query_agent import (
    cosine_similarity, 
    retrieve_relevant_docs, 
    answer_query_async,
    answer_query_sync,
    build_context_prompt
)

class TestCosineSimilarity:
    """Test cosine similarity calculations."""
    
    def test_identical_vectors(self):
        """Test similarity of identical vectors."""
        vec = [1.0, 2.0, 3.0] + [0.0] * 1533  # Make it 1536-D
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)
    
    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors."""
        vec1 = [1.0, 0.0] + [0.0] * 1534
        vec2 = [0.0, 1.0] + [0.0] * 1534
        assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)
    
    def test_zero_vectors(self):
        """Test handling of zero vectors."""
        zero_vec = [0.0] * 1536
        normal_vec = [1.0] + [0.0] * 1535
        assert cosine_similarity(zero_vec, normal_vec) == 0.0
    
    def test_invalid_vectors(self):
        """Test handling of invalid vectors."""
        assert cosine_similarity([], [1.0] * 1536) == 0.0
        assert cosine_similarity([1.0] * 100, [1.0] * 1536) == 0.0

class TestRetrievalSystem:
    """Test document retrieval functionality."""
    
    @patch('query_agent.get_pathway_table')
    @patch('query_agent.embed_text')
    def test_retrieve_relevant_docs_empty_table(self, mock_embed, mock_table):
        """Test retrieval with empty document table."""
        mock_table.return_value = []
        mock_embed.return_value = [1.0] * 1536
        
        result = retrieve_relevant_docs("test question")
        assert len(result.documents) == 0
        assert result.total_docs == 0
    
    @patch('query_agent.get_pathway_table')
    @patch('query_agent.embed_text')
    def test_retrieve_relevant_docs_with_data(self, mock_embed, mock_table):
        """Test retrieval with valid documents."""
        # Mock data
        mock_docs = [
            {
                "id": "1",
                "ticker": "AAPL",
                "headline": "Apple stock rises",
                "timestamp": datetime.now().isoformat(),
                "embedding": [0.8] + [0.0] * 1535
            },
            {
                "id": "2", 
                "ticker": "GOOGL",
                "headline": "Google announces new AI",
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "embedding": [0.6] + [0.0] * 1535
            }
        ]
        
        mock_table.return_value = mock_docs
        mock_embed.return_value = [1.0] + [0.0] * 1535  # High similarity to first doc
        
        result = retrieve_relevant_docs("Apple stock news", top_k=1)
        
        assert len(result.documents) == 1
        assert result.documents[0]["ticker"] == "AAPL"
        assert result.total_docs == 2

class TestQueryAnswering:
    """Test the main query answering functionality."""
    
    @pytest.mark.asyncio
    @patch('query_agent.retrieve_relevant_docs')
    @patch('query_agent.call_openai_with_retry')
    async def test_answer_query_async_success(self, mock_openai, mock_retrieve):
        """Test successful query answering."""
        # Mock retrieval
        mock_doc = {
            "ticker": "AAPL",
            "headline": "Apple stock surges 5%",
            "timestamp": datetime.now().isoformat(),
            "similarity": 0.9,
            "age_hours": 1.0
        }
        
        from query_agent import RetrievalResult
        mock_retrieve.return_value = RetrievalResult([mock_doc], [1.0]*1536, 1, 0.1)
        
        # Mock OpenAI response
        mock_openai.return_value = "Apple stock has surged 5% due to strong earnings."
        
        result = await answer_query_async("Why is Apple stock up?")
        
        assert "Apple" in result
        assert "5%" in result
        mock_openai.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_answer_query_async_empty_question(self):
        """Test handling of empty questions."""
        result = await answer_query_async("")
        assert "⚠️" in result
        assert "valid question" in result
    
    def test_answer_query_sync_wrapper(self):
        """Test synchronous wrapper functionality."""
        with patch('query_agent.answer_query_async') as mock_async:
            mock_async.return_value = "Test response"
            result = answer_query_sync("test question")
            assert result == "Test response"

class TestPromptBuilding:
    """Test prompt construction functionality."""
    
    def test_build_context_prompt_single_ticker(self):
        """Test prompt building with single ticker."""
        docs = [
            {
                "ticker": "AAPL",
                "headline": "Apple reports strong earnings",
                "timestamp": datetime.now().isoformat(),
                "similarity": 0.9,
                "age_hours": 1.0
            }
        ]
        
        system_prompt, user_prompt = build_context_prompt(docs, "Apple earnings")
        
        assert "MarketPulse AI" in system_prompt
        assert "AAPL" in user_prompt
        assert "Apple reports strong earnings" in user_prompt
        assert "Apple earnings" in user_prompt
    
    def test_build_context_prompt_multiple_tickers(self):
        """Test prompt building with multiple tickers."""
        docs = [
            {
                "ticker": "AAPL",
                "headline": "Apple stock up",
                "timestamp": datetime.now().isoformat(),
                "similarity": 0.9,
                "age_hours": 1.0
            },
            {
                "ticker": "GOOGL", 
                "headline": "Google announces AI breakthrough",
                "timestamp": datetime.now().isoformat(),
                "similarity": 0.8,
                "age_hours": 2.0
            }
        ]
        
        system_prompt, user_prompt = build_context_prompt(docs, "Tech stocks")
        
        assert "**AAPL:**" in user_prompt
        assert "**GOOGL:**" in user_prompt

if __name__ == "__main__":
    pytest.main([__file__, "-v"])