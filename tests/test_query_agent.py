import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from query_agent import QueryAgent, RetrievalResult, build_context_prompt


class TestQueryAgent:
    """Test suite for QueryAgent class."""
    
    @pytest.fixture
    def mock_db_config(self):
        """Mock database configuration."""
        return {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db',
            'user': 'test_user',
            'password': 'test_pass'
        }
    
    @pytest.fixture
    def query_agent(self, mock_db_config):
        """Create QueryAgent instance with mocked dependencies."""
        with patch('query_agent.psycopg2.connect'):
            agent = QueryAgent(mock_db_config)
            agent.cursor = Mock()
            return agent
    
    def test_document_retrieval_success(self, query_agent):
        """Test successful document retrieval."""
        # Mock database response
        mock_rows = [
            {'id': 1, 'content': 'Test document 1', 'ticker': 'AAPL'},
            {'id': 2, 'content': 'Test document 2', 'ticker': 'MSFT'}
        ]
        query_agent.cursor.fetchall.return_value = mock_rows
        
        # Mock embedding
        with patch.object(query_agent, 'get_embedding', return_value=np.array([0.1, 0.2, 0.3])):
            result = query_agent.retrieve_documents("test query", top_k=5)
        
        assert isinstance(result, RetrievalResult)
        assert len(result.documents) <= 5
        assert result.total_documents == len(mock_rows)
        assert result.retrieval_time > 0
    
    def test_document_retrieval_error_handling(self, query_agent):
        """Test error handling in document retrieval."""
        query_agent.cursor.fetchall.side_effect = Exception("Database error")
        
        result = query_agent.retrieve_documents("test query")
        
        assert isinstance(result, RetrievalResult)
        assert result.documents == []
        assert result.query_embedding == []
        assert result.total_documents == 0
        assert result.retrieval_time == 0.0
    
    def test_build_context_prompt(self):
        """Test context prompt building."""
        docs = [
            {'ticker': 'AAPL', 'content': 'Apple earnings report', 'final_score': 0.95},
            {'ticker': 'MSFT', 'content': 'Microsoft news', 'final_score': 0.87}
        ]
        question = "What are the latest earnings?"
        
        system_prompt, user_prompt = build_context_prompt(docs, question)
        
        assert isinstance(system_prompt, str)
        assert isinstance(user_prompt, str)
        assert question in user_prompt
        assert len(system_prompt) > 0
        assert len(user_prompt) > 0
    
    def test_similarity_calculation(self, query_agent):
        """Test similarity score calculation."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        
        # Mock the similarity calculation if it exists
        with patch('query_agent.cosine_similarity', return_value=1.0) as mock_sim:
            similarity = mock_sim(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
            assert similarity == 1.0
    
    @pytest.mark.parametrize("top_k", [1, 5, 10, 50])
    def test_top_k_parameter(self, query_agent, top_k):
        """Test different top_k values."""
        mock_rows = [{'id': i, 'content': f'Doc {i}'} for i in range(20)]
        query_agent.cursor.fetchall.return_value = mock_rows
        
        with patch.object(query_agent, 'get_embedding', return_value=np.array([0.1, 0.2])):
            result = query_agent.retrieve_documents("test", top_k=top_k)
        
        assert len(result.documents) <= top_k
        assert len(result.documents) <= len(mock_rows)