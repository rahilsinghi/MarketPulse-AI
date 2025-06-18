"""
Tests for utility functions.
"""

import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os

from utils import embed_text, validate_embedding, validate_news_record

class TestEmbedding:
    """Test embedding functionality."""
    
    @patch('utils.openai.Embedding.create')
    def test_embed_text_success(self, mock_create):
        """Test successful text embedding."""
        mock_response = {
            "data": [{"embedding": [0.1] * 1536}]
        }
        mock_create.return_value = mock_response
        
        result = embed_text("test text")
        assert len(result) == 1536
        assert all(isinstance(x, float) for x in result)
    
    @patch('utils.openai.Embedding.create')
    def test_embed_text_empty_input(self, mock_create):
        """Test embedding with empty input."""
        result = embed_text("")
        assert result is None
        mock_create.assert_not_called()
    
    @patch('utils.openai.Embedding.create')
    def test_embed_text_long_input(self, mock_create):
        """Test embedding with very long input."""
        long_text = "word " * 2000  # Very long text
        mock_response = {
            "data": [{"embedding": [0.1] * 1536}]
        }
        mock_create.return_value = mock_response
        
        result = embed_text(long_text)
        # Should truncate and still work
        assert len(result) == 1536

class TestValidation:
    """Test validation functions."""
    
    def test_validate_embedding_valid(self):
        """Test validation of valid embeddings."""
        valid_embedding = [0.1] * 1536
        assert validate_embedding(valid_embedding) is True
    
    def test_validate_embedding_wrong_dimension(self):
        """Test validation of wrong dimension embedding."""
        wrong_dim = [0.1] * 100
        assert validate_embedding(wrong_dim) is False
    
    def test_validate_embedding_invalid_type(self):
        """Test validation of invalid embedding types."""
        assert validate_embedding(None) is False
        assert validate_embedding("not a list") is False
        assert validate_embedding([]) is False
    
    def test_validate_news_record_valid(self):
        """Test validation of valid news record."""
        valid_record = {
            "id": "test_id",
            "ticker": "AAPL", 
            "headline": "Test headline",
            "timestamp": "2024-01-01T00:00:00"
        }
        assert validate_news_record(valid_record) is True
    
    def test_validate_news_record_missing_fields(self):
        """Test validation of incomplete news record."""
        incomplete_record = {
            "id": "test_id",
            "ticker": "AAPL"
            # Missing headline and timestamp
        }
        assert validate_news_record(incomplete_record) is False

if __name__ == "__main__":
    pytest.main([__file__, "-v"])