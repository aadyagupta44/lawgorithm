import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from api import app
import numpy as np

@pytest.fixture
def api_client():
    """Provides a TestClient for FastAPI endpoints."""
    return TestClient(app)

@pytest.fixture
def mock_pinecone():
    """Mocks the Pinecone embedder to prevent actual API calls."""
    with patch("ingestion.embedder.Pinecone") as mock_pc:
        mock_instance = MagicMock()
        mock_pc.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_google_genai():
    """Mocks LLM factory to prevent actual API calls."""
    with patch("utils.llm_factory.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        yield mock_llm

@pytest.fixture
def mock_embeddings():
    """Mocks the SentenceTransformer embeddings."""
    with patch("ingestion.embedder.SentenceTransformer") as mock_st:
        mock_instance = MagicMock()
        mock_st.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def sample_document():
    """Returns a simple generic document string."""
    return "This is a sample legal document. It contains an indemnification clause and standard termination conditions."