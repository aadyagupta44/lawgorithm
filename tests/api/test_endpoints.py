import pytest
import io
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

def test_ingest_text_happy(api_client, mock_pinecone, mock_embeddings):
    """Happy Path: Ingesting text successfully."""
    response = api_client.post(
        "/ingest/text",
        json={
            "text": "Valid test contract text.",
            "document_name": "Contract.pdf"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Text ingested successfully."
    assert "namespace" in data

def test_ingest_text_sad_missing_payload(api_client):
    """Sad Path: Missing payload properties triggers 422 Unprocessable Entity."""
    # Missing 'text'
    response = api_client.post(
        "/ingest/text",
        json={"document_name": "Contract.pdf"}
    )
    assert response.status_code == 422

def test_ingest_file_sad_wrong_type(api_client):
    """Sad Path: Uploading a non-PDF file results in 400 Bad Request."""
    fake_file = io.BytesIO(b"dummy image data")
    response = api_client.post(
        "/ingest/file",
        files={"file": ("image.png", fake_file, "image/png")}
    )
    
    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported."

@patch("api.run_graph")
def test_query_happy(mock_run_graph, api_client):
    """Happy Path: Query endpoint wrapper returns 200 properly."""
    # Mock LangGraph execution
    mock_run_graph.return_value = {
        "generation": "It is legal.",
        "source_files": ["Contract.pdf"],
        "source_pages": [1],
        "loop_count": 0,
        "confidence_score": 0.9,
        "relevance_score": 0.8,
        "hallucination_score": 0.1
    }
    
    response = api_client.post(
        "/query",
        json={
            "question": "Is this legal?",
            "namespace": "contract-pdf"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["generation"] == "It is legal."
    assert data["source_files"] == ["Contract.pdf"]

def test_query_sad_missing_namespace(api_client):
    """Sad Path: Query with missing required fields triggers 422."""
    response = api_client.post(
        "/query",
        json={
            "question": "Is this legal?"
        }
    )
    
    assert response.status_code == 422
