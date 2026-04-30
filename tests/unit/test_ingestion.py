import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from ingestion.document_loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from ingestion.embedder import PineconeEmbedder

def test_document_loader_text(sample_document):
    """Happy Path: Test loading text returns proper page structure."""
    loader = DocumentLoader()
    result = loader.load_text(sample_document, "test_doc.txt")
    
    assert len(result) == 1
    page = result[0]
    assert page["filename"] == "test_doc.txt"
    assert page["content"] == sample_document
    assert page["page_number"] == 1
    assert page["file_type"] == "text"
    assert "document_id" in page

def test_document_loader_pdf_not_found():
    """Sad Path: Loading unexisting PDF gracefully returns empty list."""
    loader = DocumentLoader()
    result = loader.load_pdf("nonexistent_file.pdf")
    assert len(result) == 0

@patch("ingestion.document_loader.fitz.open")
def test_document_loader_pdf_happy(mock_fitz_open):
    """Happy Path: Mocking PyMuPDF to return pages."""
    mock_doc = MagicMock()
    mock_doc.page_count = 2
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Mocked PDF text"
    mock_doc.load_page.return_value = mock_page
    mock_doc.is_encrypted = False
    
    mock_fitz_open.return_value = mock_doc
    
    loader = DocumentLoader()
    result = loader.load_pdf("fake.pdf")
    
    assert len(result) == 2
    assert result[0]["content"] == "Mocked PDF text"
    assert result[0]["file_type"] == "pdf"
    assert result[0]["total_pages"] == 2

@patch("ingestion.chunker.CHUNK_SIZE", 10)
@patch("ingestion.chunker.CHUNK_OVERLAP", 2)
def test_chunker_basic():
    """Happy path: chunking a page splits properly."""
    chunker = DocumentChunker()
    page = {
        "content": "This is a simple little test string for chunking.",
        "filename": "test.txt",
        "page_number": 1,
        "document_id": "123",
        "file_type": "text",
        "total_pages": 1
    }
    
    # Text length is 49. Chunk size 10 means multiple chunks.
    chunks = chunker.chunk_documents([page])
    
    assert len(chunks) > 1
    # Check metadata was preserved
    assert chunks[0]["filename"] == "test.txt"
    assert chunks[0]["page_number"] == 1

def test_chunker_empty():
    """Sad path: chunking empty docs."""
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents([])
    assert len(chunks) == 0

def test_pinecone_embedder_namespace():
    """Test namespace generation is robust."""
    embedder = PineconeEmbedder()
    ns1 = embedder.namespace_from_filename("My Contract 2023!.pdf")
    ns2 = embedder.namespace_from_filename("   ")
    
    assert ns1 == "My_Contract_2023_"
    assert ns2 == "default_namespace"

def test_pinecone_embedder_store():
    """Happy path: test embedding storage returns correct count."""
    embedder = PineconeEmbedder()
    mock_idx = MagicMock()
    embedder.index = mock_idx
    embedder.model = MagicMock()
    # Return numpy array so .tolist() works correctly
    embedder.model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
    
    mock_chunks = [{"content": "abc"}, {"content": "def"}]
    result = embedder.embed_and_store(mock_chunks, "test-ns")
    assert result == 2
    mock_idx.upsert.assert_called_once()