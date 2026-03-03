"""
Ingestion package for CodeMind.

- This package contains the components used to ingest local documents (PDF/text):
- `DocumentLoader` reads PDFs and text into page-level dictionaries
- `DocumentChunker` splits text into overlapping chunks for embedding
- `PineconeEmbedder` computes embeddings and stores them in Pinecone

Importing the package exposes the three primary classes for use in
the higher-level ingestion pipeline and tests.
"""

# export classes from submodules for convenient imports
from .document_loader import DocumentLoader  # noqa: F401  (re-export)
from .chunker import DocumentChunker  # noqa: F401  (re-export)
from .embedder import PineconeEmbedder  # noqa: F401  (re-export)

__all__ = ["DocumentLoader", "DocumentChunker", "PineconeEmbedder"]
