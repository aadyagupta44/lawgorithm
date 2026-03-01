"""
Configuration module for CodeMind project.

This file is responsible for loading environment variables from a
`.env` file (via python-dotenv) and defining both dynamic and
hardcoded configuration constants used throughout the application.
All values are typed, and sensible defaults are provided where
appropriate to avoid crashes when environment variables are missing.
"""

import os  # operating system utilities for environment access
from dotenv import load_dotenv  # helper to load .env files into env vars

# load environment variables from a .env file located at project root
load_dotenv()

# Dynamic configuration values read from environment variables
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")  # API key for Groq LLM
PINECONE_API_KEY: str | None = os.getenv("PINECONE_API_KEY")  # Pinecone service key
GITHUB_TOKEN: str | None = os.getenv("GITHUB_TOKEN")  # GitHub personal access token
# allow specifying a custom index name; default to 'codemind' if unset
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "codemind")

# Hardcoded constants used across the application
PINECONE_DIMENSION: int = 384  # dimensionality of embedding vectors
GROQ_MODEL: str = "llama-3.3-70b-versatile"  # default LLM model for generation
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # embedding model name
ALLOWED_EXTENSIONS: list[str] = [".md", ".txt", ".rst", ".py"]  # file types to ingest
CHUNK_SIZE: int = 1000  # size of text chunks for splitting documents
CHUNK_OVERLAP: int = 200  # overlap between text chunks to preserve context
MAX_LOOP_COUNT: int = 3  # maximum number of self-correction iterations
TOP_K_RESULTS: int = 5  # number of top documents to retrieve from Pinecone

