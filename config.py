"""
Configuration module for Lawgorithm — Intelligent Legal Document Assistant.

This file loads sensitive configuration from environment variables
and defines hardcoded constants used throughout the application.
"""

import os  # access environment variables
from dotenv import load_dotenv  # load .env file into environment

# load environment variables from .env file at project root
load_dotenv()

# -------------------------
# Environment-provided keys
# -------------------------

# Google Gemini API key — loaded from environment, never hardcoded
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

# Groq API key — loaded from environment, never hardcoded
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")

# Pinecone vector database API key
PINECONE_API_KEY: str | None = os.getenv("PINECONE_API_KEY")

# Pinecone index name — defaults to lawgorithm if not set
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "lawgorithm")

# -------------------------
# Hardcoded constants
# -------------------------

# Gemini model — flash variant is fast and capable for legal tasks
GEMINI_MODEL: str = "gemini-2.0-flash-lite"

# Groq model — versatile model good for legal analysis tasks
GROQ_MODEL: str = "llama-3.3-70b-versatile"

# HuggingFace embedding model — compact and effective for semantic search
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# Vector dimension — must match Pinecone index configuration exactly
PINECONE_DIMENSION: int = 384

# Chunk size in characters — good for capturing full legal clauses
CHUNK_SIZE: int = 1000

# Overlap between chunks — prevents clauses being cut at boundaries
CHUNK_OVERLAP: int = 200

# Maximum self-correction attempts before accepting current answer
MAX_LOOP_COUNT: int = 3

# Number of chunks to retrieve from Pinecone per query
TOP_K_RESULTS: int = 5

# Maximum chat history turns to keep for conversational memory
MAX_CHAT_HISTORY: int = 10
