"""
Utility functions and helpers for Lawgorithm.

This module provides shared utility functions for logging, error handling,
and other common operations across the Lawgorithm system.
"""

# re-export the LLM factory function so callers can import from utils
from .llm_factory import get_llm

__all__ = ["get_llm"]
