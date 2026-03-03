"""
Graph package for Lawgorithm — Intelligent Legal Document Assistant.

This package builds and exposes the compiled StateGraph used to coordinate
legal document retrieval, answer generation, and self-correction. Importing
the package gives access to `build_graph` and `run_graph` convenience functions.
"""

# Import graph building and execution functions from workflow module
from .workflow import build_graph, run_graph  # noqa: F401

# Define public API - what external modules can import from this package
__all__ = ["build_graph", "run_graph"]
