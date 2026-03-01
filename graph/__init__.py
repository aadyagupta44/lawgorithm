"""
Graph package for CodeMind.

This package builds and exposes the compiled RAG StateGraph used to
coordinate retrieval, generation, and self-correction. Importing the
package gives access to `build_graph` and `run_graph` convenience functions.
"""

from .workflow import build_graph, run_graph  # noqa: F401

__all__ = ["build_graph", "run_graph"]
