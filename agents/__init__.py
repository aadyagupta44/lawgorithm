"""
Agents package for Lawgorithm — Intelligent Legal Document Assistant.

This package exposes the high-level agent classes used by the graph.
"""

from .router import RouterAgent
from .evaluate import EvaluateAgent

__all__ = [
    "RouterAgent",
    "EvaluateAgent",
]
