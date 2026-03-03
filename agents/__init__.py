"""
Agents package for Lawgorithm — Intelligent Legal Document Assistant.

This package exposes the high-level agent classes used by the graph:
- `RouterAgent` decides whether a question is relevant to legal documents
- `RelevanceGrader`, `HallucinationGrader`, `AnswerGrader`, `RiskFlagGrader` provide LLM-based judgments
- `QueryRewriter` rewrites queries for improved legal document retrieval
- `PlainEnglishExplainer` explains legal text in plain English
- `ClauseIdentifierAgent` identifies and extracts major clauses from documents
- `ContractSummarizerAgent` creates structured summaries of contracts
- `ComparisonAgent` compares two legal documents
- `DeadlineExtractorAgent` extracts deadlines and time obligations
- `FavorabilityAgent` analyzes contract favorability from user perspective
- `RedlineAgent` suggests improvements to unfavorable clauses

Importing these names from the package gives convenient access to agents
when constructing the graph and node functions.
"""

from .router import RouterAgent  # noqa: F401
from .graders import (  # noqa: F401
    RelevanceGrader,
    HallucinationGrader,
    AnswerGrader,
    RiskFlagGrader
)
from .rewriter import QueryRewriter  # noqa: F401
from .explainer import PlainEnglishExplainer  # noqa: F401
from .clause_identifier import ClauseIdentifierAgent  # noqa: F401
from .summarizer import ContractSummarizerAgent  # noqa: F401
from .comparison import ComparisonAgent  # noqa: F401
from .deadline_extractor import DeadlineExtractorAgent  # noqa: F401
from .favorability import FavorabilityAgent  # noqa: F401
from .redline import RedlineAgent  # noqa: F401

__all__ = [
    "RouterAgent",
    "RelevanceGrader",
    "HallucinationGrader",
    "AnswerGrader",
    "RiskFlagGrader",
    "QueryRewriter",
    "PlainEnglishExplainer",
    "ClauseIdentifierAgent",
    "ContractSummarizerAgent",
    "ComparisonAgent",
    "DeadlineExtractorAgent",
    "FavorabilityAgent",
    "RedlineAgent",
]
