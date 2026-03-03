"""
Query rewriter agent for self-correction and optimization.

This module implements `QueryRewriter` which uses the Google Gemini LLM to rewrite
user queries into more specific, legal-focused forms aimed at improving
vector retrieval quality for legal documents. If the LLM fails, the original 
question is returned.
"""

from langchain_groq import ChatGroq
from config import GROQ_API_KEY, GROQ_MODEL


class QueryRewriter:
    """Agent that rewrites a user question to improve legal document retrieval."""

    def __init__(self) -> None:
        """Initialize ChatGroq client for rewriting tasks."""
        try:
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL
            )
            print(f"[INFO] QueryRewriter initialized with model {GROQ_MODEL}")
        except Exception as e:
            self.llm = None  # type: ignore[assignment]
            print(f"[ERROR] Failed to initialize QueryRewriter LLM: {e}")

    def rewrite(self, question: str) -> str:
        """Rewrite the `question` to be more specific and technical for legal retrieval.

        Parameters:
            question: The original user question string.

        Returns:
            A rewritten question string optimized for legal document retrieval,
            or the original question on error.
        """
        if not self.llm:
            print("[WARN] QueryRewriter LLM not available — returning original question")
            return question

        system_prompt = (
            "You are a query rewriter specialized in legal document retrieval. "
            "Given a user's natural-language question about legal matters, contracts, "
            "or legal clauses, produce a more specific, technical rewrite optimized for "
            "semantic search over legal documents. Use proper legal terminology. "
            "Examples:\n"
            "- 'what if I quit' -> 'employee termination clause notice period obligations resignation'\n"
            "- 'can they sue me' -> 'liability clause indemnification third party claims damages'\n"
            "- 'who owns what I create' -> 'intellectual property assignment work for hire ownership rights'\n"
            "Return only the rewritten question. Do not include explanations or quotes."
        )

        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ])
            rewritten = str(response.content).strip()
            if not rewritten:
                return question
            return rewritten
        except Exception as e:
            print(f"[ERROR] QueryRewriter failed: {e}")
            return question
