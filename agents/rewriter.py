"""
Query rewriter agent for self-correction and optimization.

This module implements `QueryRewriter` which uses the Groq LLM to rewrite
user queries into more specific, technical forms aimed at improving
vector retrieval quality. If the LLM fails, the original question is returned.
"""

from langchain_groq import ChatGroq
from config import GROQ_API_KEY, GROQ_MODEL


class QueryRewriter:
	"""Agent that rewrites a user question to improve retrieval."""

	def __init__(self) -> None:
		"""Initialize ChatGroq client for rewriting tasks."""
		try:
			self.llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)
			print(f"[INFO] QueryRewriter initialized with model {GROQ_MODEL}")
		except Exception as e:
			self.llm = None  # type: ignore[assignment]
			print(f"[ERROR] Failed to initialize QueryRewriter LLM: {e}")

	def rewrite(self, question: str) -> str:
		"""Rewrite the `question` to be more specific and technical for retrieval.

		Parameters:
			question: The original user question string.

		Returns:
			A rewritten question string, or the original question on error.
		"""

		if not self.llm:
			print("[WARN] QueryRewriter LLM not available — returning original question")
			return question

		system_prompt = (
			"You are a query rewriter. Given a user's natural-language question about a codebase, "
			"produce a more specific, technical rewrite optimized for semantic search over code and "
			"documentation. Return only the rewritten question." 
		)

		try:
			response = self.llm.invoke([
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": question},
			])
			rewritten = response.content.strip()
			if not rewritten:
				return question
			return rewritten
		except Exception as e:
			print(f"[ERROR] QueryRewriter failed: {e}")
			return question

