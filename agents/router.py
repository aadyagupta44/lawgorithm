"""
Router agent for query relevance classification.

This module implements `RouterAgent` which uses the Groq LLM (via
LangChain's Groq wrapper) to judge whether a user question pertains to
the ingested codebase (documentation/code) or is out of scope. The
agent returns a simple string decision: 'relevant' or 'irrelevant'.
"""

from typing import Any
from pydantic import BaseModel
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, GROQ_MODEL


class RouterAgent:
	"""Agent that classifies whether a question should trigger retrieval.

	The agent provides a `route` method that returns 'relevant' or
	'irrelevant' based on the content of the question.
	"""

	def __init__(self) -> None:
		"""Initialize the ChatGroq client used for routing decisions."""
		try:
			self.llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)
			print(f"[INFO] RouterAgent initialized with model {GROQ_MODEL}")
		except Exception as e:
			self.llm = None  # type: ignore[assignment]
			print(f"[ERROR] Failed to initialize RouterAgent LLM: {e}")

	def route(self, question: str) -> str:
		"""Decide if `question` is relevant to the codebase.

		Parameters:
			question: The user's original question string.

		Returns:
			'relevant' if the question is about the repository; otherwise 'irrelevant'.
		"""

		# default to 'irrelevant' in case of any errors
		default_decision = "irrelevant"

		if not self.llm:
			print("[WARN] Router LLM not available — defaulting to 'irrelevant'")
			return default_decision

		# system prompt instructing the model on its role
		system_prompt = (
			"You are a router that classifies whether a user's question pertains to a codebase. "
			"Respond only with the single word 'relevant' if the question is about code, documentation, "
			"APIs, functions, or repository usage. Otherwise respond only with 'irrelevant'."
		)

		try:
			# call the model with a simple chat-style invocation
			response = self.llm.invoke([{"role": "system", "content": system_prompt},
										{"role": "user", "content": question}])
			# normalize model response to a simple decision
			text = str(response).strip().lower()
			if "relevant" in text:
				return "relevant"
			else:
				return "irrelevant"
		except Exception as e:
			print(f"[ERROR] RouterAgent failed to classify question: {e}")
			return default_decision

