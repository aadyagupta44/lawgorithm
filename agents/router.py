"""
Router agent for query relevance classification.

This module implements `RouterAgent` which uses the Google Gemini LLM (via
LangChain's Google GenAI wrapper) to judge whether a user question pertains to
legal documents or is out of scope. The agent returns a simple string decision:
'relevant' or 'irrelevant'.
"""

from typing import Any
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, GROQ_MODEL


class RouterAgent:
    """Agent that classifies whether a question should trigger legal document retrieval.

    The agent provides a `route` method that returns 'relevant' or
    'irrelevant' based on the content of the question.
    """

    def __init__(self) -> None:
        """Initialize the ChatGroq client used for routing decisions."""
        try:
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL
            )
            print(f"[INFO] RouterAgent initialized with model {GROQ_MODEL}")
        except Exception as e:
            self.llm = None  # type: ignore[assignment]
            print(f"[ERROR] Failed to initialize RouterAgent LLM: {e}")

    def route(self, question: str) -> str:
        """Decide if `question` is relevant to legal documents.

        Parameters:
            question: The user's original question string.

        Returns:
            'relevant' if the question is about legal documents, contracts, 
            or legal matters; otherwise 'irrelevant'.
        """
        # default to 'relevant' in case of any errors - legal queries should be answered
        default_decision = "relevant"

        if not self.llm:
            print("[WARN] Router LLM not available — defaulting to 'relevant'")
            return default_decision

        # system prompt instructing the model on its role for legal document routing
        system_prompt = (
            "You are a router that classifies whether a user's question pertains to legal documents. "
            "Legal documents include contracts, agreements, legal clauses, terms of service, "
            "employment agreements, NDAs, leases, and any other legal or contract-related content. "
            "Respond only with the single word 'relevant' if the question is about legal matters, "
            "contracts, legal clauses, terms, obligations, rights, or legal document analysis. "
            "Otherwise respond only with 'irrelevant'."
        )

        try:
            # call the model with a simple chat-style invocation
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ])
            # normalize model response to a simple decision
            text = str(response.content).strip().lower()
            if "relevant" in text:
                return "relevant"
            else:
                return "irrelevant"
        except Exception as e:
            print(f"[ERROR] RouterAgent failed to classify question: {e}")
            return default_decision
