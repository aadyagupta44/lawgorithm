"""
Router agent for query relevance classification and query enhancement.

This module implements `RouterAgent` which uses the LLM to judge whether a user
question pertains to legal documents. If it is relevant, it uses the HyDE
(Hypothetical Document Embeddings) technique to enhance the query.

It returns a tuple: (decision, enhanced_query).
"""

from typing import Tuple
from utils import get_llm


class RouterAgent:
    """Agent that classifies questions and enhances them using HyDE.

    The agent provides a `route` method that returns ('relevant', enhanced_query) 
    or ('irrelevant', '').
    """

    def __init__(self) -> None:
        """Initialize the language model client."""
        try:
            self.llm = get_llm()
            print("[INFO] RouterAgent initialized with dynamic LLM provider")
        except Exception as e:
            self.llm = None  # type: ignore[assignment]
            print(f"[ERROR] Failed to initialize RouterAgent LLM: {e}")

    def route(self, question: str) -> Tuple[str, str]:
        """Decide if `question` is relevant, and if so, enhance it via HyDE.

        Parameters:
            question: The user's original question string.

        Returns:
            Tuple of (decision, enhanced_query):
            - decision: 'relevant' or 'irrelevant'
            - enhanced_query: The query optionally appended with a hypothetical document.
        """
        if not self.llm:
            print("[WARN] Router LLM not available — defaulting to 'relevant' and original query")
            return "relevant", question

        # 1. Classification
        classification_prompt = (
            "You are a router that classifies whether a user's question pertains to legal documents. "
            "Legal documents include contracts, agreements, clauses, terms of service, "
            "employment agreements, NDAs, leases, and any other legal content. "
            "Respond ONLY with 'relevant' if it pertains to legal matters. Respond ONLY with 'irrelevant' otherwise."
        )

        try:
            response = self.llm.invoke([
                {"role": "system", "content": classification_prompt},
                {"role": "user", "content": question}
            ])
            decision = str(response.content).strip().lower()
            if "irrelevant" in decision:
                return "irrelevant", ""
            if "relevant" not in decision:
                return "irrelevant", ""
        except Exception as e:
            print(f"[ERROR] Router classification failed: {e}")
            return "relevant", question

        # 2. HyDE Enhancement (if relevant)
        hyde_prompt = (
            "You are a legal document expert. Given a legal question, generate a brief hypothetical "
            "legal document or clause that would directly answer the question. The document should be "
            "realistic and contain key legal terms. Keep it concise (100-150 words) and relevant.\n\n"
            "Output ONLY the hypothetical document text, with no extra explanation."
        )

        try:
            hyde_response = self.llm.invoke([
                {"role": "system", "content": hyde_prompt},
                {"role": "user", "content": question}
            ])
            hypothetical_doc = str(hyde_response.content).strip()
            
            enhanced_query = f"{question}\n\nExample document:\n{hypothetical_doc}"
            print(f"[INFO] Router identified query as relevant and generated HyDE enhancement.")
            return "relevant", enhanced_query
            
        except Exception as e:
            print(f"[WARN] HyDE enhancement failed: {e}")
            return "relevant", question
