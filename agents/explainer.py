"""
Plain English legal explainer agent.

This module implements `PlainEnglishExplainer` which uses the Google Gemini LLM
to translate complex legal language into simple, understandable English for
non-lawyers. The agent never loses the original meaning while simplifying.
"""

from langchain_groq import ChatGroq
from config import GROQ_API_KEY, GROQ_MODEL


class PlainEnglishExplainer:
    """Agent that explains legal text in plain, simple English."""

    def __init__(self) -> None:
        """Initialize ChatGroq client for explanation tasks."""
        try:
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL
            )
            print(f"[INFO] PlainEnglishExplainer initialized with model {GROQ_MODEL}")
        except Exception as e:
            self.llm = None  # type: ignore[assignment]
            print(f"[ERROR] Failed to initialize PlainEnglishExplainer LLM: {e}")

    def explain(self, legal_text: str) -> str:
        """Explain the given legal text in plain English.

        Parameters:
            legal_text: The legal clause or text to explain.

        Returns:
            A simple English explanation aimed at someone with zero legal knowledge,
            without losing the original meaning.
        """
        if not self.llm:
            print("[WARN] PlainEnglishExplainer LLM not available — returning original text")
            return legal_text

        system_prompt = (
            "You are a legal translator who explains complex legal language in simple, "
            "plain English. Your goal is to help someone with ZERO legal knowledge "
            "understand what a legal clause means in practice. "
            "Follow these rules:\n"
            "1. Use simple words (avoid legal jargon like 'heretofore', 'pursuant to', etc.)\n"
            "2. Use short sentences\n"
            "3. Explain WHO is affected, WHAT they must do, and WHEN they must do it\n"
            "4. Never lose the original legal meaning - be accurate above all\n"
            "5. If there are potential consequences (like fines, liability, etc.), explain them clearly\n"
            "6. Use a friendly, helpful tone\n"
            "Format your response as a clear, easy-to-read explanation."
        )

        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Explain this legal text in plain English:\n\n{legal_text}"},
            ])
            explanation = str(response.content).strip()
            if not explanation:
                return legal_text
            return explanation
        except Exception as e:
            print(f"[ERROR] PlainEnglishExplainer failed: {e}")
            return legal_text
