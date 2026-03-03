"""
Deadline and time obligation extractor agent.

This module implements `DeadlineExtractorAgent` which uses the Google Gemini LLM
to find all dates, deadlines, and time-based obligations in legal documents.
"""

from typing import List, Dict
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, GROQ_MODEL


class DeadlineExtractorAgent:
    """Agent that extracts deadlines and time obligations from legal documents."""

    def __init__(self) -> None:
        """Initialize ChatGroq client for deadline extraction."""
        try:
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL
            )
            print(f"[INFO] DeadlineExtractorAgent initialized with model {GROQ_MODEL}")
        except Exception as e:
            self.llm = None  # type: ignore[assignment]
            print(f"[ERROR] Failed to initialize DeadlineExtractorAgent LLM: {e}")

    def extract(self, document_text: str) -> List[Dict[str, str]]:
        """Extract all deadlines and time obligations from the legal document.

        Parameters:
            document_text: The full text of a legal document.

        Returns:
            A list of dictionaries, each containing:
            - description: What the deadline is for (e.g., 'Payment due date')
            - timeframe: The actual time period or date (e.g., '30 days', 'December 31, 2024')
            - importance: 'critical', 'important', or 'note'
            
        Examples of deadlines to find:
        - Payment due dates and invoice terms
        - Notice periods for termination
        - Contract duration and term
        - Renewal and expiration dates
        - Non-compete duration and restrictions
        - Probation periods
        - Delivery timelines
        - Performance milestones
        - Warranty periods
        """
        if not self.llm:
            print("[WARN] DeadlineExtractorAgent LLM not available — returning empty list")
            return []

        if not document_text or not document_text.strip():
            print("[WARN] Empty document text provided — returning empty list")
            return []

        system_prompt = (
            "You are a legal deadline extractor. Analyze the provided legal document "
            "and identify ALL dates, deadlines, and time-based obligations. "
            "Return a JSON array of objects, where each object has these fields:\n"
            "1. description: What this deadline or time obligation is for "
            "(e.g., 'Payment due date', 'Notice period for termination', "
            "'Contract renewal date', 'Non-compete duration')\n"
            "2. timeframe: The actual time period, date, or duration "
            "(e.g., '30 days', 'December 31, 2024', '2 years', '90 days after signing')\n"
            "3. importance: Rate the importance as:\n"
            "   - 'critical': Miss this and there are serious consequences (penalties, "
            "     termination, legal action, loss of rights)\n"
            "   - 'important': Should be noted but not critical\n"
            "   - 'note': Informational, low impact if missed\n"
            "Be thorough - find every time-sensitive clause in the document. "
            "If no deadlines are found, return an empty array."
        )

        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract all deadlines and time obligations from this document:\n\n{document_text}"},
            ])
            
            content = str(response.content).strip()
            
            # Handle potential markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            content = content.strip()
            
            # Try to parse as JSON
            import json
            try:
                result = json.loads(content)
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                print("[WARN] DeadlineExtractorAgent could not parse response as JSON")
            
            return []
            
        except Exception as e:
            print(f"[ERROR] DeadlineExtractorAgent failed: {e}")
            return []
