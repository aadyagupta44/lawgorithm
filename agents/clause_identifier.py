"""
Legal clause identifier agent.

This module implements `ClauseIdentifierAgent` which uses the Google Gemini LLM
to identify and extract major legal clauses from document text. It recognizes
common contract clause types and returns them in a structured dictionary.
"""

from typing import Dict
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, GROQ_MODEL


class ClauseIdentifierAgent:
    """Agent that identifies and labels major clauses in legal documents."""

    def __init__(self) -> None:
        """Initialize ChatGroq client for clause identification."""
        try:
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL
            )
            print(f"[INFO] ClauseIdentifierAgent initialized with model {GROQ_MODEL}")
        except Exception as e:
            self.llm = None  # type: ignore[assignment]
            print(f"[ERROR] Failed to initialize ClauseIdentifierAgent LLM: {e}")

    def identify(self, document_text: str) -> Dict[str, str]:
        """Identify and extract major clauses from the given legal document.

        Parameters:
            document_text: The full text of a legal document or contract.

        Returns:
            A dictionary with clause names as keys and extracted clause text as values.
            Returns an empty dict if no clauses are found or on error.
        
        Clause types searched for:
        - termination: How and when the contract can be ended
        - liability: Who is responsible for what, limits on damages
        - payment: Payment terms, amounts, schedules
        - confidentiality: Non-disclosure obligations, duration
        - jurisdiction: Which laws apply, where disputes are resolved
        - intellectual property: IP ownership, licensing, assignment
        - non-compete: Restrictions on competing business activities
        - indemnification: Who pays for certain losses or claims
        - dispute resolution: How disputes are handled (arbitration, courts, etc.)
        - warranties: Promises about the quality or condition of goods/services
        """
        if not self.llm:
            print("[WARN] ClauseIdentifierAgent LLM not available — returning empty dict")
            return {}

        if not document_text or not document_text.strip():
            print("[WARN] Empty document text provided — returning empty dict")
            return {}

        system_prompt = (
            "You are a legal document analyzer that identifies and extracts key clauses "
            "from contracts and legal documents. Read the provided document and identify "
            "all major clauses from this list:\n"
            "- termination (how and when the contract can be ended)\n"
            "- liability (who is responsible for what, limits on damages)\n"
            "- payment (payment terms, amounts, schedules, due dates)\n"
            "- confidentiality (non-disclosure obligations, duration of secrecy)\n"
            "- jurisdiction (which laws apply, where disputes are resolved)\n"
            "- intellectual property (IP ownership, licensing, work for hire)\n"
            "- non-compete (restrictions on competing business activities)\n"
            "- indemnification (who pays for certain losses or third-party claims)\n"
            "- dispute resolution (arbitration, mediation, court jurisdiction)\n"
            "- warranties (promises about quality, condition, or performance)\n"
            "For each clause found, extract the relevant text exactly as written. "
            "Return your response as a JSON object with the clause name as key "
            "(use the exact names listed above) and the clause text as value. "
            "If a clause type is not found, do not include it in the result. "
            "Only include clauses where you can extract meaningful text content."
        )

        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Identify all major clauses in this legal document:\n\n{document_text}"},
            ])
            
            # Try to parse as JSON
            content = str(response.content).strip()
            
            # Handle potential markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            content = content.strip()
            
            # Try to import and use json
            import json
            try:
                result = json.loads(content)
                if isinstance(result, dict):
                    # Filter out empty values
                    return {k: v for k, v in result.items() if v and str(v).strip()}
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract key-value pairs manually
                pass
            
            # Return empty dict if parsing failed
            print("[WARN] ClauseIdentifierAgent could not parse response as JSON")
            return {}
            
        except Exception as e:
            print(f"[ERROR] ClauseIdentifierAgent failed: {e}")
            return {}
