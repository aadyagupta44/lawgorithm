"""
Legal document comparison agent.

This module implements `ComparisonAgent` which uses the Google Gemini LLM
to compare two legal documents and identify similarities, differences,
and which document is more favorable for a user.
"""

from typing import Dict, List
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, GROQ_MODEL


class ComparisonAgent:
    """Agent that compares two legal documents and provides analysis."""

    def __init__(self) -> None:
        """Initialize ChatGroq client for comparison tasks."""
        try:
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL
            )
            print(f"[INFO] ComparisonAgent initialized with model {GROQ_MODEL}")
        except Exception as e:
            self.llm = None  # type: ignore[assignment]
            print(f"[ERROR] Failed to initialize ComparisonAgent LLM: {e}")

    def compare(
        self,
        doc1_text: str,
        doc2_text: str,
        doc1_name: str,
        doc2_name: str
    ) -> Dict[str, any]:
        """Compare two legal documents and provide detailed analysis.

        Parameters:
            doc1_text: The full text of the first legal document.
            doc2_text: The full text of the second legal document.
            doc1_name: A name or identifier for the first document.
            doc2_name: A name or identifier for the second document.

        Returns:
            A dictionary containing:
            - similarities: List of similar clauses found in both documents
            - differences: List of key differences between the documents
            - doc1_advantages: What is better in doc1
            - doc2_advantages: What is better in doc2
            - recommendation: Which document is more favorable and why (in plain English)
        """
        if not self.llm:
            print("[WARN] ComparisonAgent LLM not available — returning empty dict")
            return {
                "similarities": [],
                "differences": [],
                "doc1_advantages": [],
                "doc2_advantages": [],
                "recommendation": ""
            }

        if not doc1_text or not doc2_text:
            print("[WARN] Empty document text provided — returning empty comparison")
            return {
                "similarities": [],
                "differences": [],
                "doc1_advantages": [],
                "doc2_advantages": [],
                "recommendation": ""
            }

        system_prompt = (
            "You are a legal document comparison expert. Analyze two legal documents "
            "and identify their similarities, differences, and relative advantages. "
            "Return a JSON object with exactly these fields:\n"
            "1. similarities: A list of clauses or terms that are similar or the same "
            "in both documents (e.g., 'Both have confidentiality clauses')\n"
            "2. differences: A list of key differences between the documents "
            "(e.g., 'Doc1 has 30-day notice period, Doc2 has 60-day')\n"
            "3. doc1_advantages: What makes {doc1_name} more favorable "
            "(list specific advantages)\n"
            "4. doc2_advantages: What makes {doc2_name} more favorable "
            "(list specific advantages)\n"
            "5. recommendation: A clear recommendation in plain English about which "
            "document is more favorable and why (1-2 sentences)\n"
            "Be objective and focus on practical implications for someone signing "
            "or negotiating these documents."
        )

        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt.format(
                    doc1_name=doc1_name,
                    doc2_name=doc2_name
                )},
                {"role": "user", "content": f"Compare these two legal documents:\n\n"
                              f"=== {doc1_name} ===\n{doc1_text}\n\n"
                              f"=== {doc2_name} ===\n{doc2_text}"},
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
                return {
                    "similarities": result.get("similarities", []),
                    "differences": result.get("differences", []),
                    "doc1_advantages": result.get("doc1_advantages", []),
                    "doc2_advantages": result.get("doc2_advantages", []),
                    "recommendation": result.get("recommendation", "")
                }
            except json.JSONDecodeError:
                print("[WARN] ComparisonAgent could not parse response as JSON")
            
            return {
                "similarities": [],
                "differences": [],
                "doc1_advantages": [],
                "doc2_advantages": [],
                "recommendation": ""
            }
            
        except Exception as e:
            print(f"[ERROR] ComparisonAgent failed: {e}")
            return {
                "similarities": [],
                "differences": [],
                "doc1_advantages": [],
                "doc2_advantages": [],
                "recommendation": ""
            }
