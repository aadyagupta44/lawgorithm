"""
Contract favorability analyzer agent.

This module implements `FavorabilityAgent` which uses the Google Gemini LLM
to analyze a contract from a specific user's perspective and determine
how favorable the contract terms are for them.
"""

from typing import Dict, List
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, GROQ_MODEL


class FavorabilityAgent:
    """Agent that analyzes contract favorability from a user's perspective."""

    def __init__(self) -> None:
        """Initialize ChatGroq client for favorability analysis."""
        try:
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL
            )
            print(f"[INFO] FavorabilityAgent initialized with model {GROQ_MODEL}")
        except Exception as e:
            self.llm = None  # type: ignore[assignment]
            print(f"[ERROR] Failed to initialize FavorabilityAgent LLM: {e}")

    def analyze(self, document_text: str, user_role: str = "employee") -> Dict[str, any]:
        """Analyze the contract from the specified user's perspective.

        Parameters:
            document_text: The full text of a legal contract or agreement.
            user_role: The perspective to analyze from. Valid values:
                - 'employee' (default): Analyzing an employment contract
                - 'tenant': Analyzing a lease or rental agreement
                - 'client': Analyzing a contract where user is the customer/client
                - 'vendor': Analyzing a contract where user is the service provider

        Returns:
            A dictionary containing:
            - overall_score: 1-10 rating (10 = very favorable for user)
            - favorable_clauses: List of clauses that help the user
            - unfavorable_clauses: List of clauses that hurt the user
            - neutral_clauses: List of balanced or standard clauses
            - verdict: One sentence overall assessment of the contract
        """
        if not self.llm:
            print("[WARN] FavorabilityAgent LLM not available — returning default analysis")
            return self._default_response()

        if not document_text or not document_text.strip():
            print("[WARN] Empty document text provided — returning default analysis")
            return self._default_response()

        # Validate user_role
        valid_roles = ["employee", "tenant", "client", "vendor"]
        if user_role not in valid_roles:
            print(f"[WARN] Invalid user_role '{user_role}' — defaulting to 'employee'")
            user_role = "employee"

        system_prompt = (
            f"You are a contract favorability analyzer. Analyze the provided contract "
            f"from the perspective of a {user_role}. Your job is to identify how favorable "
            f"or unfavorable each major clause is for the {user_role}. "
            "Return a JSON object with exactly these fields:\n"
            "1. overall_score: A number from 1-10 where 10 means the contract is very "
            "favorable for the user and 1 means it's very unfavorable. Consider all "
            "clauses together.\n"
            "2. favorable_clauses: A list of clauses or terms that are favorable "
            f"to the {user_role} (include brief descriptions)\n"
            "3. unfavorable_clauses: A list of clauses or terms that are unfavorable "
            f"to the {user_role} (include brief descriptions)\n"
            "4. neutral_clauses: A list of clauses that are balanced or standard "
            "(not particularly favorable or unfavorable)\n"
            "5. verdict: One sentence that summarizes your overall assessment of "
            "this contract for the user.\n"
            "Be honest and objective. Focus on practical implications, not just legal technicalities."
        )

        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this contract from a {user_role}'s perspective:\n\n{document_text}"},
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
                # Validate and ensure all keys exist
                return {
                    "overall_score": self._validate_score(result.get("overall_score", 5)),
                    "favorable_clauses": result.get("favorable_clauses", []),
                    "unfavorable_clauses": result.get("unfavorable_clauses", []),
                    "neutral_clauses": result.get("neutral_clauses", []),
                    "verdict": result.get("verdict", "Analysis complete.")
                }
            except json.JSONDecodeError:
                print("[WARN] FavorabilityAgent could not parse response as JSON")
            
            return self._default_response()
            
        except Exception as e:
            print(f"[ERROR] FavorabilityAgent failed: {e}")
            return self._default_response()

    def _validate_score(self, score: any) -> int:
        """Validate and normalize the overall score to 1-10 range."""
        try:
            score = int(score)
            return max(1, min(10, score))
        except (ValueError, TypeError):
            return 5

    def _default_response(self) -> Dict[str, any]:
        """Return a default response when analysis fails."""
        return {
            "overall_score": 5,
            "favorable_clauses": [],
            "unfavorable_clauses": [],
            "neutral_clauses": [],
            "verdict": "Unable to analyze contract."
        }
