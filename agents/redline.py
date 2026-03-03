"""
Redline suggestion agent for contract negotiation.

This module implements `RedlineAgent` which uses the Google Gemini LLM
to suggest specific improvements to unfavorable contract clauses.
"""

from typing import Dict
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, GROQ_MODEL


class RedlineAgent:
    """Agent that suggests improvements to unfavorable contract clauses."""

    def __init__(self) -> None:
        """Initialize ChatGroq client for redline suggestions."""
        try:
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL
            )
            print(f"[INFO] RedlineAgent initialized with model {GROQ_MODEL}")
        except Exception as e:
            self.llm = None  # type: ignore[assignment]
            print(f"[ERROR] Failed to initialize RedlineAgent LLM: {e}")

    def suggest(self, clause_text: str, context: str = "") -> Dict[str, str]:
        """Suggest improvements to an unfavorable clause.

        Parameters:
            clause_text: The original unfavorable clause text to improve.
            context: Optional additional context about the contract or situation
                    (e.g., 'This is an employment contract', 'Small business vendor').

        Returns:
            A dictionary containing:
            - original: The original clause text
            - issue: What is wrong with the clause and why it's unfavorable
            - suggested_change: A rewritten better version of the clause
            - explanation: Why this change helps the user
        """
        if not self.llm:
            print("[WARN] RedlineAgent LLM not available — returning default response")
            return self._default_response(clause_text)

        if not clause_text or not clause_text.strip():
            print("[WARN] Empty clause text provided — returning default response")
            return self._default_response(clause_text)

        system_prompt = (
            "You are a contract negotiation expert. Your job is to analyze an unfavorable "
            "clause in a contract and suggest specific improvements that would be more "
            "fair and balanced. "
            "Return a JSON object with exactly these fields:\n"
            "1. original: The original clause text exactly as provided\n"
            "2. issue: What is wrong with this clause and why it's unfavorable "
            "(be specific about the problems)\n"
            "3. suggested_change: A rewritten version of the clause that addresses "
            "the issues while still being reasonable. Use professional legal language "
            "but make it more balanced.\n"
            "4. explanation: Why this change helps the user - explain the benefits "
            "of the suggested modification in plain English.\n"
            "Focus on practical improvements that protect the user's interests while "
            "still allowing the contract to be negotiated in good faith."
        )

        user_content = f"Clause to improve:\n{clause_text}"
        if context:
            user_content += f"\n\nAdditional context:\n{context}"

        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
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
                    "original": result.get("original", clause_text),
                    "issue": result.get("issue", "Unable to identify issue."),
                    "suggested_change": result.get("suggested_change", ""),
                    "explanation": result.get("explanation", "")
                }
            except json.JSONDecodeError:
                print("[WARN] RedlineAgent could not parse response as JSON")
            
            return self._default_response(clause_text)
            
        except Exception as e:
            print(f"[ERROR] RedlineAgent failed: {e}")
            return self._default_response(clause_text)

    def _default_response(self, clause_text: str) -> Dict[str, str]:
        """Return a default response when suggestion fails."""
        return {
            "original": clause_text,
            "issue": "Unable to analyze clause.",
            "suggested_change": "",
            "explanation": "Unable to provide suggestions at this time."
        }
