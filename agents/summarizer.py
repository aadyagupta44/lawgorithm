"""
Contract summarizer agent.

This module implements `ContractSummarizerAgent` which uses the Google Gemini LLM
to provide a structured summary of legal contracts. It identifies key information
like contract type, parties, obligations, dates, and risks.
"""

from typing import Dict, List
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, GROQ_MODEL


class ContractSummarizerAgent:
    """Agent that generates structured summaries of legal contracts."""

    def __init__(self) -> None:
        """Initialize ChatGroq client for summarization tasks."""
        try:
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL
            )
            print(f"[INFO] ContractSummarizerAgent initialized with model {GROQ_MODEL}")
        except Exception as e:
            self.llm = None  # type: ignore[assignment]
            print(f"[ERROR] Failed to initialize ContractSummarizerAgent LLM: {e}")

    def summarize(self, document_text: str) -> Dict[str, any]:
        """Generate a structured summary of the given legal document.

        Parameters:
            document_text: The full text of a legal contract or agreement.

        Returns:
            A dictionary containing:
            - contract_type: What kind of contract this is (e.g., employment, NDA, lease)
            - parties: Who are the parties involved (list of party names/roles)
            - main_obligations: List of main obligations from the contract
            - key_dates: Any important dates or deadlines found
            - key_risks: Top 3 risks identified in the contract
            - overall_summary: 2-3 sentence plain English summary of the contract
        """
        if not self.llm:
            print("[WARN] ContractSummarizerAgent LLM not available — returning empty dict")
            return {
                "contract_type": "",
                "parties": [],
                "main_obligations": [],
                "key_dates": [],
                "key_risks": [],
                "overall_summary": ""
            }

        if not document_text or not document_text.strip():
            print("[WARN] Empty document text provided — returning empty summary")
            return {
                "contract_type": "",
                "parties": [],
                "main_obligations": [],
                "key_dates": [],
                "key_risks": [],
                "overall_summary": ""
            }

        system_prompt = (
            "You are a legal contract analyzer that creates structured summaries. "
            "Analyze the provided contract and return a JSON object with exactly "
            "these fields:\n"
            "1. contract_type: What kind of contract this is (e.g., 'Employment Agreement', "
            "'Non-Disclosure Agreement', 'Service Agreement', 'Lease', 'Software License', etc.)\n"
            "2. parties: A list of all parties involved in the contract (by name or role, "
            "e.g., ['Employer', 'Employee'] or ['ABC Corp', 'XYZ Inc'])\n"
            "3. main_obligations: A list of the main obligations each party has under this contract\n"
            "4. key_dates: Any important dates or deadlines mentioned (payment due dates, "
            "start/end dates, renewal dates, notice periods, etc.)\n"
            "5. key_risks: The top 3 most significant risks in this contract (one-liners)\n"
            "6. overall_summary: A 2-3 sentence plain English summary of what this contract "
            "does and its key implications\n"
            "Be thorough but concise. Use null or empty arrays/strings for fields "
            "where no information is found."
        )

        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize this legal contract:\n\n{document_text}"},
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
                # Ensure all expected keys exist
                return {
                    "contract_type": result.get("contract_type", ""),
                    "parties": result.get("parties", []),
                    "main_obligations": result.get("main_obligations", []),
                    "key_dates": result.get("key_dates", []),
                    "key_risks": result.get("key_risks", []),
                    "overall_summary": result.get("overall_summary", "")
                }
            except json.JSONDecodeError:
                print("[WARN] ContractSummarizerAgent could not parse response as JSON")
            
            # Return empty dict if parsing failed
            return {
                "contract_type": "",
                "parties": [],
                "main_obligations": [],
                "key_dates": [],
                "key_risks": [],
                "overall_summary": ""
            }
            
        except Exception as e:
            print(f"[ERROR] ContractSummarizerAgent failed: {e}")
            return {
                "contract_type": "",
                "parties": [],
                "main_obligations": [],
                "key_dates": [],
                "key_risks": [],
                "overall_summary": ""
            }
