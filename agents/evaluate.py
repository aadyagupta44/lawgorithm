"""
Evaluate agent for assessing answer quality, hallucination, and relevance.

This module implements `EvaluateAgent` which uses the LLM to judge whether
a generated answer is grounded in documents, relevant to the query, and useful.
It produces a confidence score (0.0 to 1.0) combining hallucination and relevance.
"""

from typing import Any, Dict, Tuple
from utils import get_llm


class EvaluateAgent:
    """Agent that evaluates answer quality and confidence.

    The agent assesses:
    1. Hallucination: Is the answer grounded in the documents?
    2. Relevance: Does the answer address the user query?
    3. Confidence: Combined score from both factors (0.0 to 1.0)
    """

    def __init__(self) -> None:
        """Initialize the language model client for evaluation."""
        try:
            self.llm = get_llm()
            print("[INFO] EvaluateAgent initialized with dynamic LLM provider")
        except Exception as e:
            self.llm = None  # type: ignore[assignment]
            print(f"[ERROR] Failed to initialize EvaluateAgent LLM: {e}")

    def evaluate(
        self, 
        query: str, 
        answer: str, 
        documents: list
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate answer for hallucination, relevance, and confidence.

        Parameters:
            query: The user's original question string.
            answer: The generated answer to evaluate.
            documents: List of retrieved document chunks used for context.

        Returns:
            Tuple of:
            - confidence_score: Float between 0.0 and 1.0 representing overall quality
            - details: Dict with keys:
                - hallucination_score: 0.0-1.0 (1.0 = fully grounded, 0.0 = hallucinated)
                - relevance_score: 0.0-1.0 (1.0 = perfectly relevant, 0.0 = irrelevant)
                - reasoning: Explanation of evaluation
        """
        
        if not self.llm:
            print("[WARN] Evaluator LLM not available — defaulting to 0.7 confidence")
            return 0.7, {
                "hallucination_score": 0.7,
                "relevance_score": 0.7,
                "reasoning": "LLM unavailable - conservative estimate"
            }

        # Build document content summary for grounding check
        doc_summary = "\n".join([
            f"[Doc {i+1}] {doc.get('content', '')[:200]}"
            for i, doc in enumerate(documents[:3])  # Use first 3 docs
        ])

        system_prompt = (
            "You are an expert legal document evaluator. Assess the provided answer for:\n"
            "1. GROUNDING: Is every claim grounded in the provided documents? Score from 0 (total hallucination) to 100 (fully grounded).\n"
            "2. RELEVANCE: Does the answer directly address the user's question? Score from 0 (irrelevant) to 100 (perfectly relevant).\n"
            "3. CONFIDENCE: Overall confidence score from 0 to 100 based on grounding and relevance.\n\n"
            "Format your response EXACTLY as:\n"
            "GROUNDING: 95\n"
            "RELEVANCE: 90\n"
            "CONFIDENCE: 92\n"
            "REASONING: [Brief explanation]"
        )

        user_message = (
            f"USER QUERY:\n{query}\n\n"
            f"DOCUMENTS PROVIDED:\n{doc_summary}\n\n"
            f"ANSWER TO EVALUATE:\n{answer}"
        )

        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ])
            
            text = str(response.content).strip()
            
            import re
            
            def extract_score(text_content, key):
                # Search for KEY: followed by a number between 0 and 100
                match = re.search(fr"{key}:\s*(\d+)", text_content, re.IGNORECASE)
                if match:
                    try:
                        # scale it to 0.0 to 1.0 for the internal state machine logic
                        return max(0.0, min(1.0, float(match.group(1)) / 100.0))
                    except Exception:
                        pass
                return None

            hallucination_score = extract_score(text, "GROUNDING")
            relevance_score = extract_score(text, "RELEVANCE")
            confidence_score = extract_score(text, "CONFIDENCE")
            
            # Fallbacks if parsing fails
            if hallucination_score is None: hallucination_score = 0.5
            if relevance_score is None: relevance_score = 0.5
            if confidence_score is None: confidence_score = (hallucination_score + relevance_score) / 2.0
            
            # Extract reasoning
            reasoning = "Evaluation complete."
            reasoning_match = re.search(r"REASONING:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            
            details = {
                "hallucination_score": hallucination_score,
                "relevance_score": relevance_score,
                "reasoning": reasoning,
                "raw_response": text
            }
            
            return confidence_score, details
            
        except Exception as e:
            print(f"[ERROR] EvaluateAgent evaluation failed: {e}")
            return 0.5, {
                "hallucination_score": 0.5,
                "relevance_score": 0.5,
                "reasoning": f"Evaluation error: {str(e)}"
            }
