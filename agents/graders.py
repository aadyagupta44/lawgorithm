"""
Grader agents for evaluating retrieval, hallucination, answer quality, and risk assessment.

This module implements four grader classes that each use the Google Gemini LLM to
make binary or ternary judgments for legal document analysis. Each grader initializes
a `ChatGoogleGenerativeAI` client and exposes a `grade` method that returns a 
deterministic string result.
"""

from typing import List, Dict
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, GROQ_MODEL


class RelevanceGrader:
    """Judge whether a single document chunk is relevant to a question.

    The grader returns the string 'relevant' if the document chunk contains
    information directly useful to answer the legal question, otherwise 'irrelevant'.
    """

    def __init__(self) -> None:
        """Initialize the underlying LLM client for relevance grading."""
        try:
            # create a ChatGroq client with credentials from config
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL
            )
            print(f"[INFO] RelevanceGrader initialized with model {GROQ_MODEL}")
        except Exception as e:
            # if initialization fails, store None and log the error
            self.llm = None  # type: ignore[assignment]
            print(f"[ERROR] Failed to initialize RelevanceGrader LLM: {e}")

    def grade(self, question: str, document: Dict[str, str]) -> str:
        """Return 'relevant' or 'irrelevant' for the given question and document.

        Parameters:
            question: The original user question string.
            document: A document chunk dictionary with a 'content' field.

        Returns:
            'relevant' if the document is relevant to the question, 'irrelevant' otherwise.
        """
        # conservative default for legal queries - better to include than exclude
        default = "relevant"
        if not self.llm:
            # if LLM is unavailable, default to 'relevant'
            print("[WARN] Relevance LLM not available — defaulting to 'relevant'")
            return default

        # instruct the model to produce a single-word judgment for legal content
        system_prompt = (
            "You are a strict judge for legal document relevance. Given a user question "
            "about legal matters and a single document chunk, respond only with the single word "
            "'relevant' if the chunk contains information directly useful to answer the legal "
            "question about contracts, clauses, terms, obligations, or legal rights. "
            "Otherwise respond with 'irrelevant'."
        )

        try:
            content = document.get("content", "")
            # call the LLM with a small chat payload
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}\n\nDocument:\n{content}"},
            ])
            # normalize and interpret response text
            text = str(response.content).strip().lower()
            if "relevant" in text:
                return "relevant"
            return "irrelevant"
        except Exception as e:
            # on error, log and return conservative default
            print(f"[ERROR] RelevanceGrader failed: {e}")
            return default


class HallucinationGrader:
    """Check if the generated answer is grounded in the provided documents.

    The grader returns 'grounded' if every factual claim in `generation` is
    supported by the documents, otherwise 'hallucinated'. This is especially 
    strict for legal content where accuracy is critical.
    """

    def __init__(self) -> None:
        """Initialize the hallucination grader's LLM client."""
        try:
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL
            )
            print(f"[INFO] HallucinationGrader initialized with model {GROQ_MODEL}")
        except Exception as e:
            self.llm = None  # type: ignore[assignment]
            print(f"[ERROR] Failed to initialize HallucinationGrader LLM: {e}")

    def grade(self, documents: List[Dict[str, str]], generation: str) -> str:
        """Return 'grounded' if generation is fully supported by documents, else 'hallucinated'.

        Parameters:
            documents: List of document chunk dictionaries with 'content'.
            generation: The generated answer text to verify.

        Returns:
            'grounded' if the generation is supported by documents, 'hallucinated' otherwise.
        """
        # optimistic default to avoid unnecessary loops - but verify carefully
        default = "grounded"
        if not self.llm:
            print("[WARN] Hallucination LLM not available — defaulting to 'grounded'")
            return default

        try:
            # create a consolidated context for the LLM to check against
            context = "\n\n".join([d.get("content", "") for d in documents])
            # extra strict system prompt for legal content
            system_prompt = (
                "You are an evaluator that checks whether each factual claim in the generated answer "
                "is directly supported by the provided legal documents. Legal accuracy is critical - "
                "any claim not explicitly stated in the documents is considered hallucinated. "
                "Respond only with 'grounded' if every claim can be traced to the documents, "
                "otherwise respond with 'hallucinated'."
            )

            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Documents:\n{context}\n\nGeneration:\n{generation}"},
            ])
            text = str(response.content).strip().lower()
            if "hallucinat" in text:
                return "hallucinated"
            return "grounded"
        except Exception as e:
            print(f"[ERROR] HallucinationGrader failed: {e}")
            return default


class AnswerGrader:
    """Judge whether the generated answer adequately addresses the question."""

    def __init__(self) -> None:
        """Initialize the answer quality grader's LLM client."""
        try:
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL
            )
            print(f"[INFO] AnswerGrader initialized with model {GROQ_MODEL}")
        except Exception as e:
            self.llm = None  # type: ignore[assignment]
            print(f"[ERROR] Failed to initialize AnswerGrader LLM: {e}")

    def grade(self, question: str, generation: str) -> str:
        """Return 'useful' if the generation answers the question, else 'not useful'.

        Parameters:
            question: The original user question.
            generation: The generated answer to evaluate.

        Returns:
            'useful' if the answer addresses the question, 'not useful' otherwise.
        """
        default = "useful"
        if not self.llm:
            print("[WARN] Answer LLM not available — defaulting to 'useful'")
            return default

        system_prompt = (
            "You are an answer quality evaluator for legal document questions. "
            "Given a user question about legal matters and a generated answer, "
            "respond only with 'useful' if the answer addresses the question clearly and helpfully, "
            "providing accurate legal information. Otherwise respond with 'not useful'."
        )

        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}\n\nAnswer:\n{generation}"},
            ])
            text = str(response.content).strip().lower()
            if "not useful" in text:
                return "not useful"
            return "useful"
        except Exception as e:
            print(f"[ERROR] AnswerGrader failed: {e}")
            return default


class RiskFlagGrader:
    """Grade the risk level of a legal clause or chunk content.

    This grader identifies the risk level associated with specific legal clauses:
    - 'high_risk': Unlimited liability, no termination rights, forced arbitration,
      auto-renewal traps, IP assignment clauses
    - 'medium_risk': Clauses that need attention or negotiation
    - 'low_risk': Standard acceptable clauses
    """

    def __init__(self) -> None:
        """Initialize the risk flag grader's LLM client."""
        try:
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL
            )
            print(f"[INFO] RiskFlagGrader initialized with model {GROQ_MODEL}")
        except Exception as e:
            self.llm = None  # type: ignore[assignment]
            print(f"[ERROR] Failed to initialize RiskFlagGrader LLM: {e}")

    def grade(self, chunk_content: str) -> str:
        """Return the risk level for the given legal chunk content.

        Parameters:
            chunk_content: The text content of a legal clause or document chunk.

        Returns:
            'high_risk', 'medium_risk', or 'low_risk' based on the clause content.
        """
        # default to medium risk for safety - requires human review
        default = "medium_risk"
        if not self.llm:
            print("[WARN] RiskFlag LLM not available — defaulting to 'medium_risk'")
            return default

        system_prompt = (
            "You are a legal risk assessor. Analyze the given legal clause or document content "
            "and classify its risk level:\n"
            "- 'high_risk': Unlimited liability, unlimited damages, no termination rights, "
            "forced arbitration without opt-out, auto-renewal traps, automatic IP assignment "
            "to the opposing party, non-mutual termination, penalty clauses, excessive notice periods "
            "favoring one party only.\n"
            "- 'medium_risk': Clauses that may need negotiation or legal review, such as "
            "unlimited confidentiality periods, broad indemnification, non-compete with "
            "reasonable but significant restrictions, payment terms with penalties.\n"
            "- 'low_risk': Standard clauses that are typical and acceptable in most contracts, "
            "such as basic confidentiality, standard payment terms, mutual termination rights, "
            "clear jurisdiction clauses.\n"
            "Respond with only one word: 'high_risk', 'medium_risk', or 'low_risk'."
        )

        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Legal clause to assess:\n{chunk_content}"},
            ])
            text = str(response.content).strip().lower()
            if "high_risk" in text:
                return "high_risk"
            elif "medium_risk" in text:
                return "medium_risk"
            elif "low_risk" in text:
                return "low_risk"
            return default
        except Exception as e:
            print(f"[ERROR] RiskFlagGrader failed: {e}")
            return default
