"""
Grader agents for evaluating retrieval, hallucination, and answer quality.

This module implements three grader classes that each use the Groq LLM to
make binary judgments. Each grader initializes a `ChatGroq` client and
exposes a `grade` method that returns a deterministic string result.
"""

from typing import List, Dict
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, GROQ_MODEL


class RelevanceGrader:
	"""Judge whether a single document chunk is relevant to a question.

	The grader returns the string 'relevant' if the document chunk contains
	information directly useful to answer the question, otherwise 'irrelevant'.
	"""

	def __init__(self) -> None:
		"""Initialize the underlying LLM client for relevance grading."""
		try:
			# create a ChatGroq client with credentials from config
			self.llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)
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
		"""

		default = "irrelevant"  # conservative default
		if not self.llm:
			# if LLM is unavailable, default to 'irrelevant'
			print("[WARN] Relevance LLM not available — defaulting to 'irrelevant'")
			return default

		# instruct the model to produce a single-word judgment
		system_prompt = (
			"You are a strict judge. Given a user question and a single document chunk, "
			"respond only with the single word 'relevant' if the chunk contains information "
			"directly useful to answer the question, otherwise respond with 'irrelevant'."
		)

		try:
			content = document.get("content", "")
			# call the LLM with a small chat payload
			response = self.llm.invoke([
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": f"Question: {question}\n\nDocument:\n{content}"},
			])
			# normalize and interpret response text
			text = response.content.strip().lower()
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
	supported by the documents, otherwise 'hallucinated'.
	"""

	def __init__(self) -> None:
		"""Initialize the hallucination grader's LLM client."""
		try:
			self.llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)
			print(f"[INFO] HallucinationGrader initialized with model {GROQ_MODEL}")
		except Exception as e:
			self.llm = None  # type: ignore[assignment]
			print(f"[ERROR] Failed to initialize HallucinationGrader LLM: {e}")

	def grade(self, documents: List[Dict[str, str]], generation: str) -> str:
		"""Return 'grounded' if generation is fully supported by documents, else 'hallucinated'.

		Parameters:
			documents: List of document chunk dictionaries with 'content'.
			generation: The generated answer text to verify.
		"""

		default = "grounded"  # optimistic default to avoid unnecessary loops
		if not self.llm:
			print("[WARN] Hallucination LLM not available — defaulting to 'grounded'")
			return default

		try:
			# create a consolidated context for the LLM to check against
			context = "\n\n".join([d.get("content", "") for d in documents])
			system_prompt = (
				"You are an evaluator that checks whether each factual claim in the generated answer "
				"is directly supported by the provided documents. Respond only with 'grounded' if every "
				"claim can be traced to the documents, otherwise respond with 'hallucinated'."
			)

			response = self.llm.invoke([
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": f"Documents:\n{context}\n\nGeneration:\n{generation}"},
			])
			text = str(response).strip().lower()
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
			self.llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)
			print(f"[INFO] AnswerGrader initialized with model {GROQ_MODEL}")
		except Exception as e:
			self.llm = None  # type: ignore[assignment]
			print(f"[ERROR] Failed to initialize AnswerGrader LLM: {e}")

	def grade(self, question: str, generation: str) -> str:
		"""Return 'useful' if the generation answers the question, else 'not useful'.

		Parameters:
			question: The original user question.
			generation: The generated answer to evaluate.
		"""

		default = "useful"
		if not self.llm:
			print("[WARN] Answer LLM not available — defaulting to 'useful'")
			return default

		system_prompt = (
			"You are an answer quality evaluator. Given a user question and a generated answer, "
			"respond only with 'useful' if the answer addresses the question clearly and helpfully, "
			"otherwise respond with 'not useful'."
		)

		try:
			response = self.llm.invoke([
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": f"Question: {question}\n\nAnswer:\n{generation}"},
			])
			text = str(response).strip().lower()
			if "not useful" in text:
				return "not useful"
			return "useful"
		except Exception as e:
			print(f"[ERROR] AnswerGrader failed: {e}")
			return default

