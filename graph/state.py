"""
Graph state definition for CodeMind RAG workflow.

This module defines the `GraphState` TypedDict which contains all
state fields used by the LangGraph StateGraph during execution. Every
field is typed and documented so nodes and edges can update parts of
the state predictably.
"""

from typing import TypedDict, List, Dict, Any
from config import MAX_LOOP_COUNT


class GraphState(TypedDict, total=False):
	"""TypedDict describing the mutable state passed through graph nodes.

	Fields:
		question: The original user question.
		rephrased_question: Rewritten question for improved retrieval.
		documents: List of retrieved document chunk dictionaries.
		generation: The generated answer text from the LLM.
		loop_count: Number of self-correction attempts made so far.
		retrieval_score: 'relevant' or 'irrelevant' from relevance grader.
		hallucination_score: 'grounded' or 'hallucinated' from hallucination grader.
		answer_score: 'useful' or 'not useful' from answer grader.
		namespace: Pinecone namespace string corresponding to the repo.
		source_files: List of unique filenames referenced in generation.
	"""

	question: str  # original user question
	rephrased_question: str  # rewritten question when applicable
	documents: List[Dict[str, Any]]  # list of retrieved document chunk dicts
	generation: str  # generated answer from the LLM
	loop_count: int  # self-correction loop counter (max MAX_LOOP_COUNT)
	retrieval_score: str  # 'relevant' or 'irrelevant' as judged
	hallucination_score: str  # 'grounded' or 'hallucinated'
	answer_score: str  # 'useful' or 'not useful'
	namespace: str  # Pinecone namespace to search within
	source_files: List[str]  # unique filenames used as evidence

