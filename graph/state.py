"""
Graph state definition for Lawgorithm RAG workflow.

This module defines the `GraphState` TypedDict which contains all state fields
used by the LangGraph StateGraph during legal document processing. Every field
is strongly typed and documented so nodes and edges can update parts of the
state predictably and safely.
"""

# Import TypedDict for type-safe state dictionary structure
from typing import TypedDict, List, Dict, Any

# Import MAX_LOOP_COUNT and MAX_CHAT_HISTORY from configuration
from config import MAX_LOOP_COUNT, MAX_CHAT_HISTORY


class GraphState(TypedDict, total=False):
	"""TypedDict describing the mutable state passed through graph nodes.

	This schema defines all fields that flow through the LangGraph workflow
	for legal document processing. Each field is optional (total=False) and
	represents a specific piece of conversation or processing state.

	Fields:
		question: The original user question string.
		chat_history: List of previous Q&A pairs for conversational memory.
		documents: Retrieved document chunks with metadata.
		generation: The LLM-generated answer text.
		loop_count: Counter for self-correction retry attempts.
		retrieval_score: 'relevant' or 'irrelevant' judgment on documents.
		hallucination_score: 'grounded' or 'hallucinated' judgment on answer.
		answer_score: 'useful' or 'not useful' judgment on answer quality.
		namespace: Pinecone namespace string for vector database search.
		source_files: List of filenames from which documents were retrieved.
		source_pages: List of page numbers from source documents.
		risk_flags: Identified risk clauses or concerning legal terms.
		plain_english: Plain English explanation of answer.
		correction_log: Log entries recording each step during processing.
	"""

	# Original question asked by the user about legal documents
	question: str

	# Previous Q&A pairs for conversation memory - each dict has "question" and "answer" keys
	chat_history: List[Dict[str, str]]

	# List of retrieved document chunks, each with content and metadata
	documents: List[Dict[str, Any]]

	# The LLM-generated answer to the question
	generation: str

	# Counter tracking how many times the query has been rewritten (self-correction attempts)
	loop_count: int

	# Relevance judgment from RelevanceGrader: 'relevant' or 'irrelevant'
	retrieval_score: str

	# Hallucination judgment from HallucinationGrader: 'grounded' or 'hallucinated'
	hallucination_score: str

	# Answer quality judgment from AnswerGrader: 'useful' or 'not useful'
	answer_score: str

	# Pinecone namespace string for vector database search scope
	namespace: str

	# List of unique source filenames referenced in retrieved documents
	source_files: List[str]

	# List of page numbers from source documents (int values)
	source_pages: List[int]

	# List of risk flags/clauses identified by RiskFlagGrader
	risk_flags: List[str]

	# Plain English simplified explanation of the answer
	plain_english: str

	# Correction log tracking every step taken during workflow execution for display to user
	correction_log: List[str]

