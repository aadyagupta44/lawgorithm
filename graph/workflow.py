"""
LangGraph workflow assembly and execution for Lawgorithm.

This module builds and compiles the StateGraph that implements the
self-correcting legal document RAG workflow. It exposes `build_graph()`
to assemble the complete workflow and `run_graph()` as a convenience
function to execute the graph for a single question with chat history.
"""

# Import type annotations for function signatures
from typing import Any, Dict, List

# Import StateGraph and flow control constants from langgraph
from langgraph.graph import StateGraph, END, START

# Import GraphState TypedDict for type-safe state structure
from graph.state import GraphState

# Import all node functions that form the workflow
from graph.nodes import (
	retrieve,  # Retrieves documents from Pinecone
	grade_documents,  # Grades documents for relevance and risk
	generate,  # Generates answer with ChatGroq
	rewrite_query,  # Rewrites query for better retrieval
	handle_out_of_scope,  # Handles non-legal questions
	update_memory,  # Updates chat history with Q&A pair
)

# Import all edge routing functions
from graph.edges import (
	route_question,  # Routes to retrieve or handle_out_of_scope
	decide_after_grading,  # Routes to generate or rewrite_query
	decide_after_generation,  # Routes to update_memory, generate, or rewrite_query
)


def build_graph() -> Any:
	"""Assemble and compile the LangGraph StateGraph for legal document workflow.

	This function constructs the complete directed graph with all nodes and edges,
	implementing the self-correcting RAG workflow. The graph is then compiled
	into an executable form that can be invoked with an initial state.

	Returns:
		A compiled StateGraph instance ready for execution with invoke() method.
	"""

	# Log workflow construction start
	print("[INFO] Building the legal document RAG StateGraph")

	# Create the state graph using GraphState as the schema
	# This ensures type-safe state management throughout execution
	g: StateGraph = StateGraph(GraphState)

	# Add all workflow node functions to the graph
	# Each node performs a specific task in the legal document processing pipeline

	# retrieve: Embeds query, searches Pinecone, returns document chunks
	g.add_node("retrieve", retrieve)

	# grade_documents: Evaluates documents for relevance and risk flags
	g.add_node("grade_documents", grade_documents)

	# generate: Creates answer using ChatGroq LLM with document context
	g.add_node("generate", generate)

	# rewrite_query: Rewrites question to improve retrieval quality
	g.add_node("rewrite_query", rewrite_query)

	# handle_out_of_scope: Provides polite response for non-legal questions
	g.add_node("handle_out_of_scope", handle_out_of_scope)

	# update_memory: Saves current Q&A pair to chat history
	g.add_node("update_memory", update_memory)

	# Set the entry point with conditional routing from START
	# route_question determines if question is about legal documents
	# Returns: 'retrieve' or 'handle_out_of_scope'
	g.add_conditional_edges(START, route_question)

	# retrieve always transitions to grade_documents for relevance evaluation
	g.add_edge("retrieve", "grade_documents")

	# decide_after_grading routes based on document relevance
	# Returns: 'generate' if documents relevant, 'rewrite_query' if not
	g.add_conditional_edges("grade_documents", decide_after_grading)

	# decide_after_generation routes based on answer quality
	# Returns: 'update_memory' (accept), 'generate' (hallucinated), 'rewrite_query' (not useful)
	g.add_conditional_edges("generate", decide_after_generation)

	# rewrite_query always goes back to retrieve to search again with new query
	g.add_edge("rewrite_query", "retrieve")

	# handle_out_of_scope ends the workflow (no further processing needed)
	g.add_edge("handle_out_of_scope", END)

	# update_memory ends the workflow (conversation saved, answer delivered)
	g.add_edge("update_memory", END)

	# Compile the graph into an executable form
	# This creates an optimized representation ready for invocation
	compiled: Any = g.compile()

	# Log successful graph compilation
	print("[INFO] Graph compiled successfully")

	# Return the compiled graph for execution
	return compiled


def run_graph(
	question: str,
	namespace: str,
	chat_history: List[Dict[str, str]] = None,
) -> Dict[str, Any]:
	"""Build and execute the graph for a single question with chat history.

	This function creates the complete workflow, initializes the state with
	the user's question and conversation history, invokes the graph, and
	returns the complete final state including all processing details.

	Parameters:
		question: The legal document question to process (str).
		namespace: The Pinecone namespace to search within (str).
		chat_history: Optional list of previous Q&A pairs for conversation context.
			Each item is a dict with 'question' and 'answer' keys.
			Defaults to empty list if not provided.

	Returns:
		Dictionary containing the complete final state after graph execution:
		- question: Original user question
		- chat_history: Updated conversation history
		- documents: Retrieved and graded documents
		- generation: Final generated answer
		- loop_count: Number of self-correction attempts
		- retrieval_score: 'relevant' or 'irrelevant'
		- hallucination_score: 'grounded' or 'hallucinated'
		- answer_score: 'useful' or 'not useful'
		- namespace: Pinecone namespace used
		- source_files: List of source filenames
		- source_pages: List of source page numbers
		- risk_flags: Identified risk clauses
		- plain_english: Simplified explanation
		- correction_log: Step-by-step execution log
	"""

	# Default to empty list if chat_history not provided
	if chat_history is None:
		chat_history: List[Dict[str, str]] = []

	# Build and compile the complete workflow graph
	graph: Any = build_graph()

	# Initialize the state dictionary with all required fields
	# This represents the starting point for the workflow
	initial_state: GraphState = {
		# User's question about legal documents
		"question": question,
		# Chat history for conversational memory (previous Q&A pairs)
		"chat_history": chat_history,
		# Documents will be filled by retrieve node
		"documents": [],
		# Answer will be filled by generate node
		"generation": "",
		# Self-correction loop counter starts at 0
		"loop_count": 0,
		# Retrieval score will be set by grade_documents node
		"retrieval_score": "irrelevant",
		# Hallucination score will be set by generate and edges
		"hallucination_score": "grounded",
		# Answer quality score will be set by edges
		"answer_score": "useful",
		# Pinecone namespace for document search
		"namespace": namespace,
		# Source files will be collected by retrieve node
		"source_files": [],
		# Source page numbers will be collected by retrieve node
		"source_pages": [],
		# Risk flags will be identified by grade_documents node
		"risk_flags": [],
		# Plain English explanation (for future enhancement)
		"plain_english": "",
		# Correction log starts empty and is appended to by each node
		"correction_log": [],
	}

	# Log graph execution start
	print(f"[INFO] Running graph for: '{question[:80]}' in namespace '{namespace}'")

	try:
		# Invoke the compiled graph with the initial state
		# This runs the entire workflow to completion
		final_state: Dict[str, Any] = graph.invoke(initial_state)

		# Log successful execution
		print("[INFO] Graph execution completed successfully")

		# Return complete final state with all results and metadata
		return final_state

	except Exception as e:
		# Log execution error with details
		print(f"[ERROR] Graph execution failed: {e}")

		# Initialize correction log with error if not present
		if "correction_log" not in initial_state:
			initial_state["correction_log"] = []

		# Add error to correction log
		initial_state["correction_log"].append(f"Execution error: {str(e)}")

		# Return partial state with error information
		# This allows graceful handling of failures
		return initial_state
