"""
Graph node implementations for Lawgorithm legal document workflow.

This module implements the node functions that perform the key steps in the
legal document RAG workflow. Each function accepts a `GraphState` dictionary
and returns an updates dictionary containing the state changes produced by
that node.

Agents and external clients (LLMs, Pinecone, embedding model) are initialized
once at module import time to avoid repeated expensive construction during
graph execution and to improve performance.
"""

# Import type annotations for function signatures and return types
from typing import Dict, Any, List

# Import SentenceTransformer for encoding user questions into embeddings
from sentence_transformers import SentenceTransformer

# Import Pinecone client using new style (from pinecone import Pinecone)
from pinecone import Pinecone

# Import LangChain ChatGroq for LLM-based answer generation
from langchain_groq import ChatGroq

# Import configuration constants from config module
from config import (
	EMBEDDING_MODEL,  # HuggingFace sentence-transformers model name
	PINECONE_API_KEY,  # Pinecone vector database API key from environment
	PINECONE_INDEX_NAME,  # Pinecone index name for legal documents
	GROQ_API_KEY,  # Groq API key for LLM from environment
	GROQ_MODEL,  # Groq model name for legal document tasks
	TOP_K_RESULTS,  # Number of document chunks to retrieve
	MAX_LOOP_COUNT,  # Maximum self-correction retry attempts
	MAX_CHAT_HISTORY,  # Maximum previous Q&A pairs to maintain
)

# Import all agent classes from agents package for specialized legal tasks
from agents import (
	RelevanceGrader,  # Judges if retrieved documents are relevant
	HallucinationGrader,  # Checks if answer is grounded in documents
	AnswerGrader,  # Evaluates answer quality and usefulness
	RiskFlagGrader,  # Identifies risky or concerning legal clauses
	QueryRewriter,  # Rewrites queries to improve retrieval
)

# Import GraphState TypedDict for type-safe state structure
from graph.state import GraphState


# ===========================
# Module-level initialization
# ===========================

# Initialize SentenceTransformer for embedding user queries into vectors
try:
	# Create SentenceTransformer instance with model from config
	_embedder = SentenceTransformer(EMBEDDING_MODEL)
	# Log successful initialization with model name
	print(f"[INFO] Loaded embedding model: {EMBEDDING_MODEL}")
except Exception as e:
	# Set embedder to None if initialization fails to allow graceful degradation
	_embedder = None  # type: ignore[assignment]
	# Log error details for debugging
	print(f"[ERROR] Failed to load embedding model '{EMBEDDING_MODEL}': {e}")

# Initialize Pinecone vector database client using new style
try:
	# Create Pinecone client with API key from environment
	_pc = Pinecone(api_key=PINECONE_API_KEY)
	# Get reference to the legal documents index
	_pine_index = _pc.Index(PINECONE_INDEX_NAME)
	# Log successful connection to Pinecone
	print(f"[INFO] Connected to Pinecone index: {PINECONE_INDEX_NAME}")
except Exception as e:
	# Set index to None if initialization fails
	_pine_index = None
	# Log error details for debugging
	print(f"[ERROR] Failed to initialize Pinecone index '{PINECONE_INDEX_NAME}': {e}")

# Initialize all grader agents used for quality evaluation
try:
	# RelevanceGrader judges if documents answer the question
	_relevance_grader = RelevanceGrader()
	# HallucinationGrader checks if answer is grounded in documents
	_hallucination_grader = HallucinationGrader()
	# AnswerGrader evaluates if answer is useful
	_answer_grader = AnswerGrader()
	# RiskFlagGrader identifies concerning legal clauses
	_risk_flag_grader = RiskFlagGrader()
	# Log successful initialization of graders
	print("[INFO] All grader agents initialized successfully")
except Exception as e:
	# Log if grader initialization fails
	print(f"[ERROR] Failed to initialize grader agents: {e}")

# Initialize QueryRewriter for self-correction query optimization
try:
	# QueryRewriter rewrites questions to improve retrieval quality
	_rewriter = QueryRewriter()
	# Log successful initialization
	print("[INFO] QueryRewriter initialized successfully")
except Exception as e:
	# Log if rewriter initialization fails
	print(f"[ERROR] Failed to initialize QueryRewriter: {e}")

# Initialize ChatGroq LLM for answer generation
try:
	# Create ChatGroq client with API key and model from config
	_llm = ChatGroq(
		api_key=GROQ_API_KEY,
		model=GROQ_MODEL
	)
	# Log successful LLM initialization
	print(f"[INFO] ChatGroq LLM initialized with model {GROQ_MODEL}")
except Exception as e:
	# Set LLM to None if initialization fails
	_llm = None  # type: ignore[assignment]
	# Log error details
	print(f"[ERROR] Failed to initialize ChatGroq LLM: {e}")


# ===========================
# Node functions
# ===========================

def retrieve(state: GraphState) -> Dict[str, Any]:
	"""Retrieve relevant legal documents from Pinecone vector database.

	This node embeds the user question, searches Pinecone for similar documents,
	and updates the state with retrieved chunks and their metadata.

	Parameters:
		state: Current GraphState containing question and chat history.

	Returns:
		Dictionary with keys: documents, source_files, source_pages, correction_log
	"""

	# Extract question from state (use as-is, no rephrasing needed here)
	question: str = state.get("question", "")
	
	# Extract chat history for conversational context if available
	chat_history: List[Dict[str, str]] = state.get("chat_history", [])
	
	# Extract existing correction log to append to it
	correction_log: List[str] = state.get("correction_log", [])
	
	# Extract Pinecone namespace to search within
	namespace: str = state.get("namespace", "")
	
	# Build context string from last chat history item if available for relevance
	context_str: str = ""
	if chat_history:
		# Use the most recent Q&A pair for context
		last_pair = chat_history[-1]
		context_str = f"Previous context: {last_pair.get('question', '')} → {last_pair.get('answer', '')}"
	
	# Combine question with context for improved retrieval
	combined_query: str = f"{question}\n{context_str}" if context_str else question
	
	# Initialize empty lists to collect retrieved documents and metadata
	retrieved_documents: List[Dict[str, Any]] = []
	source_files: List[str] = []
	source_pages: List[int] = []
	
	# Add retrieval step to correction log for transparency
	correction_log.append("Searching documents...")
	
	# Only proceed if embedder and Pinecone index are available
	if not _embedder or not _pine_index:
		# Log warning and return empty results
		print("[WARN] Embedder or Pinecone not available - returning empty documents")
		return {
			"documents": [],
			"source_files": [],
			"source_pages": [],
			"correction_log": correction_log,
		}
	
	try:
		# Embed the combined query (question + context) into a vector
		query_vector: List[float] = _embedder.encode(combined_query).tolist()
		
		# Search Pinecone for TOP_K_RESULTS most similar document chunks
		# Include metadata for source tracking
		results = _pine_index.query(
			vector=query_vector,
			top_k=TOP_K_RESULTS,
			namespace=namespace,
			include_metadata=True
		)
		
		# Process each matched document from Pinecone
		for match in results.get("matches", []):
			# Extract metadata dictionary from match
			metadata: Dict[str, Any] = match.get("metadata", {})
			
			# Build document dictionary with content and metadata
			document: Dict[str, Any] = {
				"content": metadata.get("content", ""),  # Text content of chunk
				"filename": metadata.get("filename", "unknown"),  # Source filename
				"page_number": metadata.get("page_number", 0),  # Source page number
			}
			
			# Add document to retrieved list
			retrieved_documents.append(document)
			
			# Track unique source files (avoid duplicates)
			if document["filename"] not in source_files:
				source_files.append(document["filename"])
			
			# Track source page numbers
			if document["page_number"] not in source_pages:
				source_pages.append(document["page_number"])
		
		# Log successful retrieval to correction log
		correction_log.append(f"Retrieved {len(retrieved_documents)} documents from {len(source_files)} files")
		
		# Log successful retrieval to console
		print(f"[INFO] Retrieved {len(retrieved_documents)} documents from Pinecone")
		
	except Exception as e:
		# Log retrieval error and continue with empty results
		print(f"[ERROR] Pinecone retrieval failed: {e}")
		# Add error to correction log for user visibility
		correction_log.append(f"Error retrieving documents: {str(e)}")
	
	# Return state updates for nodes to process
	return {
		"documents": retrieved_documents,
		"source_files": source_files,
		"source_pages": source_pages,
		"correction_log": correction_log,
	}


def grade_documents(state: GraphState) -> Dict[str, Any]:
	"""Grade retrieved documents for relevance and identify risk flags.

	This node evaluates each document to ensure relevance to the legal question,
	and also identifies any risky or concerning legal clauses.

	Parameters:
		state: Current GraphState containing documents and question.

	Returns:
		Dictionary with keys: documents, retrieval_score, risk_flags, correction_log
	"""

	# Extract question to evaluate documents against
	question: str = state.get("question", "")
	
	# Extract documents to grade
	documents: List[Dict[str, Any]] = state.get("documents", [])
	
	# Extract correction log to append to it
	correction_log: List[str] = state.get("correction_log", [])
	
	# Initialize lists to track filtering and risk identification
	relevant_documents: List[Dict[str, Any]] = []
	risk_flags: List[str] = []
	
	# Only proceed if graders are initialized
	if not _relevance_grader or not _answer_grader:
		# Log warning if graders unavailable
		print("[WARN] Graders not available - passing all documents")
		# Mark all documents as relevant if graders fail
		retrieval_score: str = "relevant" if documents else "irrelevant"
		# Add to correction log
		correction_log.append(f"Grading skipped - {len(documents)} documents passed")
		return {
			"documents": documents,
			"retrieval_score": retrieval_score,
			"risk_flags": risk_flags,
			"correction_log": correction_log,
		}
	
	try:
		# Grade each document individually for relevance
		for document in documents:
			try:
				# Grade document with RelevanceGrader
				relevance: str = _relevance_grader.grade(question, document)
				
				# Only keep relevant documents in the filtered list
				if relevance == "relevant":
					# Document is relevant, add to relevant list
					relevant_documents.append(document)
					
					# Also check for risk flags in relevant documents
					try:
						# Run RiskFlagGrader on relevant documents
						flags: List[str] = _risk_flag_grader.grade(document)
						
						# Add any identified flags to collection
						for flag in flags:
							if flag not in risk_flags:
								risk_flags.append(flag)
					
					except Exception as e:
						# Log risk grading error but don't block document inclusion
						print(f"[WARN] Risk grading failed: {e}")
				
			except Exception as e:
				# Log individual document grading error
				print(f"[WARN] Failed to grade document: {e}")
				# Include document anyway for safety (better to include uncertain documents)
				relevant_documents.append(document)
		
		# Determine overall retrieval score based on filtered results
		retrieval_score: str = "relevant" if relevant_documents else "irrelevant"
		
		# Add grading results to correction log
		grading_summary: str = f"Graded {len(documents)} docs, kept {len(relevant_documents)}"
		if risk_flags:
			grading_summary += f" - {len(risk_flags)} risk flags identified"
		correction_log.append(grading_summary)
		
		# Log grading results to console
		print(f"[INFO] Grading complete: {grading_summary}")
		
	except Exception as e:
		# Log overall grading error
		print(f"[ERROR] Document grading failed: {e}")
		# Default to relevant and include all documents for safety
		relevant_documents = documents
		retrieval_score = "relevant" if documents else "irrelevant"
		# Add error to correction log
		correction_log.append(f"Grading error: {str(e)}")
	
	# Return state updates with filtered documents and grading results
	return {
		"documents": relevant_documents,
		"retrieval_score": retrieval_score,
		"risk_flags": risk_flags,
		"correction_log": correction_log,
	}


def generate(state: GraphState) -> Dict[str, Any]:
	"""Generate answer to legal question using ChatGroq LLM.

	This node builds context from relevant documents, includes chat history
	for conversational memory, and generates a comprehensive legal answer
	with source citations.

	Parameters:
		state: Current GraphState containing documents, question, chat history.

	Returns:
		Dictionary with keys: generation, hallucination_score, correction_log
	"""

	# Extract user question to answer
	question: str = state.get("question", "")
	
	# Extract retrieved documents to use as context
	documents: List[Dict[str, Any]] = state.get("documents", [])
	
	# Extract chat history for conversational context
	chat_history: List[Dict[str, str]] = state.get("chat_history", [])
	
	# Extract correction log to append to
	correction_log: List[str] = state.get("correction_log", [])
	
	# Extract source information for citations
	source_files: List[str] = state.get("source_files", [])
	source_pages: List[int] = state.get("source_pages", [])
	
	# Add generation step to correction log
	correction_log.append("Generating answer...")
	
	# Check if LLM is available for generation
	if not _llm:
		# Log error and return placeholder if LLM unavailable
		print("[ERROR] LLM not available for generation")
		generation: str = "I'm unable to process your question at the moment due to system issues."
		# Add error to correction log
		correction_log.append("LLM unavailable")
		return {
			"generation": generation,
			"hallucination_score": "grounded",
			"correction_log": correction_log,
		}
	
	try:
		# Build context string from all retrieved documents
		context_str: str = ""
		for i, doc in enumerate(documents, 1):
			# Extract document content and metadata
			content: str = doc.get("content", "")
			filename: str = doc.get("filename", "unknown")
			page_number: int = doc.get("page_number", 0)
			
			# Format context entry with document reference
			context_str += f"\nDocument {i} ({filename}, page {page_number}):\n{content}\n"
		
		# Include last 3 chat history items for conversational memory
		history_context: str = ""
		if chat_history:
			# Get the last MAX_CHAT_HISTORY items (most recent conversations)
			recent_history: List[Dict[str, str]] = chat_history[-3:]
			
			if recent_history:
				# Format history for context
				history_context = "\nPrevious conversation:\n"
				for entry in recent_history:
					question_text: str = entry.get("question", "")
					answer_text: str = entry.get("answer", "")
					history_context += f"Q: {question_text}\nA: {answer_text}\n"
		
		# Build system prompt for legal document assistant
		system_prompt: str = (
			"You are Lawgorithm, an intelligent legal document assistant. "
			"Your role is to provide accurate, helpful legal information based on provided documents. "
			"\n\nGuidelines:\n"
			"1. Always answer based only on the provided documents.\n"
			"2. Cite specific documents and page numbers when referencing information.\n"
			"3. Use plain English explanations where possible, avoiding unnecessary jargon.\n"
			"4. Always mention which document and page number you're referencing.\n"
			"5. If information is not in the documents, clearly state that.\n"
			"6. Be professional and precise in your language.\n"
			"7. For contracts and legal documents, highlight key terms and obligations."
		)
		
		# Construct the full prompt with context and question
		user_prompt: str = (
			f"{history_context}\n"
			f"Documents to analyze:\n{context_str}\n"
			f"Question: {question}\n"
			f"Please provide a comprehensive answer based on the documents, "
			f"citing specific documents and page numbers."
		)
		
		# Call the LLM to generate answer
		response = _llm.invoke([
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		])
		
		# Extract and clean response text using .content.strip()
		generation: str = response.content.strip()
		
		# Check if response is empty and provide fallback
		if not generation:
			generation = "Unable to generate a response. Please try rephrasing your question."
		
		# Add successful generation to correction log
		correction_log.append(f"Generated answer ({len(generation)} characters)")
		
		# Log generation success
		print(f"[INFO] Answer generated ({len(generation)} characters)")
		
	except Exception as e:
		# Log generation error
		print(f"[ERROR] LLM generation failed: {e}")
		# Provide error message to user
		generation: str = f"I encountered an error while generating an answer: {str(e)}"
		# Add error to correction log
		correction_log.append(f"Generation error: {str(e)}")
	
	# Return state updates with generated answer
	return {
		"generation": generation,
		"hallucination_score": "grounded",  # Will be evaluated by HallucinationGrader
		"correction_log": correction_log,
	}


def rewrite_query(state: GraphState) -> Dict[str, Any]:
	"""Rewrite user query to improve document retrieval.

	This node uses QueryRewriter to optimize the question for better semantic
	search performance in the vector database. It increments loop counter for
	self-correction tracking.

	Parameters:
		state: Current GraphState containing question and loop count.

	Returns:
		Dictionary with keys: question, loop_count, correction_log
	"""

	# Extract original question
	original_question: str = state.get("question", "")
	
	# Extract current loop count for self-correction tracking
	loop_count: int = state.get("loop_count", 0)
	
	# Extract correction log to append to
	correction_log: List[str] = state.get("correction_log", [])
	
	# Increment loop counter for next retry
	loop_count += 1
	
	# Default to original question if rewriting fails
	rewritten_question: str = original_question
	
	# Check if rewriter is available
	if not _rewriter:
		# Log warning if rewriter unavailable
		print("[WARN] QueryRewriter not available - using original question")
		# Add to correction log
		correction_log.append(f"Rewrite attempt {loop_count} skipped - rewriter unavailable")
		return {
			"question": original_question,
			"loop_count": loop_count,
			"correction_log": correction_log,
		}
	
	try:
		# Use QueryRewriter to rewrite question for better retrieval
		rewritten_question: str = _rewriter.rewrite(original_question)
		
		# Ensure rewritten question is not empty
		if not rewritten_question:
			rewritten_question = original_question
		
		# Add rewrite step to correction log with before/after
		correction_log.append(
			f"Rewrite {loop_count}: '{original_question[:50]}...' → '{rewritten_question[:50]}...'"
		)
		
		# Log rewrite to console
		print(f"[INFO] Query rewritten: {original_question[:80]}... → {rewritten_question[:80]}...")
		
	except Exception as e:
		# Log rewriting error
		print(f"[ERROR] Query rewriting failed: {e}")
		# Add error to correction log
		correction_log.append(f"Rewrite attempt {loop_count} failed: {str(e)}")
	
	# Return state updates with rewritten question and incremented loop count
	return {
		"question": rewritten_question,
		"loop_count": loop_count,
		"correction_log": correction_log,
	}


def handle_out_of_scope(state: GraphState) -> Dict[str, Any]:
	"""Handle out-of-scope questions not about legal documents.

	This node generates a polite response for questions that don't pertain
	to legal document analysis, per the router's determination.

	Parameters:
		state: Current GraphState.

	Returns:
		Dictionary with keys: generation, correction_log
	"""

	# Extract correction log to append to
	correction_log: List[str] = state.get("correction_log", [])
	
	# Add out-of-scope handling to correction log
	correction_log.append("Question deemed out of scope")
	
	# Generate a polite message about scope limitations
	generation: str = (
		"I'm specifically designed to help with legal document analysis. "
		"Your question doesn't appear to be about contracts, legal agreements, "
		"clauses, or other legal documents. "
		"Please ask me about legal documents you'd like help understanding, "
		"and I'll provide comprehensive analysis with source citations."
	)
	
	# Log out-of-scope determination
	print("[INFO] Out-of-scope question handled")
	
	# Return state updates with polite out-of-scope response
	return {
		"generation": generation,
		"correction_log": correction_log,
	}


def update_memory(state: GraphState) -> Dict[str, Any]:
	"""Update chat history with current question and answer.

	This node maintains conversational memory by adding the current
	Q&A pair to the chat history, keeping only the most recent pairs
	for context in subsequent questions.

	Parameters:
		state: Current GraphState containing question, generation, chat_history.

	Returns:
		Dictionary with keys: chat_history, correction_log
	"""

	# Extract current question asked
	question: str = state.get("question", "")
	
	# Extract generated answer
	generation: str = state.get("generation", "")
	
	# Extract existing chat history
	chat_history: List[Dict[str, str]] = state.get("chat_history", [])
	
	# Extract correction log to append to
	correction_log: List[str] = state.get("correction_log", [])
	
	# Create new Q&A pair to add to history
	new_pair: Dict[str, str] = {
		"question": question,
		"answer": generation,
	}
	
	# Add new pair to chat history
	chat_history.append(new_pair)
	
	# Trim chat history to keep only MAX_CHAT_HISTORY most recent pairs
	if len(chat_history) > MAX_CHAT_HISTORY:
		# Remove oldest entries to maintain size limit
		chat_history = chat_history[-MAX_CHAT_HISTORY:]
	
	# Add memory update to correction log
	correction_log.append(f"Memory updated - {len(chat_history)} conversation pairs retained")
	
	# Log memory update to console
	print(f"[INFO] Chat history updated - {len(chat_history)} pairs in memory")
	
	# Return state updates with updated chat history
	return {
		"chat_history": chat_history,
		"correction_log": correction_log,
	}

