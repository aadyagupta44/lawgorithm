"""
Conditional edge functions for graph routing in Lawgorithm workflow.

Each function inspects the current GraphState and returns the name of the
next node to execute. These functions are used by the LangGraph StateGraph
to implement conditional transitions between nodes based on evaluation
results. Every routing decision is logged to the correction_log for
transparency and debugging.
"""

# Import END constant to signal workflow termination
from langgraph.graph import END

# Import GraphState for type-safe state access
from graph.state import GraphState

# Import graders for evaluation of generated answers
from agents import HallucinationGrader, AnswerGrader

# Import configuration constants
from config import MAX_LOOP_COUNT

# Initialize graders at module level to avoid repeated instantiation
try:
	# HallucinationGrader checks if answer is grounded in documents
	_hallucination_grader = HallucinationGrader()
	# AnswerGrader evaluates if answer is useful
	_answer_grader = AnswerGrader()
	# Log successful grader initialization
	print("[INFO] Graders initialized for edge routing")
except Exception as e:
	# Log error if grader initialization fails
	print(f"[ERROR] Failed to initialize graders: {e}")
	# Set to None to allow graceful degradation
	_hallucination_grader = None  # type: ignore[assignment]
	_answer_grader = None  # type: ignore[assignment]


def route_question(state: GraphState) -> str:
	"""Route question to retrieval or out-of-scope handler.

	This edge function determines whether the user's question pertains to
	legal documents. If relevant, it routes to the retrieve node. If out of
	scope, it routes to handle_out_of_scope node.

	Parameters:
		state: Current GraphState containing the user's question.

	Returns:
		'retrieve' if question is about legal documents, 'handle_out_of_scope' otherwise.
	"""

	# Import RouterAgent here to avoid circular imports with graph module
	from agents import RouterAgent
	
	# Create fresh RouterAgent instance for routing decision
	router = RouterAgent()
	
	# Extract question from state
	question: str = state.get("question", "")
	
	# Extract correction log to record routing decision
	correction_log: list = state.get("correction_log", [])
	
	try:
		# Use router agent to evaluate question relevance to legal documents
		decision: str = router.route(question)
		
		# Check if router determined question is relevant to legal documents
		if decision == "relevant":
			# Log routing decision to correction log
			correction_log.append("Router: question is about legal documents → retrieve")
			# Log to console
			print("[INFO] route_question: routing to retrieve")
			# Route to retrieve node for document search
			return "retrieve"
		else:
			# Log routing decision to correction log
			correction_log.append("Router: question is out of scope → handle_out_of_scope")
			# Log to console
			print("[INFO] route_question: routing to handle_out_of_scope")
			# Route to out-of-scope handler
			return "handle_out_of_scope"
	
	except Exception as e:
		# Log routing error
		print(f"[ERROR] route_question failed: {e}")
		# Add error to correction log
		correction_log.append(f"Router error: {str(e)} - defaulting to retrieve")
		# Default to retrieve for safety - better to try than reject
		return "retrieve"


def decide_after_grading(state: GraphState) -> str:
	"""Decide next step after document relevance grading.

	This edge function examines the retrieval_score to determine if documents
	are relevant. If relevant, proceed to generation. If irrelevant, rewrite
	the query for better retrieval.

	Parameters:
		state: Current GraphState containing retrieval_score and loop_count.

	Returns:
		'generate' if documents are relevant, 'rewrite_query' if not.
	"""

	# Extract retrieval score from grading
	retrieval_score: str = state.get("retrieval_score", "irrelevant")
	
	# Extract loop count to track self-correction attempts
	loop_count: int = int(state.get("loop_count", 0) or 0)
	
	# Extract correction log to record routing decision
	correction_log: list = state.get("correction_log", [])
	
	# Check if maximum retry attempts exceeded
	if loop_count >= MAX_LOOP_COUNT:
		# Log max attempts reached
		print(f"[WARN] Maximum loop count ({MAX_LOOP_COUNT}) reached - proceeding to generate")
		# Add to correction log
		correction_log.append(f"Max retries ({MAX_LOOP_COUNT}) reached - forcing generate")
		# Proceed to generation even with irrelevant documents
		return "generate"
	
	# Check if documents are relevant
	if retrieval_score == "relevant":
		# Log that documents are relevant
		print("[INFO] decide_after_grading: documents relevant - routing to generate")
		# Add to correction log
		correction_log.append("Grading result: relevant documents found → generate")
		# Route to generate node for answer creation
		return "generate"
	else:
		# Log that documents are irrelevant
		print("[WARN] decide_after_grading: documents irrelevant - routing to rewrite_query")
		# Add to correction log
		correction_log.append("Grading result: no relevant documents → rewrite query")
		# Route to rewrite_query node to improve search
		return "rewrite_query"


def decide_after_generation(state: GraphState) -> str:
	"""Decide next step after answer generation.

	This edge function evaluates the generated answer on two criteria:
	1. Hallucination: Is the answer grounded in the documents?
	2. Usefulness: Is the answer actually useful?

	Based on these scores, the answer is either accepted (update_memory),
	reworded (generate), or the query is rewritten (rewrite_query).

	Parameters:
		state: Current GraphState containing generation and grading scores.

	Returns:
		'update_memory' if answer is good, 'generate' if hallucinated,
		'rewrite_query' if not useful.
	"""

	# Extract generated answer to evaluate
	generation: str = state.get("generation", "")
	
	# Extract documents used as context for hallucination check
	documents: list = state.get("documents", [])
	
	# Extract loop count to track retry attempts
	loop_count: int = int(state.get("loop_count", 0) or 0)
	
	# Extract correction log to record routing decision
	correction_log: list = state.get("correction_log", [])
	
	# Default to good scores in case graders unavailable
	hallucination_score: str = "grounded"
	answer_score: str = "useful"
	
	# Check if graders are available for evaluation
	if _hallucination_grader and _answer_grader:
		try:
			# Grade if answer is hallucinated (not grounded in documents)
			hallucination_score: str = _hallucination_grader.grade(documents, generation)
			# Log hallucination check result
			print(f"[INFO] Hallucination check: {hallucination_score}")
			
		except Exception as e:
			# Log hallucination grading error
			print(f"[WARN] Hallucination grading failed: {e}")
			# Add to correction log
			correction_log.append(f"Hallucination check error: {str(e)}")
		
		try:
			# Grade if answer is useful and relevant
			answer_score: str = _answer_grader.grade(generation, documents)
			# Log answer quality check result
			print(f"[INFO] Answer quality check: {answer_score}")
			
		except Exception as e:
			# Log answer grading error
			print(f"[WARN] Answer grading failed: {e}")
			# Add to correction log
			correction_log.append(f"Answer quality check error: {str(e)}")
	
	else:
		# Log warning if graders not available
		print("[WARN] Graders not available - defaulting to good scores")
		# Add to correction log
		correction_log.append("Graders unavailable - accepting answer")
	
	# Update state with grading scores for next evaluation
	state["hallucination_score"] = hallucination_score
	state["answer_score"] = answer_score
	
	# Decision logic: Check for hallucination first (critical for legal documents)
	if hallucination_score == "hallucinated":
		# Answer contains information not in documents - problematic for legal content
		print("[WARN] decide_after_generation: hallucination detected - routing to generate")
		# Add to correction log
		correction_log.append("Answer hallucinated - regenerating without rewrite")
		# Route back to generate to try again without rewriting query
		return "generate"
	
	# Check if answer is useful
	if answer_score == "useful":
		# Answer is grounded and useful - accept it
		print("[INFO] decide_after_generation: answer useful - routing to update_memory")
		# Add to correction log
		correction_log.append("Answer accepted - updating memory")
		# Route to update_memory to save conversation
		return "update_memory"
	
	# If not useful and retry attempts remain, try new query
	if loop_count < MAX_LOOP_COUNT:
		# Answer not useful and we have retries left
		print("[WARN] decide_after_generation: answer not useful - routing to rewrite_query")
		# Add to correction log
		correction_log.append("Answer not useful - rewriting query for retry")
		# Route to rewrite_query to improve next attempt
		return "rewrite_query"
	
	else:
		# Out of retries - accept answer anyway
		print("[WARN] decide_after_generation: max retries reached - accepting answer")
		# Add to correction log
		correction_log.append("Max retries reached - accepting answer despite quality concerns")
		# Route to update_memory to save conversation and end
		return "update_memory"
