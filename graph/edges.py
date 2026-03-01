"""
Conditional edge functions for graph routing.

Each function inspects the current GraphState and returns the name
of the next node to execute. These functions are used by the StateGraph
to implement conditional transitions between nodes.
"""

from langgraph.graph import END
from graph.state import GraphState
from agents import HallucinationGrader, AnswerGrader
from config import MAX_LOOP_COUNT

# initialize graders once at module level
_hallucination_grader = HallucinationGrader()
_answer_grader = AnswerGrader()


def route_question(state: GraphState) -> str:
    """Route question to retrieval or out of scope handler.

    Returns 'retrieve' or 'handle_out_of_scope'.
    """

    # import here to avoid circular imports
    from agents import RouterAgent
    router = RouterAgent()

    question = state.get("question", "")
    try:
        decision = router.route(question)
        if decision == "relevant":
            print("[INFO] route_question: routing to retrieve")
            return "retrieve"
        print("[INFO] route_question: routing to handle_out_of_scope")
        return "handle_out_of_scope"
    except Exception as e:
        print(f"[ERROR] route_question failed: {e}")
        return "retrieve"


def decide_after_grading(state: GraphState) -> str:
    """Decide next step after relevance grading.

    Returns 'generate' or 'rewrite_query'.
    """

    retrieval_score = state.get("retrieval_score", "irrelevant")
    loop_count = int(state.get("loop_count", 0) or 0)

    if retrieval_score == "relevant":
        print("[INFO] decide_after_grading: documents relevant, going to generate")
        return "generate"

    if loop_count >= MAX_LOOP_COUNT:
        print("[INFO] decide_after_grading: max loops reached, forcing generate")
        return "generate"

    print("[INFO] decide_after_grading: no relevant docs, rewriting query")
    return "rewrite_query"


def decide_after_generation(state: GraphState) -> str:
    """Decide next step after generation.

    Returns 'generate', 'rewrite_query', or END.
    """

    loop_count = int(state.get("loop_count", 0) or 0)
    documents = state.get("documents", []) or []
    generation = state.get("generation", "")
    question = state.get("question", "")

    # if we have hit max loops just end no matter what
    if loop_count >= MAX_LOOP_COUNT:
        print("[INFO] decide_after_generation: max loops reached, ending")
        return END

    # check for hallucination
    try:
        hallucination_result = _hallucination_grader.grade(documents, generation)
        print(f"[INFO] decide_after_generation: hallucination_result={hallucination_result}")
    except Exception as e:
        print(f"[ERROR] Hallucination grading failed: {e}")
        hallucination_result = "grounded"

    if hallucination_result == "hallucinated":
        print("[INFO] decide_after_generation: hallucinated, regenerating")
        return "generate"

    # check answer quality
    try:
        answer_result = _answer_grader.grade(question, generation)
        print(f"[INFO] decide_after_generation: answer_result={answer_result}")
    except Exception as e:
        print(f"[ERROR] Answer grading failed: {e}")
        answer_result = "useful"

    if answer_result == "useful":
        print("[INFO] decide_after_generation: answer useful, ending")
        return END

    print("[INFO] decide_after_generation: answer not useful, rewriting query")
    return "rewrite_query"
