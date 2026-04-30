"""
Conditional edge functions for graph routing in Lawgorithm workflow.

Each function inspects the current GraphState and returns the name of the
next node to execute. Every routing decision is logged to the correction_log
for transparency.
"""

from graph.state import GraphState


def route_after_router(state: GraphState) -> str:
    """Route question based on relevance classification.

    This edge function determines whether the user's question pertains to
    legal documents based on the `is_relevant` field set by router_node.

    Parameters:
        state: Current GraphState containing the is_relevant decision.

    Returns:
        'retrieve' if question is about legal documents, 'handle_out_of_scope' otherwise.
    """
    decision: str = state.get("is_relevant", "relevant")
    correction_log = state.get("correction_log", [])

    if decision == "relevant":
        print("[INFO] route_after_router: routing to retrieve")
        return "retrieve"
    else:
        print("[INFO] route_after_router: routing to handle_out_of_scope")
        return "handle_out_of_scope"


def decide_after_evaluate(state: GraphState) -> str:
    """Decide next step after answer evaluation.

    This edge function checks if the answer's confidence score meets the
    threshold. If it doesn't, and we haven't hit the loop limit, it routes
    back to generate for self-correction.

    Parameters:
        state: Current GraphState containing scores and threshold.

    Returns:
        'update_memory' if answer is good or max loops reached,
        'generate' if retry is needed.
    """
    confidence_score: float = state.get("confidence_score", 1.0)
    # Default threshold is 0.7 if not set
    confidence_threshold: float = state.get("confidence_threshold", 0.7)
    loop_count: int = state.get("loop_count", 0)
    correction_log = state.get("correction_log", [])

    # The user requested exactly 3 retries max
    # We interpret this as allowing up to 3 regeneration loops.
    # So if loop_count is 0, 1, or 2, we can retry.
    if confidence_score < confidence_threshold and loop_count < 3:
        print(f"[WARN] Score {confidence_score:.2f} < {confidence_threshold:.2f} - routing to generate (loop count: {loop_count})")
        correction_log.append(f"🔄 Score below threshold. Initiating retry (loop {loop_count + 1}/3)")
        return "generate"
    elif confidence_score < confidence_threshold and loop_count >= 3:
        print(f"[WARN] Score {confidence_score:.2f} < {confidence_threshold:.2f} but max loops reached - routing to update_memory")
        correction_log.append("⚠️ Max retries (3) reached. Accepting answer despite low score.")
        return "update_memory"
    else:
        print(f"[INFO] Score {confidence_score:.2f} >= {confidence_threshold:.2f} - routing to update_memory")
        correction_log.append("✅ Score meets threshold. Workflow complete.")
        return "update_memory"
