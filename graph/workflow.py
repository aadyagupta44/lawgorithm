"""
LangGraph workflow assembly and execution for Lawgorithm.

This module builds and compiles the StateGraph that implements the
simplified legal document RAG workflow. It exposes `build_graph()`
to assemble the complete workflow and `run_graph()` as a convenience
function to execute the graph for a single question with chat history.
"""

from typing import Any, Dict, List
from langgraph.graph import StateGraph, END, START
from graph.state import GraphState

from graph.nodes import (
    router_node,
    retrieve,
    generate,
    evaluate_node,
    handle_out_of_scope,
    update_memory,
)

from graph.edges import (
    route_after_router,
    decide_after_evaluate,
)


def build_graph() -> Any:
    """Assemble and compile the LangGraph StateGraph for legal document workflow."""

    print("[INFO] Building the legal document RAG StateGraph (Simplified Workflow)")

    g: StateGraph = StateGraph(GraphState)

    # Add all workflow node functions to the graph
    g.add_node("router_node", router_node)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("evaluate_node", evaluate_node)
    g.add_node("handle_out_of_scope", handle_out_of_scope)
    g.add_node("update_memory", update_memory)

    # Set the entry point to the router_node
    g.add_edge(START, "router_node")

    # Routing after router_node
    g.add_conditional_edges(
        "router_node",
        route_after_router,
        {
            "retrieve": "retrieve",
            "handle_out_of_scope": "handle_out_of_scope"
        }
    )

    # Retrieve transitions directly to generate
    g.add_edge("retrieve", "generate")

    # Generate transitions to evaluate
    g.add_edge("generate", "evaluate_node")

    # Routing after evaluation check
    g.add_conditional_edges(
        "evaluate_node",
        decide_after_evaluate,
        {
            "generate": "generate",
            "update_memory": "update_memory"
        }
    )

    # End nodes
    g.add_edge("handle_out_of_scope", END)
    g.add_edge("update_memory", END)

    compiled: Any = g.compile()
    print("[INFO] Graph compiled successfully")
    return compiled


def run_graph(
    question: str,
    namespace: str,
    chat_history: List[Dict[str, str]] = None,
    confidence_threshold: float = 0.7,
) -> Dict[str, Any]:
    """Build and execute the graph for a single question.

    Parameters:
        question: User query
        namespace: Pinecone namespace to search within
        chat_history: Conversation history
        confidence_threshold: Confidence score needed to accept answer
    """
    if chat_history is None:
        chat_history = []

    graph: Any = build_graph()

    initial_state: GraphState = {
        "question": question,
        "chat_history": chat_history,
        "documents": [],
        "generation": "",
        "loop_count": 0,
        "confidence_score": 1.0,
        "confidence_threshold": confidence_threshold,
        "hallucination_score": 1.0,
        "relevance_score": 1.0,
        "evaluation_reasoning": "",
        "namespace": namespace,
        "source_files": [],
        "source_pages": [],
        "correction_log": [],
        "previous_failed_answer": "",
        "previous_failure_reason": "",
        "is_relevant": "",
        "enhanced_query": "",
    }

    print(f"[INFO] Running graph for: '{question[:80]}' in namespace '{namespace}'")

    try:
        final_state: Dict[str, Any] = graph.invoke(initial_state)
        print("[INFO] Graph execution completed successfully")
        return final_state

    except Exception as e:
        print(f"[ERROR] Graph execution failed: {e}")
        if "correction_log" not in initial_state:
            initial_state["correction_log"] = []
        initial_state["correction_log"].append(f"Execution error: {str(e)}")
        return initial_state
