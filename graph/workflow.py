"""
LangGraph workflow assembly and compilation.

This module builds and compiles the StateGraph that implements the
self-correcting RAG workflow. It exposes `build_graph()` to assemble
the graph and `run_graph()` as a convenience to execute the graph for
a single question/namespace pair.
"""

from typing import Any, Dict
from langgraph.graph import StateGraph, END, START

from graph.state import GraphState
from graph.nodes import (
    retrieve,
    grade_documents,
    generate,
    rewrite_query,
    handle_out_of_scope,
)
from graph.edges import route_question, decide_after_grading, decide_after_generation


def build_graph():
    """Assemble and compile the LangGraph StateGraph for the RAG workflow.

    Returns:
        A compiled StateGraph instance ready for execution.
    """

    print("[INFO] Building the RAG StateGraph")

    # create the state graph with our GraphState schema
    g = StateGraph(GraphState)

    # add all node functions to the graph
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("generate", generate)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("handle_out_of_scope", handle_out_of_scope)

    # set entry point with conditional routing from START
    g.add_conditional_edges(START, route_question)

    # retrieve always goes to grade_documents
    g.add_edge("retrieve", "grade_documents")

    # after grading decide whether to generate or rewrite
    g.add_conditional_edges("grade_documents", decide_after_grading)

    # after generation decide whether to accept or retry
    g.add_conditional_edges("generate", decide_after_generation)

    # rewrite always goes back to retrieve to try again
    g.add_edge("rewrite_query", "retrieve")

    # out of scope always ends
    g.add_edge("handle_out_of_scope", END)

    # compile the graph into an executable form
    compiled = g.compile()
    print("[INFO] Graph compiled successfully")
    return compiled


def run_graph(question: str, namespace: str) -> Dict[str, Any]:
    """Build and run the graph for a single question and namespace.

    Parameters:
        question: The user question to process.
        namespace: The Pinecone namespace to search in.

    Returns:
        Final state dictionary after graph execution.
    """

    # build and compile the graph
    graph = build_graph()

    # set up the initial state with all required fields
    initial_state: GraphState = {
        "question": question,
        "rephrased_question": "",
        "documents": [],
        "generation": "",
        "loop_count": 0,
        "retrieval_score": "irrelevant",
        "hallucination_score": "grounded",
        "answer_score": "useful",
        "namespace": namespace,
        "source_files": [],
    }

    print(f"[INFO] Running graph for: '{question[:80]}' in namespace '{namespace}'")

    try:
        # invoke the compiled graph with initial state
        final_state = graph.invoke(initial_state)
    except Exception as e:
        print(f"[ERROR] Graph execution failed: {e}")
        initial_state["generation"] = f"[ERROR] Graph execution failed: {e}"
        return initial_state

    return final_state
