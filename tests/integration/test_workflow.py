import pytest
from unittest.mock import patch, MagicMock
from graph.workflow import build_graph

def test_workflow_compilation():
    """Happy path: ensures the graph compiles and creates a workflow successfully."""
    # Create workflow requires no external calls yet
    app = build_graph()
    
    # Just generating the compiled graph shouldn't raise exceptions
    assert app is not None

@patch("graph.nodes._router_agent")
@patch("graph.nodes._evaluate_agent")
@patch("graph.nodes.Pinecone")
@patch("graph.nodes.SentenceTransformer")
def test_workflow_execution_mocked(mock_embed, mock_pc, mock_eval, mock_router, mock_google_genai):
    """Happy path: Mock out external integrations and run graph for one step or check routing."""
    # This is a basic integration outline. A full integration test would mock specific
    # transitions in the state machine (LangGraph app).
    
    app = build_graph()
    
    # Can verify node dependencies and graph connectivity
    graph = app.get_graph()
    nodes = {node.name for node in graph.nodes.values()} if hasattr(graph.nodes, "values") else graph.nodes
    assert "route_query" in nodes or "router_node" in nodes
    assert "retrieve" in nodes
    assert "generate" in nodes
    assert "evaluate" in nodes or "evaluate_node" in nodes
