import pytest
from unittest.mock import patch, MagicMock
from agents.router import RouterAgent
from agents.evaluate import EvaluateAgent

def test_router_agent_route_in_scope(mock_google_genai):
    """Happy path: Router correctly routes a legal query."""
    # Setup mock LLM response
    mock_msg1 = MagicMock()
    mock_msg1.content = 'relevant'
    mock_msg2 = MagicMock()
    mock_msg2.content = 'hypothetical legal clause'
    mock_google_genai.invoke.side_effect = [mock_msg1, mock_msg2]
    
    router = RouterAgent()
    router.llm = mock_google_genai
    decision, enhanced = router.route("Is this legal?")
    
    assert decision == "relevant"
    assert "hypothetical" in enhanced

def test_router_agent_route_out_of_scope(mock_google_genai):
    """Sad path: Router rejects an out-of-scope query."""
    mock_msg = MagicMock()
    mock_msg.content = 'irrelevant'
    mock_google_genai.invoke.return_value = mock_msg
    
    router = RouterAgent()
    router.llm = mock_google_genai
    decision, enhanced = router.route("How to bake a cake?")
    
    assert decision == "irrelevant"
    assert enhanced == ""

def test_evaluate_agent_relevance(mock_google_genai):
    """Happy path: test document relevance detection."""
    mock_msg = MagicMock()
    mock_msg.content = '''
GROUNDING: 95
RELEVANCE: 100
CONFIDENCE: 98
REASONING: The answer is solid.
'''
    mock_google_genai.invoke.return_value = mock_msg
    
    evaluator = EvaluateAgent()
    evaluator.llm = mock_google_genai
    confidence, details = evaluator.evaluate("Question", "Answer", [{"content": "Doc"}])
    
    assert confidence == 0.98
    assert details["hallucination_score"] == 0.95
    assert details["relevance_score"] == 1.0

def test_evaluate_agent_hallucination_empty_generation(mock_google_genai):
    """Sad path: evaluating hallucination with empty generation."""
    mock_msg = MagicMock()
    mock_msg.content = '''
GROUNDING: 0
RELEVANCE: 0
CONFIDENCE: 0
REASONING: No answer provided.
'''
    mock_google_genai.invoke.return_value = mock_msg

    evaluator = EvaluateAgent()
    evaluator.llm = mock_google_genai
    confidence, details = evaluator.evaluate("Question", "", [{"content": "Doc"}])
    assert confidence == 0.0
    assert details["hallucination_score"] == 0.0