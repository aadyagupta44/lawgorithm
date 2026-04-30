"""
Graph node implementations for Lawgorithm legal document workflow.

This module implements the node functions that perform the key steps in the
simplified legal document RAG workflow:
1. Routing and HyDE (RouterAgent)
2. Retrieval (Pinecone)
3. Generation (LLM)
4. Evaluation (EvaluateAgent)
"""

from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from utils import get_llm

from config import (
    EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    TOP_K_RESULTS,
    MAX_CHAT_HISTORY,
)

from agents import RouterAgent, EvaluateAgent
from graph.state import GraphState


# ===========================
# Module-level initialization
# ===========================

# initialize embedding model for converting questions to vectors
try:
    _embedder = SentenceTransformer(EMBEDDING_MODEL)
    print(f"[INFO] Loaded embedding model: {EMBEDDING_MODEL}")
except Exception as e:
    _embedder = None  # type: ignore[assignment]
    print(f"[ERROR] Failed to load embedding model '{EMBEDDING_MODEL}': {e}")

# initialize Pinecone client and index using new style
try:
    _pc = Pinecone(api_key=PINECONE_API_KEY)
    _pine_index = _pc.Index(PINECONE_INDEX_NAME)
    print(f"[INFO] Connected to Pinecone index: {PINECONE_INDEX_NAME}")
except Exception as e:
    _pine_index = None
    print(f"[ERROR] Failed to initialize Pinecone index '{PINECONE_INDEX_NAME}': {e}")

# initialize agents
try:
    _router_agent = RouterAgent()
    _evaluate_agent = EvaluateAgent()
    print("[INFO] Router and Evaluate agents initialized successfully")
except Exception as e:
    print(f"[ERROR] Failed to initialize agents: {e}")
    _router_agent = None
    _evaluate_agent = None

# initialize LLM for answer generation via factory
try:
    _llm = get_llm()
    print("[INFO] LLM initialized by factory for answer generation")
except Exception as e:
    _llm = None  # type: ignore[assignment]
    print(f"[ERROR] Failed to initialize LLM via factory: {e}")


# ===========================
# Node functions
# ===========================

def router_node(state: GraphState) -> Dict[str, Any]:
    """Check relevance and enhance query with HyDE.

    Uses RouterAgent to determine if the query is legal. If yes, it also
    generates an enhanced query via the HyDE technique.
    """
    question: str = state.get("question", "")
    correction_log: List[str] = list(state.get("correction_log", []))
    
    correction_log.append("🧭 Routing question...")

    if not _router_agent:
        correction_log.append("⚠️ Router agent unavailable - assuming relevant")
        return {
            "is_relevant": "relevant",
            "enhanced_query": question,
            "correction_log": correction_log
        }
        
    try:
        decision, enhanced_query = _router_agent.route(question)
        if decision == "relevant":
            correction_log.append(f"✅ Question is relevant. Applied HyDE enhancement.")
        else:
            correction_log.append("❌ Question is out of scope.")
            
        return {
            "is_relevant": decision,
            "enhanced_query": enhanced_query,
            "correction_log": correction_log
        }
    except Exception as e:
        print(f"[ERROR] Router node failed: {e}")
        correction_log.append(f"⚠️ Router failed, assuming relevant: {str(e)}")
        return {
            "is_relevant": "relevant",
            "enhanced_query": question,
            "correction_log": correction_log
        }


def retrieve(state: GraphState) -> Dict[str, Any]:
    """Retrieve relevant legal document chunks from Pinecone.

    Embeds the enhanced question, searches Pinecone in the correct namespace,
    and returns matching document chunks with their metadata.
    """
    # use enhanced query if available, otherwise fallback to question
    query: str = state.get("enhanced_query", "") or state.get("question", "")
    chat_history: List[Dict[str, str]] = state.get("chat_history", [])
    correction_log: List[str] = list(state.get("correction_log", []))
    namespace: str = state.get("namespace", "")

    context_str = ""
    if chat_history:
        last_pair = chat_history[-1]
        context_str = f"Previous context: {last_pair.get('question', '')} -> {last_pair.get('answer', '')[:100]}"
    
    combined_query: str = f"{query}\n{context_str}" if context_str else query

    retrieved_documents: List[Dict[str, Any]] = []
    source_files: List[str] = []
    source_pages: List[int] = []

    correction_log.append("🔍 Searching documents...")

    if not _embedder or not _pine_index:
        correction_log.append("⚠️ Search dependencies unavailable")
        return {"documents": [], "source_files": [], "source_pages": [], "correction_log": correction_log}

    try:
        query_vector = _embedder.encode(combined_query).tolist()
        results = _pine_index.query(
            vector=query_vector,
            top_k=TOP_K_RESULTS,
            namespace=namespace,
            include_metadata=True,
        )

        for match in results.get("matches", []):
            metadata: Dict[str, Any] = match.get("metadata", {})
            content: str = metadata.get("content_preview") or metadata.get("content") or ""
            filename: str = metadata.get("filename", "unknown")
            page_number: int = int(metadata.get("page_number", 0))

            retrieved_documents.append({
                "content": content,
                "filename": filename,
                "page_number": page_number,
            })

            if filename not in source_files:
                source_files.append(filename)
            if page_number not in source_pages:
                source_pages.append(page_number)

        correction_log.append(f"✅ Retrieved {len(retrieved_documents)} chunks from {len(source_files)} file(s)")

    except Exception as e:
        print(f"[ERROR] Pinecone retrieval failed: {e}")
        correction_log.append(f"❌ Retrieval error: {str(e)}")

    return {
        "documents": retrieved_documents,
        "source_files": source_files,
        "source_pages": source_pages,
        "correction_log": correction_log,
    }


def generate(state: GraphState) -> Dict[str, Any]:
    """Generate a grounded legal answer using the configured LLM provider.

    Builds context from retrieved documents and conversation history.
    If loop_count > 0, it incorporates feedback from the evaluation node
    to correct hallucinations or relevance issues.
    """
    question: str = state.get("question", "")
    documents: List[Dict[str, Any]] = state.get("documents", [])
    chat_history: List[Dict[str, str]] = state.get("chat_history", [])
    correction_log: List[str] = list(state.get("correction_log", []))
    
    # Loop context
    loop_count: int = state.get("loop_count", 0)
    prev_answer: str = state.get("previous_failed_answer", "")
    failure_reason: str = state.get("previous_failure_reason", "")

    if loop_count > 0:
        correction_log.append(f"✍️ Regenerating answer (Attempt {loop_count + 1})...")
    else:
        correction_log.append("✍️ Generating answer from documents...")

    if not _llm:
        correction_log.append("❌ LLM unavailable")
        return {"generation": "System error: LLM unavailable.", "correction_log": correction_log}

    try:
        context_str = ""
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            filename = doc.get("filename", "unknown")
            page_number = doc.get("page_number", 0)
            context_str += f"\n--- Document {i} ({filename}, page {page_number}) ---\n{content}\n"

        history_context = ""
        if chat_history:
            recent = chat_history[-3:]
            if recent:
                history_context = "\nPrevious conversation:\n"
                for entry in recent:
                    history_context += f"Q: {entry.get('question', '')}\nA: {entry.get('answer', '')}\n"

        system_prompt = (
            "You are Lawgorithm, an expert legal advisor. Your role is to formulate legal information "
            "into clear, professional, and comprehensive insights.\n\n"
            "Guidelines:\n"
            "1. Base every answer STRICTLY on the provided documents. Never assume or invent facts.\n"
            "2. Provide crisp, yet appropriately detailed answers to fully address the user's inquiry.\n"
            "3. Clearly state actionable advice or key takeaways (rights, risks, obligations) when relevant.\n"
            "4. If the context does not contain the answer, explicitly state that it's not in the documents.\n"
            "5. Do NOT list out arbitrary citations in the main text unless specifically requested, as the system provides automatic source attachments.\n"
        )

        user_prompt = (
            f"{history_context}\n"
            f"Documents to analyze:\n{context_str}\n"
            f"Question: {question}\n\n"
        )
        
        # Inject feedback if we're in a retry loop
        if loop_count > 0 and prev_answer and failure_reason:
            user_prompt += (
                f"\n[ATTENTION] Your previous attempt was rejected for the following reason:\n"
                f"{failure_reason}\n\n"
                f"Previous rejected answer:\n{prev_answer}\n\n"
                f"Please correct the answer and ensure it perfectly addresses the feedback above.\n"
            )

        user_prompt += "Provide a comprehensive answer citing specific documents and pages."

        response = _llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        generation = str(response.content).strip()
        if not generation:
            generation = "Unable to generate a response. Please rephrase your question."

        correction_log.append(f"✅ Answer generated ({len(generation)} characters)")

    except Exception as e:
        print(f"[ERROR] LLM generation failed: {e}")
        generation = f"Error generating answer: {str(e)}"
        correction_log.append(f"❌ Generation error: {str(e)}")

    return {"generation": generation, "correction_log": correction_log}


def evaluate_node(state: GraphState) -> Dict[str, Any]:
    """Evaluate the generated answer for hallucination and relevance."""
    query: str = state.get("question", "")
    answer: str = state.get("generation", "")
    documents: List[Dict[str, Any]] = state.get("documents", [])
    correction_log: List[str] = list(state.get("correction_log", []))
    loop_count: int = state.get("loop_count", 0)
    
    correction_log.append("⚖️ Evaluating answer quality...")

    if not _evaluate_agent:
        correction_log.append("⚠️ Evaluate agent unavailable - skipping checks")
        return {
            "confidence_score": 1.0, 
            "hallucination_score": 1.0, 
            "relevance_score": 1.0, 
            "evaluation_reasoning": "Evaluator unavailable, accepting answer.",
            "correction_log": correction_log
        }
        
    try:
        confidence_score, details = _evaluate_agent.evaluate(query, answer, documents)
        hallucination_score = details.get("hallucination_score", 0.0)
        relevance_score = details.get("relevance_score", 0.0)
        reasoning = details.get("reasoning", "")
        
        correction_log.append(f"📊 Confidence Score: {confidence_score:.2f} (H: {hallucination_score:.2f}, R: {relevance_score:.2f})")
        
        updates = {
            "confidence_score": confidence_score,
            "hallucination_score": hallucination_score,
            "relevance_score": relevance_score,
            "evaluation_reasoning": reasoning,
            "correction_log": correction_log,
        }
        
        # If the score is low, record this failure so the next generation step can learn from it
        threshold = state.get("confidence_threshold", 0.7)
        if confidence_score < threshold:
            updates["previous_failed_answer"] = answer
            updates["previous_failure_reason"] = reasoning
            # We increment loop_count here since the cycle goes back to generate
            updates["loop_count"] = loop_count + 1
            
        return updates
        
    except Exception as e:
        print(f"[ERROR] Evaluator failed: {e}")
        correction_log.append(f"⚠️ Evaluator error: {str(e)}")
        return {"confidence_score": 1.0, "correction_log": correction_log}


def handle_out_of_scope(state: GraphState) -> Dict[str, Any]:
    """Handle questions that are not about legal documents."""
    correction_log: List[str] = list(state.get("correction_log", []))
    correction_log.append("⚠️ Question is out of scope for legal document analysis")

    generation: str = (
        "I'm specifically designed to help with legal document analysis. "
        "Your question doesn't appear to be about contracts, legal agreements, "
        "clauses, or other legal documents. Please upload a legal document "
        "and ask questions about its contents."
    )

    return {"generation": generation, "correction_log": correction_log}


def update_memory(state: GraphState) -> Dict[str, Any]:
    question: str = state.get("question", "")
    generation: str = state.get("generation", "")
    chat_history: List[Dict[str, str]] = list(state.get("chat_history", []))
    correction_log: List[str] = list(state.get("correction_log", []))

    chat_history.append({"question": question, "answer": generation})

    if len(chat_history) > MAX_CHAT_HISTORY:
        chat_history = chat_history[-MAX_CHAT_HISTORY:]

    correction_log.append(f"✅ Workflow completed — memory updated ({len(chat_history)} pairs)")

    return {
        "chat_history": chat_history,
        "correction_log": correction_log,
        # Explicitly carry scores forward so LangGraph doesn't drop them from final_state
        "confidence_score": state.get("confidence_score", 1.0),
        "hallucination_score": state.get("hallucination_score", 1.0),
        "relevance_score": state.get("relevance_score", 1.0),
        "evaluation_reasoning": state.get("evaluation_reasoning", ""),
        "source_files": state.get("source_files", []),
        "source_pages": state.get("source_pages", []),
        "loop_count": state.get("loop_count", 0),
    }