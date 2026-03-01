"""
Graph node implementations for CodeMind workflow.

This module implements the functions that act as nodes in the LangGraph
state graph. Each function accepts a `GraphState` dictionary and returns
an updates dictionary containing the state changes produced by that node.

Agents and external clients (LLMs, Pinecone, embedding model) are
initialized once at module import time to avoid repeated expensive
construction during graph execution.
"""

from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from config import (
	EMBEDDING_MODEL,
	PINECONE_API_KEY,
	PINECONE_INDEX_NAME,
	TOP_K_RESULTS,
)

from agents import RouterAgent, RelevanceGrader, HallucinationGrader, AnswerGrader, QueryRewriter
from graph.state import GraphState


# Initialize shared clients and agents at module load time
try:
	# embedding model for query encoding
	_embedder = SentenceTransformer(EMBEDDING_MODEL)
	print(f"[INFO] Loaded embedding model: {EMBEDDING_MODEL}")
except Exception as e:
	_embedder = None  # type: ignore[assignment]
	print(f"[ERROR] Failed to load embedding model '{EMBEDDING_MODEL}': {e}")

try:
    # initialize pinecone using new client style
    _pc = Pinecone(api_key=PINECONE_API_KEY)
    _pine_index = _pc.Index(PINECONE_INDEX_NAME)
    print(f"[INFO] Connected to Pinecone index: {PINECONE_INDEX_NAME}")
except Exception as e:
    _pine_index = None
    print(f"[ERROR] Failed to initialize Pinecone index '{PINECONE_INDEX_NAME}': {e}")
# initialize agents
_router = RouterAgent()
_relevance_grader = RelevanceGrader()
_hallucination_grader = HallucinationGrader()
_answer_grader = AnswerGrader()
_rewriter = QueryRewriter()


def retrieve(state: GraphState) -> Dict[str, Any]:
	"""Retrieve top-k document chunks from Pinecone for the given question.

	The node chooses `rephrased_question` if available, otherwise the
	original `question`. It embeds the query using the sentence-transformers
	model and queries Pinecone in the namespace provided by `state['namespace']`.

	Returns an updates dict containing `documents` and `source_files`.
	"""

	# choose which question to use for retrieval
	query_text = state.get("rephrased_question") or state.get("question", "")
	namespace = state.get("namespace", "")

	print(f"[INFO] retrieve: running vector search for namespace='{namespace}'")

	# defensive checks for required clients
	if _embedder is None or _pine_index is None:
		print("[ERROR] Retrieval dependencies missing (embedder or pinecone index)")
		return {"documents": [], "source_files": []}

	# compute embedding for the query
	try:
		q_vec = _embedder.encode([query_text])[0].tolist()  # type: ignore[attr-defined]
	except Exception as e:
		print(f"[ERROR] Failed to compute query embedding: {e}")
		return {"documents": [], "source_files": []}

	# query Pinecone for nearest neighbors
	try:
		resp = _pine_index.query(vector=q_vec, top_k=TOP_K_RESULTS, include_metadata=True, namespace=namespace)
	except Exception as e:
		print(f"[ERROR] Pinecone query failed: {e}")
		return {"documents": [], "source_files": []}

	# parse results into document dictionaries
	docs: List[Dict[str, Any]] = []
	source_files = []
	try:
		matches = getattr(resp, "matches", None) or resp.get("matches", [])
		for m in matches:
			metadata = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {})
			content = metadata.get("content") or metadata.get("text") or ""
			filename = metadata.get("filename", "")
			filepath = metadata.get("filepath", "")
			file_type = metadata.get("file_type", "")
			repo_name = metadata.get("repo_name", "")
			chunk_index = metadata.get("chunk_index", 0)

			doc = {
				"content": content,
				"filename": filename,
				"filepath": filepath,
				"file_type": file_type,
				"repo_name": repo_name,
				"chunk_index": chunk_index,
			}
			docs.append(doc)
			if filename and filename not in source_files:
				source_files.append(filename)
	except Exception as e:
		print(f"[ERROR] Failed to parse Pinecone results: {e}")

	print(f"[INFO] retrieve: returned {len(docs)} documents from Pinecone")
	return {"documents": docs, "source_files": source_files}


def grade_documents(state: GraphState) -> Dict[str, Any]:
	"""Grade retrieved documents for relevance and filter them.

	Uses `RelevanceGrader` to score each document and retains only
	documents judged 'relevant'. Updates `documents` and `retrieval_score`.
	"""

	retrieved = state.get("documents", []) or []
	print(f"[INFO] grade_documents: grading {len(retrieved)} documents")

	passed = []
	for doc in retrieved:
		try:
			score = _relevance_grader.grade(state.get("question", ""), doc)
			if score == "relevant":
				passed.append(doc)
		except Exception as e:
			print(f"[WARN] Relevance grading failed for a document: {e}")
			continue

	retrieval_score = "relevant" if passed else "irrelevant"
	print(f"[INFO] grade_documents: {len(passed)} documents passed grading; retrieval_score={retrieval_score}")
	return {"documents": passed, "retrieval_score": retrieval_score}


def generate(state: GraphState) -> Dict[str, Any]:
	"""Generate an answer from filtered documents and user question.

	Assembles a strict context from the provided documents and calls the
	Groq LLM to produce a grounded answer. The system prompt strictly
	instructs the model to use only the supplied context and not invent facts.
	"""

	documents = state.get("documents", []) or []
	question = state.get("question", "")

	# build context by concatenating document contents
	context = "\n\n".join([d.get("content", "") for d in documents])

	# system prompt enforcing grounding
	system_prompt = (
		"You are an assistant that must answer the user's question using ONLY the provided context. "
		"Do not invent or hallucinate facts. If the context does not contain the answer, say you cannot answer."
	)

	try:
		# call the Groq model via the router agent's llm instance
		# reuse router's LLM for generation if available; otherwise create a new ChatGroq
		from langchain_groq import ChatGroq
		from config import GROQ_API_KEY, GROQ_MODEL

		llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)
		payload = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
		]
		response = llm.invoke(payload)
		generation = response.content.strip()
		print("[INFO] generate: LLM produced a generation")
		return {"generation": generation}
	except Exception as e:
		print(f"[ERROR] Generation failed: {e}")
		return {"generation": ""}


def rewrite_query(state: GraphState) -> Dict[str, Any]:
	"""Rewrite the current question using the QueryRewriter and increment loop counter."""

	question = state.get("question", "")
	loop_count = int(state.get("loop_count", 0) or 0)
	try:
		rewritten = _rewriter.rewrite(question)
	except Exception as e:
		print(f"[ERROR] Query rewrite failed: {e}")
		rewritten = question

	loop_count += 1
	print(f"[INFO] rewrite_query: loop_count incremented to {loop_count}; rephrased_question='{rewritten}'")
	return {"rephrased_question": rewritten, "loop_count": loop_count}


def handle_out_of_scope(state: GraphState) -> Dict[str, Any]:
	"""Handle questions determined to be out of scope for the repository.

	Returns a polite message explaining the question is not related to the
	ingested codebase and sets `generation` accordingly.
	"""

	message = (
		"I'm sorry — that question appears unrelated to the provided codebase. "
		"I can only answer questions about the repository's code and documentation."
	)
	print("[INFO] handle_out_of_scope: returning polite out-of-scope message")
	return {"generation": message}

