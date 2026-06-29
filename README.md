#  Lawgorithm — Intelligent Legal Document Assistant

> A self-correcting, agentic **Retrieval-Augmented Generation (RAG)** system for analyzing legal documents — built on **LangGraph**, **Pinecone**, and pluggable LLM providers (**Groq / Google Gemini**).

Lawgorithm lets you upload contracts, agreements, NDAs, leases, and other legal documents, then ask questions in natural language. Answers are grounded strictly in the source material, automatically scored for **hallucination** and **relevance**, and **re-generated** if quality falls below a configurable confidence threshold — all with full source attribution (filename + page number).

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [The Self-Correcting Workflow](#the-self-correcting-workflow)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Usage](#usage)
- [REST API Reference](#rest-api-reference)
- [Testing](#testing)
- [Continuous Integration](#continuous-integration)
- [Design Notes & Trade-offs](#design-notes--trade-offs)
- [Roadmap](#roadmap)

---

## Key Features

| Feature | Description |
|---|---|
|  **Query Routing** | An LLM-based router classifies each question as `relevant` (legal) or `irrelevant`, short-circuiting out-of-scope queries before any retrieval cost is incurred. |
|  **HyDE Query Enhancement** | Relevant queries are expanded with a **Hypothetical Document Embedding** — a synthetic legal clause that answers the question — to improve semantic retrieval recall. |
|  **Grounded Generation** | Answers are constrained strictly to retrieved document chunks. The model is instructed to explicitly state when an answer is *not* present in the source. |
|  **Automated Evaluation** | Every answer is scored by an evaluator agent on **grounding** (anti-hallucination) and **relevance**, producing a combined confidence score in `[0.0, 1.0]`. |
|  **Self-Correction Loop** | If confidence < threshold, the failure reason is fed back into a re-generation step. Up to **3 retry loops** before the best-effort answer is accepted. |
|  **Namespaced Multi-Document Store** | Each document is embedded into its own Pinecone **namespace**, keeping document corpora isolated and queryable independently. |
|  **Pluggable LLM Provider** | Switch between **Groq** (`llama-3.3-70b-versatile`) and **Google Gemini** (`gemini-2.0-flash-lite`) by changing a single config line. |
|  **Source Attribution** | Responses carry the list of source filenames and page numbers used to construct the answer. |
|  **Conversational Memory** | Sliding-window chat history (default 10 turns) provides follow-up context across a conversation. |
|  **Two Interfaces** | A rich **Streamlit** web UI and a **FastAPI** REST service with auto-generated Swagger docs. |
|  **Transparent Trace** | A live "correction log" records every step (routing → retrieval → generation → evaluation → retries) for full observability. |

---

## Architecture

Lawgorithm is composed of three cooperating layers: an **ingestion pipeline**, an **agentic LangGraph workflow**, and **presentation interfaces**.

```
                          ┌──────────────────────────────────────────────┐
                          │                INTERFACES                     │
                          │   Streamlit UI (app.py)   FastAPI (api.py)    │
                          └───────────────┬──────────────────┬───────────┘
                                          │                  │
              ┌───────────────────────────┘                  └──────────────────────────┐
              ▼                                                                           ▼
 ┌────────────────────────────┐                                       ┌─────────────────────────────────┐
 │     INGESTION PIPELINE      │                                       │      LANGGRAPH WORKFLOW           │
 │                             │                                       │        (graph/)                  │
 │  DocumentLoader  (PDF/text) │                                       │                                  │
 │        │                    │                                       │   router_node ──┬─► retrieve     │
 │        ▼                    │      embeddings + metadata            │       │         │      │         │
 │  DocumentChunker            │ ───────────────────────────────────► │  (out of scope) │      ▼         │
 │  (1000 chars / 200 overlap) │                                       │       ▼         │   generate     │
 │        │                    │                                       │  handle_out_of  │      │         │
 │        ▼                    │                                       │    _scope       │      ▼         │
 │  PineconeEmbedder           │ ◄──────── retrieval (top-k=5) ──────► │                 │  evaluate_node │
 │  (all-MiniLM-L6-v2, 384-d)  │                                       │                 │      │         │
 └──────────────┬──────────────┘                                      │       confidence│≥ threshold?    │
                ▼                                                      │           ┌─────┴─────┐         │
       ┌──────────────────┐                                           │       (no, <3 loops)  (yes)      │
       │  Pinecone Index  │                                           │           │           │         │
       │  (per-doc        │◄──────────────────────────────────────── │       generate    update_memory  │
       │   namespaces)    │                                           └─────────────────────────────────┘
       └──────────────────┘
```

### Components

- **Ingestion** (`ingestion/`)
  - `DocumentLoader` — extracts text page-by-page from PDFs via **PyMuPDF**, handling encrypted files gracefully; also wraps raw pasted text. (`GitHubLoader` is also present, a legacy of the project's origins as a code-RAG system.)
  - `DocumentChunker` — splits pages into overlapping chunks with LangChain's `RecursiveCharacterTextSplitter` (`CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200`).
  - `PineconeEmbedder` — encodes chunks with `sentence-transformers/all-MiniLM-L6-v2` (384-d) and upserts to Pinecone in batches of 100, preserving rich provenance metadata (filename, page, chunk index, document id, content preview).

- **Agents** (`agents/`)
  - `RouterAgent` — relevance classification + HyDE query enhancement.
  - `EvaluateAgent` — grounding/relevance/confidence scoring with regex-parsed structured output and safe fallbacks.

- **Graph** (`graph/`)
  - `state.py` — the strongly-typed `GraphState` `TypedDict` shared across all nodes.
  - `nodes.py` — node implementations (router, retrieve, generate, evaluate, out-of-scope, memory update).
  - `edges.py` — conditional routing functions (`route_after_router`, `decide_after_evaluate`).
  - `workflow.py` — assembles and compiles the `StateGraph`; exposes `build_graph()` and `run_graph()`.

- **Utilities** (`utils/`)
  - `llm_factory.py` — single source of truth (`get_llm()`) for instantiating the configured chat model, so every agent uses the same provider.

---

## The Self-Correcting Workflow

The heart of Lawgorithm is a LangGraph state machine. A single query flows through it as follows:

1. **`router_node`** — Classifies the question. If irrelevant → `handle_out_of_scope` → `END`. If relevant, generates a HyDE-enhanced query.
2. **`retrieve`** — Embeds the enhanced query (blended with recent chat context) and runs a top-k similarity search against the document's Pinecone namespace.
3. **`generate`** — Produces a grounded answer from the retrieved chunks. On a retry, the previous rejected answer and its failure reason are injected into the prompt for targeted correction.
4. **`evaluate_node`** — Scores the answer for grounding and relevance, computing an overall confidence score.
5. **`decide_after_evaluate`** (conditional edge):
   - confidence **≥ threshold** → `update_memory` → `END`
   - confidence **< threshold** and `loop_count < 3` → back to `generate` (self-correction)
   - confidence **< threshold** and `loop_count ≥ 3` → accept best effort → `update_memory` → `END`
6. **`update_memory`** — Appends the Q&A to the sliding-window chat history and carries final scores/sources into the output state.

The default confidence threshold is **0.7** and the maximum number of regeneration loops is **3** — both configurable per request / in `config.py`.

---

## Project Structure

```
lawgorithm/
├── agents/                 # LLM-backed reasoning agents
│   ├── router.py           # relevance routing + HyDE enhancement
│   └── evaluate.py         # grounding / relevance / confidence scoring
├── graph/                  # LangGraph state machine
│   ├── state.py            # GraphState TypedDict
│   ├── nodes.py            # node implementations
│   ├── edges.py            # conditional routing logic
│   └── workflow.py         # graph assembly + run_graph()
├── ingestion/              # document → vector pipeline
│   ├── document_loader.py  # PDF/text loading (PyMuPDF)
│   ├── chunker.py          # recursive character chunking
│   ├── embedder.py         # embeddings + Pinecone upsert
│   └── github_loader.py    # (legacy) repo file loader
├── utils/
│   └── llm_factory.py      # provider-agnostic get_llm()
├── tests/                  # unit, integration, and API tests
│   ├── unit/               # agents + ingestion
│   ├── integration/        # end-to-end workflow
│   └── api/                # FastAPI endpoint tests
├── .github/workflows/      # CI: lint, multi-version tests, scheduled runs
├── app.py                  # Streamlit web application
├── api.py                  # FastAPI REST service
├── config.py               # central configuration & model selection
├── run_ingest.py           # standalone ingestion runner
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | LangGraph, LangChain |
| **LLMs** | Groq (`llama-3.3-70b-versatile`), Google Gemini (`gemini-2.0-flash-lite`) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional) |
| **Vector DB** | Pinecone (namespaced) |
| **Document Parsing** | PyMuPDF (`fitz`), pypdf, python-docx |
| **Web UI** | Streamlit |
| **API** | FastAPI + Uvicorn (OpenAPI/Swagger) |
| **Testing** | pytest, pytest-asyncio, pytest-cov, httpx |
| **Language** | Python 3.10–3.12 |

---

## Getting Started

### Prerequisites

- Python **3.10+**
- A **Pinecone** account with an index created (see below)
- An API key for **Groq** and/or **Google Gemini**

### 1. Clone & install

```bash
git clone https://github.com/aadyagupta44/lawgorithm.git
cd lawgorithm

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Create the Pinecone index

Lawgorithm does **not** auto-create the index — create it in the Pinecone dashboard with settings that match `config.py`:

- **Dimension:** `384` (must match the embedding model)
- **Metric:** `cosine`
- **Name:** `lawgorithm` (or set `PINECONE_INDEX_NAME`)

### 3. Configure environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=lawgorithm
```

> Keys are loaded via `python-dotenv` and are never hardcoded. `.env` is gitignored.

---

## Configuration

All tunable behavior lives in [`config.py`](config.py):

| Setting | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `"groq"` | Switch the entire app between `"groq"` and `"gemini"` — one line. |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq chat model. |
| `GEMINI_MODEL` | `gemini-2.0-flash-lite` | Gemini chat model. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer for embeddings. |
| `PINECONE_DIMENSION` | `384` | Must match the index and embedding model. |
| `CHUNK_SIZE` | `1000` | Chunk size in characters. |
| `CHUNK_OVERLAP` | `200` | Overlap to avoid clauses being split at boundaries. |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query. |
| `MAX_LOOP_COUNT` | `3` | Max self-correction regenerations. |
| `MAX_CHAT_HISTORY` | `10` | Conversational memory window (turns). |

---

## Usage

### Streamlit Web App

```bash
streamlit run app.py
```

Then open the local URL (default `http://localhost:8501`). From the UI you can:

- Upload PDF documents or paste raw legal text
- Adjust the confidence threshold and provider settings
- Ask questions with full conversational context
- See per-answer confidence/grounding/relevance scores, source files & pages, and the live correction log
- Use **Quick Actions** (e.g., extract deadlines, summarize) and example prompts

### Standalone Ingestion

`run_ingest.py` demonstrates the end-to-end ingestion pipeline programmatically:

```bash
python run_ingest.py
```

---

## REST API Reference

Start the API server:

```bash
uvicorn api:app --reload
```

Interactive Swagger docs are available at **`http://localhost:8000/docs`**.

### Endpoints

#### `POST /ingest/file` — ingest a PDF
Multipart upload of a `.pdf`. Returns the generated namespace and counts.

```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@/path/to/contract.pdf"
```

```json
{
  "message": "File ingested successfully.",
  "namespace": "contract",
  "chunks_created": 42,
  "vectors_stored": 42
}
```

#### `POST /ingest/text` — ingest raw text

```bash
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "This Agreement is made between...", "document_name": "MyDoc"}'
```

#### `POST /query` — ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
        "question": "What is the termination notice period?",
        "namespace": "contract",
        "confidence_threshold": 0.7,
        "chat_history": []
      }'
```

```json
{
  "generation": "The agreement requires 30 days written notice...",
  "source_files": ["contract.pdf"],
  "source_pages": [3, 4],
  "loop_count": 0,
  "confidence_score": 0.92,
  "relevance_score": 0.90,
  "hallucination_score": 0.95
}
```

> **Note:** the `namespace` returned by an ingest call is the handle you pass to `/query`. Namespaces are derived from the (sanitized) document filename.

---

## Testing

The suite uses mocked Pinecone and LLM clients (see `tests/conftest.py`), so no live API keys are required to run it.

```bash
# run everything
pytest

# with coverage
pytest --cov=. --cov-report=term-missing

# a single layer
pytest tests/unit
pytest tests/integration
pytest tests/api
```

Test layout:

- `tests/unit/` — agents and ingestion components
- `tests/integration/` — end-to-end LangGraph workflow
- `tests/api/` — FastAPI endpoint behavior

---

## Continuous Integration

GitHub Actions workflows live in `.github/workflows/`:

| Workflow | Purpose |
|---|---|
| `python-ci.yml` | Install deps and run the test suite on push/PR (Python 3.10). |
| `test.yml` | Test matrix across Python **3.10 / 3.11 / 3.12**. |
| `lint.yml` | Static linting (Python 3.11). |
| `scheduled-tests.yml` | Periodic scheduled test runs. |

Dependabot (`.github/dependabot.yml`) keeps dependencies current.

---

## Design Notes & Trade-offs

- **Why HyDE?** Short user questions embed poorly against dense legal prose. Generating a hypothetical answering clause and embedding *that* dramatically improves retrieval recall on legal corpora.
- **Why a separate evaluator + loop?** Legal answers must be grounded. Rather than trusting a single generation, Lawgorithm measures grounding/relevance and feeds concrete failure reasons back into a bounded correction loop — trading a few extra LLM calls for substantially higher answer fidelity.
- **Why namespaces per document?** Isolation prevents cross-document contamination of retrieval and lets a single shared index serve many independent documents cheaply.
- **Graceful degradation:** Every external dependency (embedder, Pinecone, LLM, agents) is initialized defensively. If a component is unavailable, the workflow logs the issue and falls back to safe defaults rather than crashing.
- **Provider abstraction:** The `get_llm()` factory means switching model providers is a one-line change with zero edits to agent code.

> ⚠️ **Disclaimer:** Lawgorithm is an AI assistant for document analysis and is **not a substitute for professional legal advice**. Always consult a qualified attorney for legal decisions.

---

## Roadmap

- Multi-document / cross-namespace querying in a single question
- Persistent chat sessions and user accounts
- Streaming responses in both UI and API
- Configurable index auto-provisioning
- Re-ranking layer on top of vector retrieval

---

<p align="center"><i>Built with LangGraph, Pinecone, and a healthy distrust of hallucinations.</i></p>
