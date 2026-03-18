#  Lawgorithm - Intelligent Legal Document Assistant

> An AI-powered legal document analysis system built with LangGraph, Groq, and Pinecone. Upload any contract or legal document and get accurate, verified answers with real-time self-correction visualization.

----

## 🎯 What Is Lawgorithm?

Lawgorithm is a **graph-orchestrated agentic self-correcting RAG system** designed for legal document analysis. It allows anyone - not just lawyers- to upload legal documents and ask questions in plain English. The system retrieves relevant passages, generates answers, and then **verifies its own output** before showing it to the user.

The key innovation is **visible self-correction**. Every step the AI takes - searching documents, grading relevance, checking for hallucinations, rewriting queries - is shown in real time on screen. This makes the system transparent, trustworthy, and genuinely useful for legal work.

---

##  Key Features

- ** Multi-Document Support** — Upload multiple PDFs or paste raw legal text
- ** Semantic Search** — Pinecone vector database finds relevant passages instantly
- ** Self-Correction Loop** — System catches its own mistakes and retries automatically
- ** Live Reasoning Trace** — Watch every AI step in real time on screen
- ** Risk Analysis** — Automatically flags dangerous or unfavorable clauses
- ** Conversational Memory** — Remembers previous questions for context
- ** Source Citations** — Every answer cites the exact document and page
- ** Plain English** — Complex legal language explained simply
- ** 10 Specialist Agents** — Each agent has a specific legal job

---

##  System Architecture

```
User Question
      │
      ▼
┌─────────────┐
│   Router    │ ← Is this a legal question?
│   Agent     │
└──────┬──────┘
       │ relevant
       ▼
┌─────────────┐
│  Retriever  │ ← Search Pinecone vector database
│    Node     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Relevance  │ ← Are these documents relevant?
│   Grader    │
└──────┬──────┘
       │ relevant        │ irrelevant
       │                 ▼
       │          ┌─────────────┐
       │          │   Query     │ ← Rewrite question
       │          │  Rewriter   │   and try again
       │          └──────┬──────┘
       │                 │ (loop back to retriever)
       ▼
┌─────────────┐
│  Generator  │ ← Create answer from documents
│    Node     │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Hallucination   │ ← Is every claim backed by documents?
│    Grader       │
└──────┬──────────┘
       │ grounded        │ hallucinated
       │                 ▼
       │          ┌─────────────┐
       │          │  Generator  │ ← Regenerate answer
       │          └─────────────┘
       ▼
┌─────────────┐
│   Answer    │ ← Is the answer actually useful?
│   Grader    │
└──────┬──────┘
       │ useful          │ not useful
       │                 ▼
       │          ┌─────────────┐
       │          │   Query     │ ← Rewrite and retry
       │          │  Rewriter   │
       │          └─────────────┘
       ▼
┌─────────────┐
│   Memory    │ ← Save Q&A to conversation history
│  Updater    │
└──────┬──────┘
       │
       ▼
   Final Answer
```

---

##  The Agent System

Lawgorithm uses **10 specialist AI agents**, each with a specific legal job:

| Agent | File | Purpose |
|-------|------|---------|
| RouterAgent | `router.py` | Decides if question is about legal documents |
| RelevanceGrader | `graders.py` | Checks if retrieved chunks answer the question |
| HallucinationGrader | `graders.py` | Verifies every claim is backed by documents |
| AnswerGrader | `graders.py` | Checks if answer is complete and useful |
| RiskFlagGrader | `graders.py` | Identifies dangerous or unfavorable clauses |
| QueryRewriter | `rewriter.py` | Rewrites questions for better retrieval |
| PlainEnglishExplainer | `explainer.py` | Translates legal jargon to simple language |
| ClauseIdentifierAgent | `clause_identifier.py` | Maps all clauses in a document |
| ContractSummarizerAgent | `summarizer.py` | Creates executive summaries |
| ComparisonAgent | `comparison.py` | Compares two documents side by side |
| DeadlineExtractorAgent | `deadline_extractor.py` | Finds all dates and deadlines |
| FavorabilityAgent | `favorability.py` | Scores contract from user's perspective |
| RedlineAgent | `redline.py` | Suggests improvements to bad clauses |

---

##  Project Structure

```
lawgorithm/
├── .env                          # API keys (never committed)
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── config.py                     # Configuration constants
├── app.py                        # Streamlit UI entry point
│
├── ingestion/                    # Document processing pipeline
│   ├── __init__.py
│   ├── document_loader.py        # PDF and text loading
│   ├── chunker.py                # Text splitting
│   └── embedder.py               # Pinecone vector storage
│
├── agents/                       # All AI specialist agents
│   ├── __init__.py
│   ├── router.py                 # Question router
│   ├── graders.py                # All grader agents
│   ├── rewriter.py               # Query rewriter
│   ├── explainer.py              # Plain English explainer
│   ├── clause_identifier.py      # Clause mapper
│   ├── summarizer.py             # Contract summarizer
│   ├── comparison.py             # Document comparator
│   ├── deadline_extractor.py     # Deadline finder
│   ├── favorability.py           # Favorability scorer
│   └── redline.py                # Clause improver
│
├── graph/                        # LangGraph workflow
│   ├── __init__.py
│   ├── state.py                  # GraphState TypedDict
│   ├── nodes.py                  # All node functions
│   ├── edges.py                  # Conditional edge logic
│   └── workflow.py               # Graph assembly
│
└── .opencode/                    # OpenCode agent briefs
    └── agents/                   # Markdown briefs for agents
```

---

## 🛠️ Tech Stack

| Technology | Role | Why |
|------------|------|-----|
| Python 3.11+ | Language | Modern, type-safe |
| LangGraph | Graph orchestration | Self-correction loops |
| LangChain | LLM framework | Agent abstractions |
| Groq (llama-3.3-70b) | LLM | Fast, free, powerful |
| Pinecone | Vector database | Production-grade search |
| sentence-transformers | Embeddings | Free, accurate |
| PyMuPDF | PDF reading | Fast, reliable |
| Streamlit | UI | Rapid professional UI |
| OpenCode | Agent development | AI-assisted coding |

---

##  Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/lawgorithm.git
cd lawgorithm
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Get API Keys

You need three free API keys:

**Groq API Key:**
- Go to console.groq.com
- Sign up and go to API Keys
- Create new key
- Free tier: 100,000 tokens/day

**Pinecone API Key:**
- Go to pinecone.io
- Create free account
- Go to API Keys
- Create index named `lawgorithm` with dimension 384, metric cosine

**GitHub Token (optional, for ingesting GitHub repos):**
- Go to github.com → Settings → Developer Settings
- Personal Access Tokens → Tokens Classic
- Generate with `repo` scope

### 5. Create .env File

```bash
# Windows
New-Item .env
notepad .env
```

Add these lines:

```
GROQ_API_KEY=your_groq_key_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=lawgorithm
```

### 6. Run the Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

##  How to Use

### Step 1 — Upload Documents
- Click "Upload Legal Documents" in the sidebar
- Upload one or more PDF files
- Or paste raw legal text directly
- Click " Process Documents"

### Step 2 — Wait for Processing
The system will:
1. Extract text from PDFs page by page
2. Split text into overlapping chunks
3. Generate semantic embeddings
4. Store vectors in Pinecone

### Step 3 — Ask Questions
Type any legal question in the chat:
- "Summarize this contract"
- "What are my obligations under this agreement?"
- "Flag any risky or unfavorable clauses"
- "Explain the termination clause in plain English"
- "What happens if I want to exit early?"
- "Extract all important dates and deadlines"
- "Is this contract favorable to me as an employee?"

### Step 4 — Watch the AI Reason
The **AI Reasoning Trace** panel on the right shows every step:
-  Searching documents
-  Grading relevance
-  Generating answer
-  Detecting issues
-  Self-correcting
-  Final verification

---

##  Self-Correction System — The Core Innovation

Most AI systems just answer questions. Lawgorithm **verifies its own answers** through a multi-step correction loop:

**Loop 1 — Relevance Check**
After retrieving documents, the system grades each one. If none are relevant, it automatically rewrites the query and searches again.

**Loop 2 — Hallucination Check**
After generating an answer, the system checks every claim against the source documents. If any claim is not backed by a document, it regenerates.

**Loop 3 — Quality Check**
After passing hallucination check, the system evaluates if the answer actually addresses the question. If not, it rewrites the query and retries.

Maximum 3 correction attempts. Every attempt is logged and shown on screen.

This makes Lawgorithm particularly suitable for legal work where accuracy is critical.

---

##  Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| GROQ_API_KEY | Yes | Groq LLM API key |
| PINECONE_API_KEY | Yes | Pinecone vector DB key |
| PINECONE_INDEX_NAME | Yes | Index name (default: lawgorithm) |

---

##  Requirements

```
streamlit>=1.32.0
langchain>=0.2.0
langchain-groq>=0.1.0
langchain-google-genai>=1.0.0
langchain-text-splitters>=0.2.0
langgraph>=0.1.0
pinecone>=3.0.0
sentence-transformers>=2.7.0
pymupdf>=1.23.0
pypdf>=3.0.0
python-dotenv>=1.0.0
PyGithub>=2.0.0

---

##  Built With

- **LangGraph** — Graph-based agent orchestration
- **Groq** — Ultra-fast LLM inference
- **Pinecone** — Production vector database
- **Streamlit** — Professional web UI
- **OpenCode** — AI-assisted agent development

---

