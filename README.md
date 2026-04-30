<<<<<<< HEAD
# Capstone RAG - Intelligent Legal Document Assistant

Professional-grade RAG (Retrieval-Augmented Generation) system for legal document analysis using LangGraph, Groq/Gemini LLMs, and Pinecone vector database.

## Features

✨ **Self-Correcting RAG** - LangGraph workflow automatically retries and improves answers
🧠 **Multiple LLM Support** - Switch between Groq (fast) and Google Gemini (capable) with one config
📚 **Legal Document Processing** - Load PDF, text, and Word documents
🔍 **Vector Search** - Pinecone-powered semantic search with namespaces
💾 **Chat Memory** - Maintains conversation history across queries
⚖️ **Risk Analysis** - Identifies potential issues in legal documents
🚀 **Production Ready** - FastAPI backend + Streamlit UI
🧪 **Full Test Coverage** - Unit, integration, and API tests
🤖 **CI/CD Pipeline** - GitHub Actions for automated testing

## Quick Start

### 1. Clone & Setup

```bash
# Clone repository
git clone https://github.com/yourusername/capstone-rag.git
cd capstone-rag

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file in project root:

```env
# LLM Configuration
LLM_PROVIDER=groq                           # or "gemini"
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=lawgorithm

# Optional
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

**Get API Keys:**
- [Groq API](https://console.groq.com) - Free tier available
- [Google Gemini](https://aistudio.google.com) - Free API key
- [Pinecone](https://app.pinecone.io) - Free tier (1M vectors)

### 3. Run Everything

```bash
# Run tests first
pytest tests/ -v

# Start Streamlit UI
streamlit run app.py

# Or use FastAPI backend
python -m uvicorn api:app --reload --port 8000

# Run ingestion pipeline
python run_ingest.py
```

## Project Structure

```
capstone-rag/
├── agents/                      # Intelligent agents
│   ├── router.py               # Query relevance classifier + HyDE enhancement
│   ├── evaluate.py             # Answer quality evaluator
│   └── __init__.py
├── graph/                       # LangGraph workflow
│   ├── workflow.py             # Main graph assembly
│   ├── nodes.py                # Node implementations
│   ├── edges.py                # Conditional routing logic
│   ├── state.py                # Graph state schema
│   └── __init__.py
├── ingestion/                   # Document processing pipeline
│   ├── document_loader.py      # Load PDF, text, Word files
│   ├── chunker.py              # Split documents into chunks
│   ├── embedder.py             # Vector embeddings + Pinecone storage
│   ├── github_loader.py        # Load from GitHub repos
│   └── __init__.py
├── utils/                       # Utilities
│   ├── llm_factory.py          # LLM provider switching
│   ├── helpers.py              # Helper functions
│   └── __init__.py
├── tests/                       # Test suite
│   ├── conftest.py             # Pytest fixtures
│   ├── unit/                   # Unit tests (fast)
│   ├── integration/            # Integration tests
│   └── api/                    # API endpoint tests
├── .github/
│   ├── workflows/              # GitHub Actions CI/CD
│   │   ├── test.yml            # Run tests on push/PR
│   │   ├── lint.yml            # Code quality checks
│   │   └── scheduled-tests.yml # Nightly tests
│   └── GITHUB_ACTIONS_GUIDE.md # Full CI/CD documentation
├── app.py                       # Streamlit UI
├── api.py                       # FastAPI backend
├── config.py                    # Configuration
├── requirements.txt             # Python dependencies
├── TESTING.md                   # Testing documentation
└── README.md                    # This file
```

## Core Components

### 1. Router Agent (`agents/router.py`)
- Classifies if query is legal-related
- Uses HyDE (Hypothetical Document Embeddings) for query enhancement
- Returns: `(decision: "relevant"/"irrelevant", enhanced_query: str)`

### 2. Ingestion Pipeline (`ingestion/`)
- **DocumentLoader** - Load PDFs, text, Word docs
- **DocumentChunker** - Split into overlapping chunks (1000 chars, 200 overlap)
- **PineconeEmbedder** - Generate embeddings + store in Pinecone

### 3. LangGraph Workflow (`graph/workflow.py`)
```
START → router_node → retrieve → generate → evaluate_node → END
                        ↓
                   (if irrelevant)
                        ↓
                handle_out_of_scope → END
                
(if confidence < threshold, retry generate)
```

### 4. Evaluate Agent (`agents/evaluate.py`)
- Scores answer on: hallucination, relevance, confidence (0.0-1.0)
- Extracts reasoning for scores
- Enables self-correction loop

### 5. FastAPI Backend (`api.py`)
- `/ingest/text` - Ingest text documents
- `/ingest/file` - Upload PDF files
- `/query` - Query documents with chat history

### 6. Streamlit UI (`app.py`)
- Document ingestion interface
- Query UI with chat history
- Real-time reasoning traces
- Risk analysis display

## Running the Application

### Option 1: Streamlit UI (Recommended)
```bash
# Terminal 1: Start Streamlit
streamlit run app.py

# Streamlit opens at http://localhost:8501
# Upload documents → Ask questions → Get answers with reasoning
```

### Option 2: FastAPI Backend
```bash
# Terminal 1: Start API server
python -m uvicorn api:app --reload --port 8000

# Terminal 2: Query the API
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main terms?",
    "namespace": "contract-pdf"
  }'

# OpenAPI docs at http://localhost:8000/docs
```

### Option 3: Command Line
```bash
# Ingest documents
python run_ingest.py

# Query programmatically
python -c "
from graph import run_graph
result = run_graph(
    question='What are the payment terms?',
    namespace='my-documents'
)
print(result['generation'])
"
```

## Testing

### Run All Tests
```bash
# Quick unit tests only
pytest tests/unit/ -v

# Full test suite
pytest tests/ -v

# With coverage report
pytest tests/ --cov=agents --cov=ingestion --cov=graph --cov-report=html
```

### Test Coverage
- **Unit Tests** - Fast, isolated component tests
- **Integration Tests** - Full workflow testing
- **API Tests** - Endpoint validation

See `TESTING.md` for detailed testing guide.

## GitHub Actions CI/CD

Automated testing on every push and PR:

1. **test.yml** - Runs tests on Python 3.9, 3.10, 3.11, 3.12
2. **lint.yml** - Code quality checks (Ruff, Black, isort)
3. **scheduled-tests.yml** - Nightly test runs

View results in repo **Actions** tab.

See `.github/GITHUB_ACTIONS_GUIDE.md` for full CI/CD documentation.

## Configuration

Edit `config.py` to customize:

```python
# LLM Provider
LLM_PROVIDER = "groq"              # Switch to "gemini" for Google

# Chunking
CHUNK_SIZE = 1000                  # Characters per chunk
CHUNK_OVERLAP = 200                # Overlap between chunks

# Retrieval
TOP_K_RESULTS = 5                  # Documents to retrieve

# Self-Correction
MAX_LOOP_COUNT = 3                 # Max retry attempts
CONFIDENCE_THRESHOLD = 0.7         # Min acceptable confidence

# Embedding
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PINECONE_DIMENSION = 384
```

## API Reference

### POST /ingest/text
Ingest text document

```bash
curl -X POST "http://localhost:8000/ingest/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Legal document text here...",
    "document_name": "contract.pdf"
  }'
```

**Response:**
```json
{
  "message": "Text ingested successfully.",
  "namespace": "contract_pdf",
  "chunks_created": 5,
  "vectors_stored": 5
}
```

### POST /ingest/file
Upload PDF file

```bash
curl -X POST "http://localhost:8000/ingest/file" \
  -F "file=@contract.pdf"
```

### POST /query
Query documents

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the payment terms?",
    "namespace": "contract-pdf",
    "confidence_threshold": 0.7,
    "chat_history": [
      {"role": "user", "content": "Previous question"},
      {"role": "assistant", "content": "Previous answer"}
    ]
  }'
```

**Response:**
```json
{
  "generation": "The payment terms are...",
  "source_files": ["contract.pdf"],
  "source_pages": [1, 2],
  "loop_count": 1,
  "confidence_score": 0.92,
  "relevance_score": 0.95,
  "hallucination_score": 0.88
}
```

## Troubleshooting

### API Keys Not Working
- Check `.env` file exists and is readable
- Verify API keys are correct (check provider's console)
- Ensure `python-dotenv` is installed

### Pinecone Index Not Found
```bash
# Create index in Pinecone dashboard:
# Name: lawgorithm
# Dimension: 384
# Metric: cosine
```

### Tests Failing
```bash
# Run specific test with full output
pytest tests/unit/test_agents.py -v -s

# Check imports work
python -c "from agents.router import RouterAgent; print('OK')"
```

### Slow Responses
- Reduce `TOP_K_RESULTS` in config
- Use Groq instead of Gemini (faster)
- Check Pinecone connection status

## Development

### Code Style
```bash
# Auto-format code
black .
isort .

# Check for issues
ruff check .
flake8 .
```

### Adding Tests
```python
# tests/unit/test_new_feature.py
import pytest
from agents.router import RouterAgent

def test_new_feature(mock_google_genai):
    """Test description."""
    agent = RouterAgent()
    agent.llm = mock_google_genai
    result = agent.route("test query")
    assert result[0] == "relevant"
```

### Contributing
1. Create feature branch: `git checkout -b feature/my-feature`
2. Write tests: `pytest tests/`
3. Run linting: `ruff check .`
4. Commit: `git commit -m "Add feature"`
5. Push: `git push origin feature/my-feature`
6. Open PR on GitHub

## License

[Your License Here] - See LICENSE file

## Support

- 📖 Documentation: See `TESTING.md` and `.github/GITHUB_ACTIONS_GUIDE.md`
- 🐛 Issues: Open GitHub Issue
- 💬 Discussions: GitHub Discussions
- 📧 Email: your-email@example.com

## Roadmap

- [ ] Web UI deployment (Vercel/Heroku)
- [ ] Multi-language support
- [ ] Real-time collaboration
- [ ] Advanced analytics dashboard
- [ ] Custom fine-tuning for legal domains
- [ ] Mobile app

---

**Built with** 💜 using LangGraph, Groq, and Pinecone
=======
#  Lawgorithm - Intelligent Legal Document Assistant

> An AI-powered legal document analysis system built with LangGraph, Groq, and Pinecone. Upload any contract or legal document and get accurate, verified answers with real-time self-correction visualization.


---

##  What Is Lawgorithm?

Lawgorithm is a **graph-orchestrated agentic self-correcting RAG system** designed for legal document analysis. It allows anyone - not just lawyers- to upload legal documents and ask questions in plain English. The system retrieves relevant passages, generates answers, and then **verifies its own output** before showing it to the user.

The key innovation is **visible self-correction**. Every step the AI takes - searching documents, grading relevance, checking for hallucinations, rewriting queries - is shown in real time on screen. This makes the system transparent, trustworthy, and genuinely useful for legal work.

---

##  Key Features

- **Multi-Document Support** — Upload multiple PDFs or paste raw legal text
- **Semantic Search** — Pinecone vector database finds relevant passages instantly
- **Self-Correction Loop** — System catches its own mistakes and retries automatically
- **Live Reasoning Trace** — Watch every AI step in real time on screen
- **Risk Analysis** — Automatically flags dangerous or unfavorable clauses
- **Conversational Memory** — Remembers previous questions for context
- **Source Citations** — Every answer cites the exact document and page
- **Plain English** — Complex legal language explained simply
- **10 Specialist Agents** — Each agent has a specific legal job

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

##  Tech Stack

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
```

---

##  Built With

- **LangGraph** — Graph-based agent orchestration
- **Groq** — Ultra-fast LLM inference
- **Pinecone** — Production vector database
- **Streamlit** — Professional web UI
- **OpenCode** — AI-assisted agent development

---

>>>>>>> 08c2714cf11edb433cd8c6ab1821821e134bcace
