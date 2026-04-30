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
