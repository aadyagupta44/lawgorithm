# Quick Start Guide - Capstone RAG

Complete setup and run instructions in 5 minutes.

## 1. Setup (2 minutes)

```bash
# Clone and enter directory
git clone <your-repo-url>
cd capstone-rag

# Create virtual environment
python -m venv venv

# Activate (choose your OS)
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Configure (1 minute)

Create `.env` file in project root:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
GEMINI_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=lawgorithm
```

Get free API keys:
- **Groq**: https://console.groq.com (sign up, get API key)
- **Gemini**: https://aistudio.google.com (sign up, get API key)
- **Pinecone**: https://app.pinecone.io (sign up, create index `lawgorithm` with 384 dimensions)

## 3. Test (1 minute)

```bash
# Run tests to verify setup
pytest tests/unit/ -v

# You should see all tests pass ✓
```

## 4. Run (1 minute)

Choose one option:

### Option A: Streamlit UI (Easiest)
```bash
streamlit run app.py
# Opens browser at http://localhost:8501
# Upload documents → Ask questions → Get answers
```

### Option B: FastAPI Backend
```bash
python -m uvicorn api:app --reload --port 8000
# API docs at http://localhost:8000/docs
# Test with curl or Postman
```

### Option C: Python Script
```python
# query_example.py
from graph import run_graph

result = run_graph(
    question="What are the payment terms?",
    namespace="my-documents"
)
print(result['generation'])
```

Then run:
```bash
python query_example.py
```

## Common Commands

```bash
# Run all tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=agents --cov=ingestion --cov=graph

# Check code quality
ruff check .

# Format code
black .
isort .

# Ingest documents
python run_ingest.py

# Activate environment (do this first!)
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

## File Locations

- **Config**: `config.py` (change LLM, chunk size, etc.)
- **Environment**: `.env` (store API keys)
- **Tests**: `tests/` directory
- **Workflows**: `.github/workflows/` (CI/CD)
- **API**: `api.py` (FastAPI endpoints)
- **UI**: `app.py` (Streamlit interface)
- **Docs**: `TESTING.md`, `README.md`, `.github/GITHUB_ACTIONS_GUIDE.md`

## Troubleshooting

**Issue: "ModuleNotFoundError: No module named 'agents'"**
- Solution: Make sure you're in project root and venv is activated

**Issue: "API key not found"**
- Solution: Create `.env` file with your API keys
- Make sure you added `.env` to `.gitignore` (never commit keys!)

**Issue: "Pinecone index not found"**
- Solution: Create index in Pinecone dashboard named `lawgorithm` with 384 dimensions

**Issue: Tests fail**
- Solution: Run `pytest tests/unit/ -v -s` to see detailed output

## Next Steps

1. Read `README.md` for full documentation
2. See `TESTING.md` for testing details
3. Check `.github/GITHUB_ACTIONS_GUIDE.md` for CI/CD setup
4. Start ingesting documents
5. Ask questions and get answers!

---

**You're ready to go!** 🚀

Start with Streamlit: `streamlit run app.py`
