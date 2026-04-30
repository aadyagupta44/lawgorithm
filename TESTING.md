# Testing Guide - Capstone RAG

## Quick Start

```bash
# Run all tests
pytest tests/ -v

# Run unit tests only (fastest)
pytest tests/unit/ -v

# Run with coverage report
pytest tests/ --cov=agents --cov=ingestion --cov=graph --cov=utils
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures for all tests
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_agents.py       # Router and Evaluate agent tests
│   ├── test_ingestion.py    # Document loading, chunking, embedding tests
├── integration/             # Integration tests (medium speed)
│   └── test_workflow.py     # Full workflow graph tests
└── api/                     # API endpoint tests
    └── test_endpoints.py    # FastAPI endpoint tests
```

## Fixtures (conftest.py)

All test files use these fixtures from `tests/conftest.py`:

| Fixture | Purpose |
|---------|---------|
| `api_client` | FastAPI TestClient for endpoint testing |
| `mock_google_genai` | Mocks LLM factory (prevents API calls) |
| `mock_pinecone` | Mocks Pinecone client |
| `mock_embeddings` | Mocks SentenceTransformer |
| `sample_document` | Sample legal document text |

### Using Fixtures

```python
def test_something(mock_google_genai):
    """Fixture is injected as parameter."""
    mock_google_genai.invoke.return_value = "mocked response"
    # ... test code ...
```

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Suite
```bash
pytest tests/unit/ -v                    # Unit tests only
pytest tests/integration/ -v             # Integration tests
pytest tests/api/ -v                     # API tests
```

### Run Single Test File
```bash
pytest tests/unit/test_agents.py -v
```

### Run Single Test
```bash
pytest tests/unit/test_agents.py::test_router_agent_route_in_scope -v
```

### Run with Options
```bash
pytest tests/ -v -s                      # Show print statements
pytest tests/ -v -x                      # Stop on first failure
pytest tests/ -v --tb=short              # Short traceback format
pytest tests/ -v --lf                    # Run last failed tests
pytest tests/ -v -k "router"             # Run tests matching keyword
```

## Coverage Reports

### Generate Coverage Report
```bash
pip install pytest-cov
pytest tests/ --cov=agents --cov=ingestion --cov=graph --cov=utils --cov-report=html
```

This creates an HTML report in `htmlcov/index.html`. Open in browser to see:
- % of code covered by tests
- Which lines are tested/untested
- Coverage by module

### View Terminal Coverage
```bash
pytest tests/ --cov=agents --cov-report=term-missing
```

Shows coverage in terminal with missing lines listed.

## Test Categories

### Unit Tests (tests/unit/)

**test_agents.py:**
- `test_router_agent_route_in_scope` - Router classifies legal query as relevant
- `test_router_agent_route_out_of_scope` - Router rejects non-legal query
- `test_evaluate_agent_relevance` - Evaluator scores answers correctly
- `test_evaluate_agent_hallucination_empty_generation` - Handles empty answers

**test_ingestion.py:**
- `test_document_loader_text` - Load text as document
- `test_document_loader_pdf_happy` - Load PDF successfully
- `test_document_loader_pdf_not_found` - Handle missing PDF gracefully
- `test_chunker_basic` - Split document into chunks
- `test_chunker_empty` - Handle empty documents
- `test_pinecone_embedder_namespace` - Generate valid namespaces
- `test_pinecone_embedder_store` - Store embeddings and return count

### Integration Tests (tests/integration/)

**test_workflow.py:**
- `test_workflow_compilation` - Graph compiles successfully
- `test_workflow_execution_mocked` - Graph nodes exist and connect

### API Tests (tests/api/)

**test_endpoints.py:**
- `test_ingest_text_happy` - POST /ingest/text works
- `test_ingest_text_sad_missing_payload` - Missing fields return 422
- `test_ingest_file_sad_wrong_type` - Non-PDF files rejected
- `test_query_happy` - POST /query works
- `test_query_sad_missing_namespace` - Missing fields return 422

## Environment Variables

Tests use these environment variables (optional, defaults provided):

```bash
export GROQ_API_KEY="your-key-here"
export GEMINI_API_KEY="your-key-here"
export PINECONE_API_KEY="your-key-here"
export PINECONE_INDEX_NAME="lawgorithm"
```

If not set, tests use mocked versions and still pass.

## Mocking

Tests mock external APIs to avoid:
- Real API calls (slow, expensive)
- Network failures
- Rate limiting
- Dependency on API availability

### Mock Usage Examples

```python
# Mock LLM response
mock_google_genai.invoke.return_value = "mocked response"

# Mock Pinecone index
embedder.index = MagicMock()
embedder.index.upsert.assert_called_once()

# Mock embeddings
embedder.model.encode.return_value = np.array([[0.1, 0.2]])
```

## Debugging Tests

### See Print Statements
```bash
pytest tests/unit/test_agents.py -v -s
```

### Drop into Debugger
```python
def test_something():
    import pdb; pdb.set_trace()  # Execution pauses here
    # ... code ...
```

### Run with Full Traceback
```bash
pytest tests/ -v --tb=long
```

### Show Local Variables on Failure
```bash
pytest tests/ -v -l
```

## Common Issues

### Import Errors
```
ModuleNotFoundError: No module named 'agents'
```
**Solution:** Run from project root: `cd capstone-rag && pytest tests/`

### Fixture Not Found
```
fixture 'mock_google_genai' not found
```
**Solution:** Make sure `tests/conftest.py` exists in same directory

### Tests Pass Locally but Fail on GitHub
- Check Python version (use 3.9-3.12)
- Check environment variables
- Check file paths (/ vs \)

### Mock Not Working
```python
# Wrong - patches after import
from agents.router import RouterAgent
with patch("agents.router.get_llm"):
    pass

# Right - patch where it's used
with patch("utils.llm_factory.get_llm"):
    pass
```

## Best Practices

✅ **DO:**
- Run tests locally before pushing
- Keep tests isolated and fast
- Mock external APIs
- Test edge cases (empty, invalid input)
- Use descriptive test names

❌ **DON'T:**
- Make real API calls in tests
- Use global state between tests
- Create files/databases in tests
- Test implementation details, test behavior

## CI/CD Integration

Tests run automatically on:
1. **Every push** - GitHub Actions runs `test.yml`
2. **Every PR** - GitHub blocks merge if tests fail
3. **Nightly** - `scheduled-tests.yml` runs full suite

See `.github/GITHUB_ACTIONS_GUIDE.md` for details.

## Continuous Improvement

- Add tests when finding bugs
- Increase coverage gradually (aim for 80%+)
- Review test results in Actions tab
- Fix failing tests immediately
