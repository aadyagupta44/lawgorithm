# GitHub Actions Setup - Capstone RAG

## Overview

This project uses **GitHub Actions** for continuous integration and continuous deployment (CI/CD). Workflows automatically test, lint, and validate your code every time you push changes or create a pull request.

## What are GitHub Actions?

GitHub Actions is an automation tool built into GitHub that:
- **Watches** your repository for events (push, PR, schedule)
- **Triggers** workflows automatically when events occur
- **Runs** jobs (tests, linting, deployments) on GitHub's servers or self-hosted runners
- **Reports** results back to your commits and PRs

### Real-World Example
Instead of manually running `pytest tests/` every time you make changes, a robot automatically:
1. Checks out your code
2. Installs dependencies
3. Runs all tests
4. Reports if anything broke
5. Blocks merging if tests fail

## Project Workflows

### 1. **test.yml** - Continuous Testing
**Triggers:** Every push to `main`/`develop` and every PR

**What it does:**
- Runs tests on Python 3.9, 3.10, 3.11, 3.12
- Tests 3 suites: unit, integration, API
- Generates coverage reports
- Uploads coverage to Codecov

**Status:** Must pass before merging PRs

### 2. **lint.yml** - Code Quality Checks
**Triggers:** Every push and PR

**What it does:**
- Checks code with Ruff (fast Python linter)
- Validates formatting with Black
- Checks import order with isort
- Runs Flake8 for additional checks

**Status:** Warnings don't block merging; errors do

### 3. **scheduled-tests.yml** - Nightly Tests
**Triggers:** Every night at 2 AM UTC + manual trigger

**What it does:**
- Runs full test suite with coverage
- Tests with real API keys from secrets
- Catches intermittent failures
- Can be triggered manually via GitHub UI

**Status:** Informational; doesn't block anything

### 4. **dependabot.yml** - Dependency Updates
**Triggers:** Weekly

**What it does:**
- Checks for outdated Python packages
- Creates PRs with updates
- Limits to 5 open PRs at a time

**Status:** Helps keep dependencies current

---

## How to Use GitHub Actions

### Automatic Usage (No Action Needed)

1. **Push code to GitHub:**
   ```bash
   git add .
   git commit -m "Add feature"
   git push origin main
   ```

2. **GitHub automatically runs workflows:**
   - Tests run immediately
   - Results appear in 2-5 minutes

3. **Check results:**
   - Go to **Actions** tab in your repo
   - See green ✓ or red ✗ status

### Manual Trigger

1. Go to **Actions** tab in your repo
2. Select workflow (e.g., "Scheduled Nightly Tests")
3. Click **Run workflow** → **Run workflow**

### View Results

**In Actions Tab:**
- See all workflow runs with timestamps
- Click any run to see detailed logs
- Scroll through output to find errors

**In Pull Requests:**
- GitHub shows test status automatically
- Red ✗ = must fix before merging
- Green ✓ = tests passed

---

## Setting Up Secrets

GitHub Actions needs API keys to run integration/API tests. Store them securely as **Secrets**.

### Add Secrets to GitHub

1. Go to your GitHub repo
2. Click **Settings** (top right)
3. Click **Secrets and variables** → **Actions** (left sidebar)
4. Click **New repository secret**

**Add these secrets:**

| Secret Name | Where to Get |
|------------|-------------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com) |
| `PINECONE_API_KEY` | [Pinecone Console](https://app.pinecone.io) |

### Using Secrets in Workflows

Secrets are automatically available in workflows as `${{ secrets.SECRET_NAME }}`

```yaml
env:
  GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
  GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
  PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}
```

**Note:** Secrets are never exposed in logs for security

---

## Local Testing (Before Pushing)

Run tests locally to catch issues before GitHub Actions:

### Quick Test (Unit Tests Only)
```bash
pytest tests/unit/ -v
```

### Full Test Suite
```bash
pytest tests/ -v
```

### With Coverage Report
```bash
pytest tests/ --cov=agents --cov=ingestion --cov=graph --cov-report=html
# Opens coverage report in htmlcov/index.html
```

### Run Specific Test
```bash
pytest tests/unit/test_agents.py::test_router_agent_route_in_scope -v
```

### Show Print Statements
```bash
pytest tests/ -v -s
```

### Stop on First Failure
```bash
pytest tests/ -x
```

### Lint Locally
```bash
# Install linting tools
pip install ruff black isort

# Check issues
ruff check .

# Auto-fix formatting
black .
isort .
```

---

## Workflow Syntax Explained

### Basic Structure

```yaml
name: Workflow Name                    # Display name in Actions tab

on:                                    # When to trigger
  push:                                # Trigger on push
    branches: [ main, develop ]        # Only these branches

jobs:
  test:                                # Job name
    runs-on: ubuntu-latest             # Run on GitHub's Ubuntu server
    
    strategy:
      matrix:
        python-version: ['3.9', '3.10']  # Run job for each Python version
    
    steps:
      - uses: actions/checkout@v4      # Step 1: Get code from repo
      
      - name: Set up Python             # Step 2: Install Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies      # Step 3: Install packages
        run: pip install -r requirements.txt
      
      - name: Run tests                 # Step 4: Execute tests
        run: pytest tests/ -v
```

### Key Concepts

**Triggers (`on`):**
- `push` - When code is pushed
- `pull_request` - When PR is created/updated
- `schedule` - On a schedule (cron)
- `workflow_dispatch` - Manual trigger

**Jobs:**
- Run on separate machines
- Can run in parallel or sequence
- Each has its own environment

**Steps:**
- Run sequentially
- Can use actions (`uses:`) or commands (`run:`)
- `continue-on-error: true` = don't fail workflow if step fails

---

## Common Tasks

### Skip Workflow for a Commit
```bash
git commit -m "Fix typo [skip ci]"    # Won't trigger workflows
```

### Run Only on Certain Files
```yaml
on:
  push:
    paths:
      - 'ingestion/**'                # Only if files in ingestion/ change
      - 'requirements.txt'
```

### Run Jobs in Sequence
```yaml
jobs:
  test:
    # ...
  deploy:
    needs: test                        # Only runs after test succeeds
    # ...
```

### Create Matrix for Multiple Versions
```yaml
strategy:
  matrix:
    python-version: ['3.9', '3.10', '3.11', '3.12']  # Runs 4 times
```

---

## Troubleshooting

### Workflow Won't Trigger
- Check branch name (only `main`/`develop`)
- Check if `.github/workflows/` path is correct
- Try manual trigger via GitHub UI

### Tests Fail on GitHub but Pass Locally
- Environment variable differences
- Python version mismatch
- File path issues (Windows vs Linux)

### Need API Keys for Tests
- Add secrets in GitHub Settings
- Reference with `${{ secrets.KEY_NAME }}`
- They're never exposed in logs

### View Detailed Logs
1. Go to **Actions** tab
2. Click the failed workflow run
3. Click the job name (e.g., "test")
4. Expand steps to see detailed output

---

## Best Practices

✅ **DO:**
- Run tests locally before pushing
- Keep workflows simple and fast
- Add meaningful job/step names
- Use caching for dependencies
- Store sensitive data as secrets

❌ **DON'T:**
- Hardcode API keys in workflows
- Use `continue-on-error: true` for critical tests
- Run expensive operations on every commit
- Leave workflows broken for long

---

## Next Steps

1. **Add secrets** in GitHub Settings (GROQ_API_KEY, GEMINI_API_KEY, PINECONE_API_KEY)
2. **Push code** with the `.github/workflows/` directory
3. **Watch** the Actions tab as workflows run
4. **Check** results and fix any failures

Your project now has professional CI/CD! 🚀
