# GitHub Actions Setup Checklist

Complete these steps to activate CI/CD for your Capstone RAG project.

## ✅ Step 1: Verify Local Tests Pass

```bash
cd c:\Users\aadya\OneDrive\Desktop\capstone-rag
python -m pytest tests/ -v
```

**Expected**: All 18 tests pass ✅

- [x] Unit tests pass (11 tests)
- [x] Integration tests pass (2 tests)
- [x] API tests pass (5 tests)

---

## ✅ Step 2: Review Created Files

All necessary files have been created:

- [x] `.github/workflows/test.yml` - Main testing workflow
- [x] `.github/workflows/lint.yml` - Code quality checks
- [x] `.github/workflows/scheduled-tests.yml` - Nightly tests
- [x] `.github/dependabot.yml` - Dependency updates
- [x] `.github/GITHUB_ACTIONS_GUIDE.md` - CI/CD documentation
- [x] `README.md` - Project overview
- [x] `TESTING.md` - Testing guide
- [x] `QUICKSTART.md` - Quick start guide
- [x] `IMPLEMENTATION_SUMMARY.md` - What was implemented

---

## ✅ Step 3: Create GitHub Repository

```bash
# Initialize git if not already done
git init
git add .
git commit -m "Initial commit with CI/CD setup"

# Create empty repo on GitHub (https://github.com/new)
# Then:
git remote add origin https://github.com/yourusername/capstone-rag.git
git branch -M main
git push -u origin main
```

**Or if repo already exists:**
```bash
git remote set-url origin https://github.com/yourusername/capstone-rag.git
git push -u origin main
```

---

## ⏳ Step 4: Add GitHub Secrets (Required for Full CI/CD)

1. Go to your GitHub repository
2. Click **Settings** (top right)
3. Click **Secrets and variables** → **Actions** (left sidebar)
4. Click **New repository secret** button
5. Add these three secrets:

### Secret 1: GROQ_API_KEY
- **Name**: `GROQ_API_KEY`
- **Value**: Get from https://console.groq.com
- Click "Add secret"

### Secret 2: GEMINI_API_KEY
- **Name**: `GEMINI_API_KEY`
- **Value**: Get from https://aistudio.google.com
- Click "Add secret"

### Secret 3: PINECONE_API_KEY
- **Name**: `PINECONE_API_KEY`
- **Value**: Get from https://app.pinecone.io
- Click "Add secret"

**Result**: You should see all 3 secrets listed

---

## ✅ Step 5: Verify Workflows Activated

1. Push code to GitHub:
```bash
git push origin main
```

2. Go to your repo on GitHub
3. Click **Actions** tab (top right area)
4. You should see workflow runs:
   - "Tests" - Running on multiple Python versions
   - "Lint & Code Quality" - Checking code style
   - "Scheduled Nightly Tests" - Scheduled (appears later)

4. Wait 2-5 minutes for workflows to complete
5. All should show ✅ green checkmarks

---

## 📋 What Each Workflow Does

### test.yml (On Every Push/PR)
- Runs tests on Python 3.9, 3.10, 3.11, 3.12
- Tests 3 suites: Unit → Integration → API
- Generates coverage report
- **Status**: 🟢 GREEN if all pass, 🔴 RED if any fail
- **Blocks PR merge** if fails

### lint.yml (On Every Push/PR)
- Checks code with Ruff, Black, isort, Flake8
- Ensures code quality
- **Status**: 🟢 GREEN if all pass, 🟡 YELLOW if warnings
- **Doesn't block** merge (unless critical errors)

### scheduled-tests.yml (Every Night at 2 AM UTC)
- Runs full test suite with coverage
- Tests with real API keys (from secrets)
- **Status**: Shows in Actions tab
- **Informational** (doesn't affect anything)

### dependabot.yml (Every Week)
- Checks for outdated packages
- Creates PRs with updates
- **Status**: PRs appear in repo
- **Optional** (you can accept/ignore)

---

## 🧪 Test Runs in GitHub

### What You'll See in Actions Tab

```
✅ Completed
   └─ test (Python 3.11)           → All tests passed
   └─ test (Python 3.10)           → All tests passed
   └─ test (Python 3.9)            → All tests passed
   └─ test (Python 3.12)           → All tests passed
   └─ Lint & Code Quality          → All checks passed
```

### What If Tests Fail?

1. Click the failed workflow run
2. Click the job (e.g., "test (Python 3.11)")
3. Expand steps to see error details
4. Fix code locally and push again
5. GitHub automatically re-runs

---

## 🚀 Making Changes Going Forward

### Typical Workflow

```bash
# 1. Make changes
# Edit files...

# 2. Test locally first
pytest tests/ -v

# 3. Commit and push
git add .
git commit -m "Your message"
git push origin main

# 4. GitHub Actions runs automatically
# Check Actions tab to see results

# 5. Create PR for code review (team projects)
# GitHub blocks merge if tests fail
```

### Before Creating Pull Request

```bash
# Always test locally first
pytest tests/ -v

# Check linting
ruff check .
black --check .

# Format if needed
black .
isort .

# Then commit and push
git add .
git commit -m "Feature: add something"
git push origin feature-branch

# Create PR on GitHub
# CI/CD runs automatically
# Team reviews before merging
```

---

## 📊 Monitoring Your CI/CD

### In GitHub

1. Go to **Actions** tab
2. See all workflow runs with status
3. Click any run to see details
4. View test output and coverage

### Set Up Notifications

1. Go to your GitHub **Settings**
2. Click **Notifications**
3. Choose how you're notified:
   - Email on workflow failures
   - Browser notifications
   - GitHub feed

---

## 🆘 Troubleshooting

### Workflows Don't Appear

**Problem**: Pushed code but no workflows running

**Solutions**:
- Wait 1-2 minutes (GitHub needs time)
- Refresh page (Ctrl+R or Cmd+R)
- Check `.github/workflows/` files exist in repo
- Check branch is `main` or `develop`
- Verify files have `.yml` extension

### Tests Fail on GitHub but Pass Locally

**Problem**: ✅ Local tests pass but 🔴 GitHub tests fail

**Reasons**:
- Python version difference (use 3.11 to match)
- Missing environment variables (add as secrets)
- Path issues (GitHub uses `/` not `\`)
- API key problems (check secrets added)

**Solution**:
1. Run: `python -c "import sys; print(sys.version)"`
2. Check tests pass on your Python version
3. Verify secrets added in Settings
4. Check for hardcoded paths

### Secrets Not Working

**Problem**: Tests fail saying API key missing

**Solution**:
1. Go to Settings → Secrets and variables → Actions
2. Verify all 3 secrets are listed:
   - GROQ_API_KEY
   - GEMINI_API_KEY
   - PINECONE_API_KEY
3. Click each to verify value is set
4. Re-run workflow (secrets cached)

### Lint Failures

**Problem**: 🔴 Lint workflow fails

**Solution**:
```bash
# Check locally
ruff check .
black --check .

# Fix automatically
black .
isort .

# Commit fixed code
git add .
git commit -m "Fix: format code"
git push origin main
```

---

## ✨ Next Level Setup (Optional)

### 1. Require Tests to Pass Before Merge

Go to repo Settings → Branches → Add rule:
- Branch name pattern: `main`
- Require status checks to pass
- Select your workflows
- Now PRs can't be merged until tests pass

### 2. Run Tests on Schedule

Already configured! Every night at 2 AM UTC:
- `scheduled-tests.yml` runs full test suite
- See results in Actions tab next morning

### 3. Deploy Automatically (Future)

You can add more workflows to:
- Deploy to Heroku, Vercel, AWS
- Send notifications to Slack/Discord
- Create releases on GitHub
- (See `.github/GITHUB_ACTIONS_GUIDE.md` for examples)

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| `QUICKSTART.md` | 5-minute setup |
| `README.md` | Full project overview |
| `TESTING.md` | How to run tests |
| `.github/GITHUB_ACTIONS_GUIDE.md` | CI/CD details |
| `IMPLEMENTATION_SUMMARY.md` | What was implemented |

**This file**: `GITHUB_ACTIONS_SETUP_CHECKLIST.md`

---

## ✅ Final Checklist

- [x] All 18 tests pass locally
- [ ] GitHub repo created
- [ ] Pushed code to `main` branch
- [ ] Added 3 secrets (GROQ, GEMINI, PINECONE)
- [ ] Workflows appear in Actions tab
- [ ] All workflows show green ✅
- [ ] Reviewed documentation (README.md, TESTING.md)
- [ ] Shared with team (if applicable)

---

## 🎉 You're Done!

Your Capstone RAG project now has professional CI/CD!

**GitHub Actions automatically:**
- ✅ Tests your code on every push
- ✅ Checks code quality
- ✅ Prevents merging broken code
- ✅ Runs nightly automated tests
- ✅ Updates dependencies weekly

**Your workflow is now:**
1. Code locally
2. Test locally (`pytest tests/ -v`)
3. Push to GitHub
4. GitHub Actions tests automatically
5. View results in Actions tab

---

## 🚀 Ready to Use!

```bash
# Next time you make changes:
pytest tests/ -v          # Test locally first
git push origin main      # Push to GitHub
# GitHub Actions runs automatically!
```

---

**Questions?** See:
- `TESTING.md` for testing help
- `.github/GITHUB_ACTIONS_GUIDE.md` for CI/CD help
- `README.md` for project overview
- Individual workflow YAML files for configuration

Happy coding! 🎉
