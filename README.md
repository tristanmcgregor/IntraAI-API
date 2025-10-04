# IntraAI RAG Backend (FastAPI)

Minimal Retrieval-Augmented Generation backend:
- Ingest company files (PDF, DOCX, PPTX, HTML, TXT)
- Store embeddings in SQLite (OpenAI embeddings)
- Query with OpenAI chat (fallback to context-only if no key)

## One-command start
- Double-click `backend\start.bat` (or run `backend\start.ps1` in PowerShell)
- Optional: pass `-NoInstall` to skip reinstalling dependencies

## Notes
- Corrupt Office files: DOCX/PPTX must be valid; we attempt a fallback parser for DOCX. If a file still fails, open and re-save as the same format, or export to PDF.

## Quickstart (manual)

```bash
# From repo root
cd backend

# Create venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install deps
pip install -U pip
pip install -r requirements.txt

# Env options
# 1) Shell only (temporary):
$env:OPENAI_API_KEY = "sk-..."
# 2) Env file (auto-loaded):
Copy-Item env.example .env; notepad .env

# Prepare documents
mkdir .\documents

# Run API
uvicorn app.main:app --reload
```

## Test client
```bash
# From backend with venv activated
python scripts/test_client.py health
python scripts/test_client.py scan
python scripts/test_client.py ingest .\documents\yourfile.pdf
python scripts/test_client.py query "What is our PTO policy?" --top_k 5
```

## Endpoints
- GET `/health`
- POST `/ingest/files`
- POST `/ingest/scan`
- POST `/query`

Data is persisted to SQLite at `backend/storage/index.db`. Documents live in `backend/documents/` by default.

Notes:
- `.env` is git-ignored (see `backend/.gitignore`).
- In production, prefer real environment variables or a secrets manager. 