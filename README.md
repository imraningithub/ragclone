# Jan RAG clone

A Python + web clone of Jan's "chat with documents" feature (Jan 0.8.6 `@janhq/rag-extension` + `vector-db` plugin).
See `../jan-rag-clone-plan.md` for the reverse-engineering notes and `CONTRACTS.md` for module/API contracts.

## Setup
```bash
py -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
cp .env.example .env   # then put your OPENAI_API_KEY in .env
```

## Run
```bash
.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000
```
Open http://127.0.0.1:8000. Set `PROVIDER=fake` in `.env` to try the UI offline without an API key.

## Test
```bash
.venv/Scripts/python -m pytest -q
```

## How it maps to Jan
| Jan | Clone |
|---|---|
| `plugin:rag\|parse_document` | `app/parsing.py` |
| `plugin:vector-db\|chunk_text` | `app/chunking.py` |
| `plugin:vector-db\|*` (sqlite-vec + linear fallback) | `app/store.py` |
| `attachments_<threadId>` collections, auto inline vs embed | `app/rag.py` |
| Threads | `app/threads.py` |
| Settings → Attachments | `/api/settings`, settings modal |
