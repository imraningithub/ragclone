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

## DTM Studienassistent (knowledge-base mode)
Answers first-semester DTM questions **only** from the official documents in `kb/dtm/`, with page citations,
and refuses (pointing to the Study and Career Service / programme coordinator) when the answer isn't there.

```bash
# 1. put the PDFs into kb/dtm/ (current set: SPO de+en, ASPO, module handbook, study plans, exam plan,
#    timetable, flyer, presentation), then build the index (re-run after changes; unchanged files are skipped)
.venv/Scripts/python scripts/ingest_kb.py
# 2. run the assistant (kb.env sets APP_MODE=kb on top of .env)
.venv/Scripts/python -m uvicorn app.main:app --env-file kb.env --port 8001
# 3. evaluate (42 questions: answers, follow-ups, off-topic, prompt injection)
.venv/Scripts/python scripts/eval_kb.py
```

How an answer is produced: plan (standalone question, language, German + English search queries) →
hybrid retrieval (vectors + SQLite FTS5 keywords, reciprocal rank fusion, max 4 chunks per document) →
gate (no chunk above the similarity threshold → fixed refusal, no LLM call) → grounded answer with `gpt-4.1`
(`KB_CHAT_MODEL`) → output check (`NO_ANSWER`/uncited → refusal) → only cited sources are shown.
Last eval: 27/28 answers correct, 8/8 off-topic and 6/6 injection attempts refused, 100% cited.
