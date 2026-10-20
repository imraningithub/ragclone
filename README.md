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

How an answer is produced: plan with `gpt-4.1-mini` (`KB_PLAN_MODEL`: standalone question, language,
German + English search queries) → hybrid retrieval (vectors + SQLite FTS5 keywords, reciprocal rank fusion,
max 4 chunks per document) → gate (no chunk above the similarity threshold → fixed refusal, no LLM call) →
grounded answer with `gpt-4.1` (`KB_CHAT_MODEL`), streamed → output check (`NO_ANSWER`/uncited → refusal) →
only cited sources are shown.
Streaming never shows ungrounded text: nothing is sent until the answer contains a valid `[n]` citation and no
`NO_ANSWER` marker; refusals are sent as the fixed refusal text only. If the final check changes what was already
shown (rare: a late `NO_ANSWER`, or a corrected closing line) the server sends a `reset` event with the final text.
Last eval: 26/28 answers correct (earlier runs: 27/28), 8/8 off-topic and 6/6 injection attempts refused,
100% cited. The two misses: the application deadline (not retrieved) and the total of 210 ECTS, whose SPO chunk
(§ 2 (4)) ranks outside the top 20 for "how many ECTS in total", so the model says the documents give no total.

Latency (24 answer questions, `scripts/bench_kb.py` against a local uvicorn, Oct 2026): median 2.7 s before
(planner on gpt-4.1, answer buffered) → first text after 2.5 s, complete after 2.8 s now. Per request the server
logs `timing` (`plan_s`, `retrieve_s`, `first_token_s`, `total_s`) in `data/kb_log.jsonl`; medians were
plan 1.2 s, retrieval 0.5 s, first visible answer text 0.8 s later. Answers are short and usually cite only in
their second half, so streaming mainly helps long answers.

### Hosting it for other students
Each browser gets an anonymous, random id in an httpOnly cookie (`dtm_client`, `SameSite=Lax`, `Secure` on HTTPS),
and sees only its own chats: other visitors' threads answer 404. No accounts; clearing cookies starts fresh.
Threads store only a hash of the id. Chats created before this change have no owner and are not shown to anyone.

Example `.env` on the server (no `kb.env` needed there):
```bash
PROVIDER=openai
OPENAI_API_KEY=sk-...            # use a separate key with a monthly spending limit
APP_MODE=kb
KB_CHAT_MODEL=gpt-4.1
KB_PLAN_MODEL=gpt-4.1-mini
RATE_LIMIT_PER_MIN=10            # questions per minute per browser (0 = off)
RATE_LIMIT_IP_PER_MIN=60         # per IP; campus Wi-Fi puts many students behind one IP, so keep it generous
ADMIN_TOKEN=<long random string> # needed for PUT /api/settings (open the UI with ?admin=1)
DATA_DIR=/srv/dtm/data           # rag.db, kb_threads.db, kb_log.jsonl; keep it on a persistent disk
```
Build the index on the server (`python scripts/ingest_kb.py`, needs the PDFs in `kb/dtm/`) or copy `data/rag.db`.
Run one worker behind an HTTPS reverse proxy and let uvicorn trust its `X-Forwarded-*` headers, otherwise every
student shares the proxy's IP for rate limiting and the cookie is not marked `Secure`:
```bash
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips="<proxy IP>"
```
(`--forwarded-allow-ips="*"` only if the app port is not reachable except through the proxy.) The rate limiters are
in memory, per process: with `--workers N` each worker counts separately. Estimated OpenAI cost per question
(tokens x prices in `app/costs.py`) is recorded in `data/kb_costs.db`; open `/admin/costs` with the admin token to
see cost per question and spend. Prices can be
overridden without a code change in `data/pricing.json` (format in CONTRACTS.md; applies to new questions). `kb_log.jsonl` records every question
and answer for quality review; tell students so, and delete it regularly.
