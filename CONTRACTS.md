# RAG clone — shared contracts

Every module is built against these interfaces. Do not change a signature without updating this file.
Python 3.14, venv at `ragclone/.venv`. Run tests with `.venv/Scripts/python -m pytest -q` from `ragclone/`.
The package is `app/` (import as `from app.x import y`). Tests live in `tests/`.

## Layout and owners
```
ragclone/
  app/config.py      Settings + RagSettings            (lead, done)
  app/providers.py   Provider protocol, OpenAI + Fake   (lead, done)
  app/parsing.py     parse_document                     (agent A)
  app/chunking.py    chunk_text                         (agent A)
  app/store.py       VectorStore (SQLite + sqlite-vec)  (agent B)
  app/rag.py         ingest + prompt building           (agent C)
  app/threads.py     ThreadStore (chat history)         (agent C)
  app/main.py        FastAPI app + routes               (agent C)
  web/index.html, web/app.js, web/styles.css            (agent D)
  tests/test_*.py    one file per module, owner writes it
```

## app/parsing.py
```python
SUPPORTED_EXTENSIONS: set[str]  # {".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx", ".json", ".html", ".htm"}
class UnsupportedFileType(ValueError): ...
class ParseError(RuntimeError): ...
def parse_document(path: str, file_type: str | None = None) -> str
```
- Picks the parser by extension (falling back to the MIME `file_type`). Returns plain UTF-8 text.
- PDF: `pypdf`, pages joined with `\n\n`. DOCX: paragraphs + table cells. CSV/XLSX: `pandas` → one line per row, `col: value; col: value`, one block per sheet with a `# Sheet: name` header. HTML: strip tags (stdlib `html.parser`). TXT/MD/JSON: read with utf-8, errors="replace".
- Empty extracted text is returned as `""` (not an error). Unreadable/corrupt input raises `ParseError`.

## app/chunking.py
```python
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]
```
- Character-based (Jan's `chunk_size_chars` / `overlap_chars`). Each chunk is at most `chunk_size` chars. Consecutive chunks overlap by `overlap` chars.
- Prefer to break at a paragraph break, then a sentence end, then whitespace, if one exists in the last ~20% of the window. Strip whitespace; drop empty chunks.
- `overlap >= chunk_size` → ValueError. Empty text → `[]`.

## app/store.py
```python
class DimensionMismatch(ValueError): ...
class DuplicateFile(ValueError): ...
class NotFound(KeyError): ...

class VectorStore:
    def __init__(self, db_path: str) -> None
    ann_available: bool                       # sqlite-vec loaded
    def create_collection(self, name: str, dim: int, embed_model: str) -> None   # idempotent; DimensionMismatch if existing dim/model differs
    def get_collection(self, name: str) -> dict | None      # {"name","dim","embed_model","created_at"}
    def delete_collection(self, name: str) -> None          # removes files, chunks, vec table
    def add_file(self, collection: str, *, name: str, path: str, type: str, size: int,
                 mode: str, inline_text: str | None = None) -> str   # returns file_id (uuid4 hex); DuplicateFile if same name+path in collection
    def insert_chunks(self, collection: str, file_id: str, texts: list[str], embeddings: list[list[float]]) -> int  # returns count; updates chunk_count
    def list_files(self, collection: str) -> list[dict]     # FileInfo, oldest first
    def get_file(self, collection: str, file_id: str) -> dict          # FileInfo + "inline_text"; NotFound
    def delete_file(self, collection: str, file_id: str) -> None       # and its chunks
    def get_chunks(self, collection: str, file_id: str, start: int = 0, end: int | None = None) -> list[dict]  # ChunkInfo by chunk_index
    def search(self, collection: str, query_embedding: list[float], limit: int = 3, threshold: float = 0.3,  # store-level defaults; the app passes RagSettings (5, 0.2)
               mode: str = "auto", file_ids: list[str] | None = None) -> list[dict]   # SearchHit, best first
    def status(self) -> dict                                # {"ann_available": bool, "collections": int}
    def close(self) -> None
```
- FileInfo: `{"id","collection","name","path","type","size","mode","chunk_count","created_at"}` where `mode` is `"embeddings"` or `"inline"`. Collection-less file is impossible: add_file requires an existing collection **unless** mode == "inline", in which case a collection row is created with dim=0 if missing. (Inline-only threads have no embedding dim yet.) If a dim-0 collection later gets `create_collection(name, dim>0, model)`, it is upgraded, not a mismatch.
- ChunkInfo: `{"id","file_id","chunk_index","text"}`.
- SearchHit: `{"chunk_id","file_id","file_name","chunk_index","text","score"}`; `score` = cosine similarity in [-1, 1]. Hits with score < threshold are dropped in every mode.
- `mode`: `"ann"` = sqlite-vec vec0 with `distance_metric=cosine` (raise RuntimeError if unavailable); `"linear"` = numpy cosine over stored BLOBs; `"auto"` = ann if available else linear. `file_ids` restricts the search.
- Embeddings stored as float32 BLOB. Validate: length == collection dim, all finite → else ValueError (Jan errors: "embedding is empty", "non-finite value").
- One SQLite file; thread-safe via `check_same_thread=False` + a `threading.Lock`. `created_at` = unix float.

## app/config.py and app/providers.py (done — read them)
- `get_settings() -> Settings` (env/.env). `RagSettings` holds Jan's defaults; `load_rag_settings(path)` / `save_rag_settings(path, rs)` persist UI overrides to `data/rag_settings.json`.
- `get_provider(settings) -> Provider` with `.embed(texts) -> list[list[float]]`, `.chat_stream(messages) -> Iterator[str]`, `.count_tokens(text) -> int`, `.name`, `.chat_model`, `.embed_model`.
- `PROVIDER=fake` in env selects `FakeProvider` (deterministic, offline) — **all tests use it**.

## app/rag.py
```python
INLINE_TEMPLATE, RETRIEVAL_TEMPLATE: str
def collection_for_thread(thread_id: str) -> str            # f"attachments_{thread_id}"
def decide_mode(text: str, parse_mode: str, provider, settings, rag: RagSettings) -> str  # "inline" | "embeddings"
def ingest_file(store, provider, settings, rag, *, collection: str, path: str, name: str,
                file_type: str, parse_mode: str | None = None) -> dict   # FileInfo
def build_messages(store, provider, settings, rag, *, collection: str,
                   history: list[dict], user_message: str) -> tuple[list[dict], list[dict]]  # (messages, citations)
```
- `decide_mode`: parse_mode "inline"/"embeddings" → as given; "auto" → inline when `count_tokens(text) <= auto_inline_context_ratio * CONTEXT_WINDOW`, else embeddings. ("prompt" is resolved by the UI before upload; treat as "auto" if it arrives.)
- Size limit: file larger than `max_file_size_mb` → ValueError.
- Embeddings ingest: parse → chunk_text(rag.chunk_size_chars, rag.overlap_chars) → provider.embed in batches of 64 → create_collection(dim, embed_model) → add_file → insert_chunks.
- build_messages: system prompt + (inline files' full text in INLINE_TEMPLATE) + (top-k hits for user_message in RETRIEVAL_TEMPLATE, numbered `[1]`, `[2]`… with file name) + history + user message. Citations: `[{"n","file_id","file_name","chunk_index","score","text"}]`.

## app/threads.py
```python
class ThreadStore:
    def __init__(self, db_path: str)
    def create(self, title: str = "New chat") -> dict          # {"id","title","created_at","updated_at"}
    def list(self) -> list[dict]                               # newest updated first
    def get(self, thread_id: str) -> dict                      # + "messages"; KeyError
    def rename(self, thread_id: str, title: str) -> dict
    def delete(self, thread_id: str) -> None
    def add_message(self, thread_id: str, role: str, content: str, citations: list | None = None) -> dict  # {"id","role","content","citations","created_at"}
```

## HTTP API (app/main.py) — the web UI uses exactly this
All JSON under `/api`. Errors: `{"detail": "..."}` with 400/404/409/413/422.
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/` | | `web/index.html`; `/static/*` serves `web/` |
| GET | `/api/status` | | `{"provider","chat_model","embed_model","ann_available","context_window"}` |
| GET | `/api/settings` | | RagSettings dict |
| PUT | `/api/settings` | partial RagSettings | full RagSettings |
| GET | `/api/threads` | | `[Thread]` |
| POST | `/api/threads` | `{"title"?}` | Thread |
| GET | `/api/threads/{id}` | | Thread + `messages` |
| PATCH | `/api/threads/{id}` | `{"title"}` | Thread |
| DELETE | `/api/threads/{id}` | | `{"ok":true}` (also deletes its collection) |
| GET | `/api/threads/{id}/files` | | `[FileInfo]` |
| POST | `/api/threads/{id}/files` | multipart `file`, optional form `parse_mode` | FileInfo (409 duplicate, 413 too large, 415 unsupported) |
| DELETE | `/api/threads/{id}/files/{file_id}` | | `{"ok":true}` |
| GET | `/api/threads/{id}/files/{file_id}/chunks` | | `[ChunkInfo]` |
| POST | `/api/threads/{id}/chat` | `{"message"}` | `text/event-stream` |

Chat SSE events (each `event: <name>\ndata: <json>\n\n`):
1. `citations` → `[Citation]` (may be `[]`), sent first.
2. `token` → `{"text": "..."}` repeated.
3. `done` → `{"message_id": "...", "title": "..."}` (title auto-set from the first user message if thread is still "New chat").
4. `error` → `{"detail": "..."}` then stream ends.
The user message is saved before streaming; the assistant message (with citations) is saved when done.
Uploaded files are saved under `data/uploads/<thread_id>/<file_id>_<name>` (the store `path` field).

---

# KB mode (DTM Studienassistent) — additional contracts

Settings (app/config.py, done): `app_mode` ("chat"|"kb"), `kb_name`, `kb_dir`, `kb_collection` (= "kb_<kb_name>"),
`refusal_contact`, `chat_temperature`, `admin_token`, `rate_limit_per_min` (0=off), `kb_chunk_size` (350),
`kb_overlap` (70), `kb_top_k` (6), `kb_threshold` (0.2), `kb_log_path`. All have defaults, so existing
`Settings(...)` constructions in tests keep working.

## app/parsing.py (agent A)
```python
def normalize_pdf_text(text: str) -> str      # rejoin "Prü-\nfung" -> "Prüfung" (lowercase continuation only),
                                              # join single wrapped lines inside paragraphs, keep blank-line paragraphs,
                                              # keep lines that look like list items / headings (start with -, •, digit+'.' , '§')
def parse_pdf_pages(path: str) -> list[tuple[int, str]]   # 1-based page numbers, normalized text, blank pages skipped
```
`parse_document` output for PDFs stays as today (no normalization) so existing behaviour/tests don't change.

## app/store.py (agent A)
- New nullable `meta TEXT` (JSON) column on `chunks`; added via `ALTER TABLE` when missing (old DBs keep working).
- `insert_chunks(collection, file_id, texts, embeddings, metas: list[dict] | None = None)`.
- SearchHit and ChunkInfo gain `"meta": dict` (`{}` when NULL). Everything else unchanged.

## app/rag.py (agent A)
```python
def ingest_kb_file(store, provider, settings, *, collection: str, path: str, name: str) -> dict  # FileInfo
```
- Always embeddings mode. PDFs: `parse_pdf_pages`, then `chunk_pages(pages, settings.kb_chunk_size, settings.kb_overlap)`
  (chunks each page with `chunk_text`, meta `{"page": n}`). A chunk that starts inside a legal section (`§ n Title`)
  gets that heading as its first line, so the title is embedded and keyword-indexed with it. Changing the chunking
  needs `scripts/ingest_kb.py --rebuild`.
  Other types: `parse_document` + chunk, meta `{}`. Embed in batches of 64, `create_collection`, `add_file`, `insert_chunks`.
- Empty text -> ValueError("no extractable text") (KB files must have text).

## scripts/ingest_kb.py (agent A)
`python scripts/ingest_kb.py [--rebuild] [--dir PATH]` — ingests every supported file in `settings.kb_dir` into
`settings.kb_collection`. Manifest `data/kb_manifest.json` maps file name -> sha256; unchanged files are skipped,
changed ones re-ingested (delete_file + ingest), files removed from the folder are deleted from the store.
Prints one line per file: name, pages, chunks, status (new/updated/unchanged/removed).

## app/providers.py (agent B)
```python
def chat_stream(self, messages, temperature: float | None = None) -> Iterator[str]
def complete(self, messages, *, json_mode: bool = False, temperature: float | None = None) -> str
```
OpenAI: `complete` uses `response_format={"type": "json_object"}` when json_mode. FakeProvider.complete: if json_mode,
return a JSON plan `{"standalone": <last user msg>, "language": "en" if it looks English else "de", "queries_de": [<last user msg>]}`;
otherwise return an answer that cites `[1]` when the system prompt contains `<excerpt`, else `NO_ANSWER`.

## app/retrieval.py (agent B)
```python
def plan_query(provider, history: list[dict], question: str) -> dict
    # {"standalone": str, "language": "de"|"en"|..., "queries_de": list[str] (1-3)}; falls back to
    # {"standalone": question, "language": guess_language(question), "queries_de": [question]} on any error/bad JSON
def guess_language(text: str) -> str                         # cheap heuristic "de"/"en"
def retrieve(store, provider, *, collection: str, plan: dict, question: str, top_k: int, threshold: float) -> list[dict]
    # embeds [question, standalone, *queries_de] (deduped) in one call, searches each (mode "auto"),
    # merges by chunk_id keeping max score, sorts desc, returns top_k SearchHits (with meta)
```

## app/guardrails.py (agent B)
```python
MAX_QUESTION_CHARS = 1000
NO_ANSWER = "NO_ANSWER"
def validate_question(text: str) -> str                      # strip; ValueError if empty or too long
class RateLimiter: __init__(per_minute: int); allow(key: str) -> bool   # per_minute <= 0 -> always True
def refusal_text(language: str, contact: str) -> str         # German for "de", English otherwise
SYSTEM_PROMPT: str                                           # strict grounding rules (see plan §4.3)
def build_kb_messages(*, question: str, plan: dict, hits: list[dict], history: list[dict]) -> tuple[list[dict], list[dict]]
    # excerpts as <excerpt id="n" doc="file" page="p">text</excerpt> inside the system message;
    # citations: [{"n","file_id","file_name","page","chunk_index","score","text"}]
def finalize_answer(text: str, citations: list[dict], *, language: str, contact: str) -> tuple[str, str | None]
    # returns (final_text, refusal_reason). reason: "no_answer" (model said NO_ANSWER / mostly NO_ANSWER),
    # "uncited" (no valid [n] with 1<=n<=len(citations)), None when accepted. Refusals return refusal_text(...).
def log_event(path, record: dict) -> None                    # append JSON line, never raises
```

## app/main.py in kb mode (agent C)
- `settings.app_mode == "kb"`:
  - `POST/DELETE /api/threads/{id}/files*` -> 403 {"detail": "Uploads are disabled in this assistant"}.
  - `PUT /api/settings` -> 403 unless header `X-Admin-Token` equals `settings.admin_token` (if no token set: 403).
  - `GET /kb/{name}` serves a file from `settings.kb_dir` (name must be a plain file name present in the folder; else 404).
  - `GET /api/status` adds `"mode"`, `"kb_name"`, `"refusal_contact"`, `"documents": [{"name","chunk_count"}]`.
    (In chat mode it adds `"mode": "chat"` only.)
  - `POST /api/threads/{id}/chat`: validate_question (400), RateLimiter by client IP (429), plan_query(history),
    retrieve(top_k=kb_top_k, threshold=kb_threshold). No hits -> refusal without calling the chat model.
    Else build_kb_messages -> provider.complete(temperature=chat_temperature) (buffered) -> finalize_answer.
    SSE: `citations` (`[]` if refused), `token` events (stream the final text in small word groups), `done`
    (`{"message_id","title","refused": reason|null}`). Save user + assistant messages (citations only if accepted).
    log_event(kb_log_path, {ts, thread_id, question, plan, hits:[{file,page,score}], reason, answer}).
- Chat mode behaviour unchanged.

## Web UI in kb mode (agent D)
Status `mode == "kb"`: hide attach button, drag-drop, file chips; title "DTM Studienassistent"; disclaimer under the
composer (DE/EN by `navigator.language`); empty state with 4 example question chips that send on click; citation chips
show `file · S. page` (EN: `p.`) and the expanded excerpt has a link `/kb/<file>#page=<page>` (target _blank);
settings gear hidden unless URL has `?admin=1`. `done.refused` -> style the message as a muted "info" bubble.

## KB mode — changes made during tuning (supersede the text above where they differ)
- `plan_query(provider, history, question, model=None)` also returns `queries_en`; the documents are mostly English.
- `retrieve(..., max_per_file=4)` is hybrid: vector + `VectorStore.keyword_search` (FTS5/BM25, weight 0.5), RRF fusion,
  cosine gate on the best similarity, programme name stripped from search queries. Hits carry `score` (cosine) and `rrf`.
- `VectorStore.keyword_search(collection, query, limit, file_ids=None)` and `score_chunks(collection, emb, ids)`;
  the `chunks_fts` index is kept in sync by triggers.
- `provider.complete(..., model=None)`; KB mode uses `settings.kb_chat_model` (default gpt-4.1) for plan + answer.
- `guardrails.cited_only(answer, citations)`: the API sends only cited sources. The binding line is forced into the
  answer language. KB mode stores chats in `data/kb_threads.db`. `parse_pdf_pages` uses pypdfium2 (pypdf fallback).

## KB mode — streaming answers and per-visitor threads (supersede the text above where they differ)
- Settings: `kb_plan_model` (`KB_PLAN_MODEL`, default `gpt-4.1-mini`) is used by `plan_query`; `kb_chat_model` only for
  the answer. `rate_limit_per_min` counts per visitor (owner id, IP when missing); `rate_limit_ip_per_min`
  (`RATE_LIMIT_IP_PER_MIN`, 0 = off) counts per client IP. Either one exceeded -> 429.
- `provider.chat_stream(messages, temperature=None, model=None)`. FakeProvider streams its KB answer when the system
  prompt contains `<excerpt`.
- `guardrails.StreamGate(n_citations, language)`: `feed(token) -> str`, `flush() -> str` return text that may be shown
  now; `raw` (all model output), `sent` (released, cleaned like `finalize_answer`), `stuck`. Releases nothing until a
  valid `[n]` and no NO_ANSWER marker are present; holds back a trailing open `[` and trailing whitespace.
- `POST /api/threads/{id}/chat` in kb mode: plan + retrieval + the first answer chunk happen before the response
  starts (provider errors there -> 502). SSE order: `token`* (released text; for refusals the fixed refusal text in
  word groups), optional `reset` `{"text": final}` (replaces everything shown so far), `citations` (cited sources,
  `[]` when refused), `done` `{"message_id","title","refused"}`. A provider error mid-stream -> `error` and nothing is
  saved. User + assistant messages are saved and the log line written only after the answer is complete; the log
  record adds `timing: {plan_s, retrieve_s, first_token_s, total_s}`.
- Visitors: kb mode middleware gives every request `request.state.owner` = sha256 of the `dtm_client` cookie
  (random `secrets.token_urlsafe(32)`, set when missing/malformed: httpOnly, SameSite=Lax, Secure on https, 1 year).
  `ThreadStore.create(title, owner=None)`, `list(owner=None)`, `get(id, owner=None)`; with an owner, other owners'
  threads raise KeyError (-> 404 for get/patch/delete/files/chat). `threads.owner` column is added to old DBs; their
  threads keep owner NULL. Chat mode passes no owner (unscoped, no cookie).

## KB mode — cost tracking (spec: imraningithub/ragclone#1)
- `app/costs.py`: `PRICES` (USD per 1M tokens: input, cached input, output; `PRICES_AS_OF`, `PRICES_SOURCE`, from
  `docs/research/openai-pricing.md`), `call_cost(model, input, cached, output) -> float | None` (None = model not in
  the table, never 0), `UsageMeter` (one per question; `step(name)` returns the recorder passed to a provider as
  `usage`; `calls`, `total_usd`, `unknown`), `CostStore(db_path, clock)` (`data/kb_costs.db`: `questions` and `calls`
  tables; `record_question(meter, visitor, reason, question, timing)`, `summary(range)` with range
  today/7d/30d/all in Europe/Berlin calendar days).
- Providers: `complete(..., usage=None)`, `chat_stream(..., usage=None)`, `embed(texts, usage=None)`; when given,
  `usage(model, input_tokens, cached_tokens, output_tokens)` is called once per API call (stream: after the last
  token). FakeProvider reports `plan_usage` / `answer_usage` / `embed_tokens` when set, else ~4 chars per token.
  `retrieval.plan_query(..., usage=None)` and `retrieve(..., usage=None)` pass it on only when given.
- `create_app(settings, provider, clock=time.time)`. KB chat records steps planner / embeddings / answer and writes
  one cost record when the answer is complete; the `kb_log.jsonl` line adds `usage` (the calls) and `cost_usd`.
- `GET /api/admin/costs?range=` (admin token, else 403; bad range 400) -> `{range, questions, spend_usd,
  per_question: {mean, median, p95 (nearest rank)}, unknown_cost_questions, pricing: {as_of, source, currency,
  estimated}}`. `GET /admin/costs` serves `web/admin.html` (no data; token in sessionStorage `dtm_admin_token`).
- Outcomes (#4): `costs.ANSWERED` (stored as reason NULL), `no_context` / `no_answer` / `uncited`, `costs.ABORTED`
  (stream closed before the answer completed) and `costs.ERROR` (a model call failed, before or during streaming;
  calls already billed are kept, a failed call is not). KB chat writes the record exactly once via one helper; the
  response streams through `main._closing`, which closes the sync generator when the response ends early. When
  the answer's usage never arrived, the answer call is estimated (~4 chars/token of prompt and generated text,
  `estimated: true`). Under uvicorn a disconnect is not noticed (the cookie middleware does not listen for it and
  uvicorn drops writes), so the answer completes and is recorded as answered with exact usage. 400/429 requests
  record nothing. Summary adds `estimated_questions` and `by_reason: [{reason, questions, spend_usd}]`
  (answered first).
- Breakdown (#5): settings `semester_start` (`SEMESTER_START`, ISO date, Europe/Berlin) enables range `semester`
  (400 when unset). `costs.Pricing(path)`: built-in `PRICES` merged with optional `data/pricing.json`
  `{"as_of", "source", "models": {"<model>": {"input", "cached_input"?, "output"?}}}` (USD per 1M tokens), read at
  startup; a broken file keeps the built-in prices and sets `override_error`. Costs are priced when recorded, so an
  override changes new questions only. `UsageMeter(pricing)`, `CostStore(..., semester_start, pricing)`. Summary
  adds `by_step` / `by_model` (`calls, input, cached, output, spend_usd`, most expensive first), `daily`
  (`[{date, planner, embeddings, answer, total}]` per Berlin day in the range, zeros included, oldest first),
  `cache` (`{input, cached, share}` over planner + answer calls; embeddings can't be cached) and
  `pricing.override_error`.
- Top lists and retention (#6): summary adds `top_questions` (max 10, most expensive first: `{id, asked_at`
  (Europe/Berlin ISO), `question` (None once purged), `reason, spend_usd, input, cached, output, steps: {step: usd}}`),
  `top_visitors` (max 10: `{visitor` = first 8 chars of the stored cookie hash, `questions, spend_usd}`) and
  `retention_days` (30). `CostStore.purge_question_text()` sets question text older than 30 days to NULL (rows,
  tokens and cost stay); it runs when the store opens (app start) and from `record_question` at most once a day.
- Test traffic (#7): a chat request with `X-Traffic: test` and a valid `X-Admin-Token` is stored with
  `questions.test = 1` (column added to older cost DBs at startup; old rows count as real traffic). The header
  without a valid token is ignored and never blocks the question. `GET /api/admin/costs?include_test=false`
  (default) excludes test questions from every figure; the response adds `include_test` and `test_questions`
  (test questions in the range, counted either way). The dashboard has an "Include test traffic" toggle
  (`?include_test=1` in the URL).
