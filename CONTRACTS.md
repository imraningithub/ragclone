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
