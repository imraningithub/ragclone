"""FastAPI app: threads, attachments (Jan-style RAG), settings and streamed chat."""
from __future__ import annotations

import hashlib
import hmac
import itertools
import json
import re
import secrets
import time
import shutil
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import guardrails, rag, retrieval
from .config import ROOT, RagSettings, Settings, get_settings, load_rag_settings, save_rag_settings
from .parsing import SUPPORTED_EXTENSIONS, ParseError, UnsupportedFileType
from .providers import Provider, get_provider
from .store import DimensionMismatch, DuplicateFile, NotFound, VectorStore
from .threads import ThreadStore

WEB_DIR = ROOT / "web"
HISTORY_LIMIT = 20
KB_HISTORY_LIMIT = 6
TITLE_LENGTH = 40
TOKEN_GROUP_WORDS = 3
# done.refused reasons: "no_context" (retrieval found nothing above kb_threshold; the chat model is not
# called) plus guardrails.finalize_answer's "no_answer" / "uncited".
NO_CONTEXT = "no_context"
UPLOADS_DISABLED = "Uploads are disabled in this assistant"
# KB mode: anonymous per-browser id so visitors only see their own chats (no accounts).
CLIENT_COOKIE = "dtm_client"
CLIENT_COOKIE_MAX_AGE = 365 * 24 * 3600
_CLIENT_ID_RE = re.compile(r"[A-Za-z0-9_-]{32,64}")
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class AppState:
    def __init__(self, settings: Settings, provider: Provider | None) -> None:
        self.settings = settings
        self.store = VectorStore(settings.db_path)
        self.threads = ThreadStore(settings.threads_db_path)
        self.rag = load_rag_settings(settings.rag_settings_path)
        self._provider = provider
        self._lock = threading.Lock()

    @property
    def provider(self) -> Provider:
        # Created lazily so the app starts (and shows a clear error) without an API key.
        with self._lock:
            if self._provider is None:
                try:
                    self._provider = get_provider(self.settings)
                except RuntimeError as e:
                    raise HTTPException(503, str(e)) from e
            return self._provider


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _safe_name(name: str) -> str:
    name = Path(name or "file").name
    return re.sub(r"[^\w.\- ]+", "_", name).strip() or "file"


def _word_groups(text: str, size: int = TOKEN_GROUP_WORDS) -> list[str]:
    """Split text into small pieces (words keep their surrounding whitespace) for token events."""
    words = re.findall(r"\s*\S+\s*", text) or ([text] if text else [])
    return ["".join(words[i:i + size]) for i in range(0, len(words), size)]


def _make_title(message: str) -> str:
    return message if len(message) <= TITLE_LENGTH else message[:TITLE_LENGTH].rstrip() + "…"


def _provider_error(e: Exception) -> HTTPException:
    status = getattr(e, "status_code", None)
    return HTTPException(502, f"Model provider error{f' ({status})' if status else ''}: {e.__class__.__name__}")


def create_app(settings: Settings | None = None, provider: Provider | None = None) -> FastAPI:
    settings = settings or get_settings()
    state = AppState(settings, provider)
    kb_mode = settings.app_mode == "kb"
    limiter = guardrails.RateLimiter(settings.rate_limit_per_min)
    ip_limiter = guardrails.RateLimiter(settings.rate_limit_ip_per_min)
    app = FastAPI(title="DTM Studienassistent" if kb_mode else "Jan RAG clone")
    app.state.rag_app = state

    @app.exception_handler(NotFound)
    @app.exception_handler(KeyError)
    async def not_found(_: Request, e: KeyError):
        return JSONResponse({"detail": str(e.args[0]) if e.args else "not found"}, status_code=404)

    if kb_mode:
        @app.middleware("http")
        async def client_cookie(request: Request, call_next):
            cid = request.cookies.get(CLIENT_COOKIE) or ""
            fresh = not _CLIENT_ID_RE.fullmatch(cid)
            if fresh:
                cid = secrets.token_urlsafe(32)
            # Threads store a hash, so a leaked threads DB does not hand out working cookies.
            request.state.owner = hashlib.sha256(cid.encode()).hexdigest()
            response = await call_next(request)
            if fresh:
                response.set_cookie(CLIENT_COOKIE, cid, max_age=CLIENT_COOKIE_MAX_AGE, httponly=True,
                                    samesite="lax", secure=request.url.scheme == "https")
            return response

    def owner(request: Request) -> str | None:
        return getattr(request.state, "owner", None) if kb_mode else None

    def thread_or_404(thread_id: str, request: Request) -> dict:
        try:
            return state.threads.get(thread_id, owner=owner(request))
        except KeyError:
            raise HTTPException(404, "thread not found")

    # -- pages -----------------------------------------------------------
    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        page = WEB_DIR / "index.html"
        if page.exists():
            return FileResponse(page)
        return HTMLResponse("<h1>Jan RAG clone</h1><p>web/index.html is missing.</p>")

    # -- status + settings -----------------------------------------------
    @app.get("/api/status")
    def status():
        out = {"provider": settings.provider, "chat_model": settings.chat_model,
               "embed_model": settings.embed_model, "ann_available": state.store.ann_available,
               "context_window": settings.context_window, "mode": "kb" if kb_mode else "chat"}
        if kb_mode:
            out.update(kb_name=settings.kb_name, refusal_contact=settings.refusal_contact,
                       documents=[{"name": f["name"], "chunk_count": f["chunk_count"]}
                                  for f in state.store.list_files(settings.kb_collection)])
        return out

    @app.get("/api/settings")
    def get_rag_settings():
        return asdict(state.rag)

    @app.put("/api/settings")
    def put_rag_settings(request: Request, data: dict = Body(...)):
        if kb_mode:
            given = request.headers.get("X-Admin-Token") or ""
            if not settings.admin_token or not hmac.compare_digest(given.encode(), settings.admin_token.encode()):
                raise HTTPException(403, "Admin token required")
        try:
            state.rag = state.rag.update(data)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e))
        save_rag_settings(settings.rag_settings_path, state.rag)
        return asdict(state.rag)

    # -- threads -----------------------------------------------------------
    @app.get("/api/threads")
    def list_threads(request: Request):
        return state.threads.list(owner=owner(request))

    @app.post("/api/threads")
    def create_thread(request: Request, data: dict = Body(default={})):
        return state.threads.create((data or {}).get("title") or "New chat", owner=owner(request))

    @app.get("/api/threads/{thread_id}")
    def get_thread(thread_id: str, request: Request):
        return thread_or_404(thread_id, request)

    @app.patch("/api/threads/{thread_id}")
    def rename_thread(thread_id: str, request: Request, data: dict = Body(...)):
        thread_or_404(thread_id, request)
        try:
            return state.threads.rename(thread_id, str(data.get("title", "")))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.delete("/api/threads/{thread_id}")
    def delete_thread(thread_id: str, request: Request):
        thread_or_404(thread_id, request)
        state.store.delete_collection(rag.collection_for_thread(thread_id))
        shutil.rmtree(settings.uploads_dir / thread_id, ignore_errors=True)
        state.threads.delete(thread_id)
        return {"ok": True}

    # -- attachments -------------------------------------------------------
    @app.get("/api/threads/{thread_id}/files")
    def list_files(thread_id: str, request: Request):
        thread_or_404(thread_id, request)
        return state.store.list_files(rag.collection_for_thread(thread_id))

    @app.post("/api/threads/{thread_id}/files")
    def upload_file(thread_id: str, request: Request, file: UploadFile = File(...),
                    parse_mode: str | None = Form(None)):
        if kb_mode:
            raise HTTPException(403, UPLOADS_DISABLED)
        thread_or_404(thread_id, request)
        if not state.rag.enabled:
            raise HTTPException(400, "Attachments are disabled in settings")
        if parse_mode not in (None, "", "auto", "inline", "embeddings"):
            raise HTTPException(400, "parse_mode must be auto, inline or embeddings")
        name = _safe_name(file.filename or "file")
        if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise HTTPException(415, f"Unsupported file type: {Path(name).suffix or name}")

        folder = settings.uploads_dir / thread_id
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / f"{uuid.uuid4().hex[:8]}_{name}"
        limit = int(state.rag.max_file_size_mb) * 1024 * 1024
        written = 0
        with dest.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    break
                out.write(chunk)
        if written > limit:
            dest.unlink(missing_ok=True)
            raise HTTPException(413, f"File is larger than the {state.rag.max_file_size_mb} MB limit")

        collection = rag.collection_for_thread(thread_id)
        # Same file name in one thread counts as a duplicate (paths are unique per upload).
        if any(f["name"] == name for f in state.store.list_files(collection)):
            dest.unlink(missing_ok=True)
            raise HTTPException(409, f"File '{name}' has already been attached to this chat")
        try:
            return rag.ingest_file(state.store, state.provider, settings, state.rag, collection=collection,
                                   path=str(dest), name=name,
                                   file_type=file.content_type or "application/octet-stream",
                                   parse_mode=parse_mode or None)
        except HTTPException:
            dest.unlink(missing_ok=True)
            raise
        except Exception as e:
            dest.unlink(missing_ok=True)
            if isinstance(e, UnsupportedFileType):
                raise HTTPException(415, str(e))
            if isinstance(e, rag.FileTooLarge):
                raise HTTPException(413, str(e))
            if isinstance(e, DuplicateFile):
                raise HTTPException(409, str(e))
            if isinstance(e, ParseError):
                raise HTTPException(422, f"Could not read the file: {e}")
            if isinstance(e, DimensionMismatch):
                raise HTTPException(409, f"{e}. This chat was indexed with a different embedding model.")
            if isinstance(e, ValueError):
                raise HTTPException(400, str(e))
            raise _provider_error(e)

    @app.delete("/api/threads/{thread_id}/files/{file_id}")
    def delete_file(thread_id: str, file_id: str, request: Request):
        if kb_mode:
            raise HTTPException(403, UPLOADS_DISABLED)
        thread_or_404(thread_id, request)
        collection = rag.collection_for_thread(thread_id)
        info = state.store.get_file(collection, file_id)
        state.store.delete_file(collection, file_id)
        path = Path(info.get("path") or "")
        if path.is_file() and settings.uploads_dir in path.parents:
            path.unlink(missing_ok=True)
        return {"ok": True}

    @app.get("/api/threads/{thread_id}/files/{file_id}/chunks")
    def file_chunks(thread_id: str, file_id: str, request: Request):
        thread_or_404(thread_id, request)
        return state.store.get_chunks(rag.collection_for_thread(thread_id), file_id)

    # -- knowledge-base documents ---------------------------------------------
    @app.get("/kb/{name}", include_in_schema=False)
    def kb_document(name: str):
        if (not kb_mode or not name or "/" in name or "\\" in name or ".." in name
                or name != Path(name).name):
            raise HTTPException(404, "document not found")
        folder = Path(settings.kb_dir).resolve()
        path = (folder / name).resolve()
        if path.parent != folder or not path.is_file():
            raise HTTPException(404, "document not found")
        media = "application/pdf" if path.suffix.lower() == ".pdf" else None
        # inline so the browser's PDF viewer opens it and honours #page=N
        return FileResponse(path, media_type=media, filename=path.name, content_disposition_type="inline")

    # -- chat ----------------------------------------------------------------
    def kb_chat(request: Request, thread_id: str, thread: dict, data: dict) -> StreamingResponse:
        try:
            question = guardrails.validate_question(str(data.get("message") or ""))
        except ValueError as e:
            raise HTTPException(400, str(e))
        client_ip = request.client.host if request.client and request.client.host else "local"
        if not ip_limiter.allow(client_ip) or not limiter.allow(owner(request) or client_ip):
            raise HTTPException(429, "Too many requests, please wait a minute")
        provider = state.provider
        history = [{"role": m["role"], "content": m["content"]}
                   for m in thread["messages"][-KB_HISTORY_LIMIT:]]
        t0 = time.perf_counter()
        timing: dict[str, float] = {}

        def mark(key: str) -> None:
            timing[key] = round(time.perf_counter() - t0, 3)

        try:
            plan = retrieval.plan_query(provider, history, question, model=settings.kb_plan_model)
            mark("plan_s")
            hits = retrieval.retrieve(state.store, provider, collection=settings.kb_collection, plan=plan,
                                      question=question, top_k=settings.kb_top_k,
                                      threshold=settings.kb_threshold)
            mark("retrieve_s")
        except Exception as e:
            raise _provider_error(e)
        language = plan.get("language") or retrieval.guess_language(question)
        tokens: Iterator[str] | None = None
        candidates: list[dict] = []
        if hits:
            messages, candidates = guardrails.build_kb_messages(question=question, plan=plan, hits=hits,
                                                                history=history)
            try:
                # Wait for the first chunk here so a provider failure is still a 502, not a broken stream.
                tokens = iter(provider.chat_stream(messages, temperature=settings.chat_temperature,
                                                   model=settings.kb_chat_model))
                tokens = itertools.chain([next(tokens, "")], tokens)
            except Exception as e:
                raise _provider_error(e)

        def stream() -> Iterator[str]:
            if tokens is None:
                # Nothing relevant in the knowledge base: refuse without calling the chat model.
                answer = guardrails.refusal_text(language, settings.refusal_contact)
                reason, citations = NO_CONTEXT, []
                mark("first_token_s")
                for piece in _word_groups(answer):
                    yield _sse("token", {"text": piece})
            else:
                gate = guardrails.StreamGate(len(candidates), language)
                try:
                    for token in tokens:
                        if out := gate.feed(token):
                            timing.setdefault("first_token_s", round(time.perf_counter() - t0, 3))
                            yield _sse("token", {"text": out})
                except Exception as e:
                    yield _sse("error", {"detail": _provider_error(e).detail})
                    return
                if out := gate.flush():
                    yield _sse("token", {"text": out})
                answer, reason = guardrails.finalize_answer(gate.raw, candidates, language=language,
                                                            contact=settings.refusal_contact)
                if not gate.sent:  # refusal (or an answer the gate could not release early)
                    mark("first_token_s")
                    for piece in _word_groups(answer):
                        yield _sse("token", {"text": piece})
                elif answer != gate.sent:
                    yield _sse("reset", {"text": answer})
                citations = [] if reason else guardrails.cited_only(answer, candidates)
            yield _sse("citations", citations)

            # Only reached when the answer is complete; a client disconnect closes the generator first.
            state.threads.add_message(thread_id, "user", question)
            title = thread["title"]
            if title == "New chat":
                title = _make_title(question)
                state.threads.rename(thread_id, title)
            saved = state.threads.add_message(thread_id, "assistant", answer, citations or None)
            mark("total_s")
            guardrails.log_event(settings.kb_log_path, {
                "ts": time.time(), "thread_id": thread_id, "question": question, "plan": plan,
                "hits": [{"file": h.get("file_name"), "page": (h.get("meta") or {}).get("page"),
                          "score": h.get("score")} for h in hits],
                "reason": reason, "answer": answer, "timing": timing})
            yield _sse("done", {"message_id": saved["id"], "title": title, "refused": reason})

        return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)

    @app.post("/api/threads/{thread_id}/chat")
    def chat(thread_id: str, request: Request, data: dict = Body(...)):
        thread = thread_or_404(thread_id, request)
        if kb_mode:
            return kb_chat(request, thread_id, thread, data)
        message = str(data.get("message") or "").strip()
        if not message:
            raise HTTPException(400, "message must not be empty")
        provider = state.provider
        history = thread["messages"][-HISTORY_LIMIT:]
        try:
            messages, citations = rag.build_messages(
                state.store, provider, settings, state.rag, collection=rag.collection_for_thread(thread_id),
                history=history, user_message=message)
        except Exception as e:
            raise _provider_error(e)

        state.threads.add_message(thread_id, "user", message)
        title = thread["title"]
        if title == "New chat":
            title = _make_title(message)
            state.threads.rename(thread_id, title)

        def stream() -> Iterator[str]:
            yield _sse("citations", citations)
            parts: list[str] = []
            try:
                for token in provider.chat_stream(messages):
                    parts.append(token)
                    yield _sse("token", {"text": token})
            except Exception as e:
                yield _sse("error", {"detail": _provider_error(e).detail})
                return
            # Only reached when the stream finished; a client disconnect closes the generator first.
            saved = state.threads.add_message(thread_id, "assistant", "".join(parts), citations)
            yield _sse("done", {"message_id": saved["id"], "title": title, "refused": None})

        return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)

    return app


app = create_app()
