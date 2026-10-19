"""Document ingest (inline vs embeddings) and prompt building with citations."""
from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from .chunking import chunk_text
from .config import RagSettings, Settings
from .parsing import parse_document, parse_pdf_pages

EMBED_BATCH = 64

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer using the attached documents when relevant. "
    "When you use a numbered excerpt, cite it as [n] with its number; when you use a full document, "
    "mention its name instead and never invent [n] numbers. "
    "If the documents don't contain the answer, or only part of it, say which part is missing "
    "instead of guessing numbers or facts."
)

# Whole-document context (files small enough to fit the context window).
INLINE_DOC_TEMPLATE = '<document name="{name}">\n{text}\n</document>'
INLINE_TEMPLATE = (
    "The user attached the following documents. Their full text is included below.\n\n"
    "{documents}"
)

# Retrieved excerpts (adapted from Jan's legacy retrieval template).
RETRIEVAL_ITEM_TEMPLATE = "[{n}] ({file_name}, chunk {chunk_index})\n{text}"
RETRIEVAL_TEMPLATE = (
    "Use the following pieces of context from the attached documents to answer the user's question. "
    "Cite them as [n].\n"
    "----------------\n"
    "CONTEXT:\n"
    "{context}\n"
    "----------------"
)


class FileTooLarge(ValueError):
    """Raised when an upload exceeds rag.max_file_size_mb."""


def collection_for_thread(thread_id: str) -> str:
    return f"attachments_{thread_id}"


def decide_mode(text: str, parse_mode: str, provider, settings: Settings, rag: RagSettings) -> str:
    if parse_mode in ("inline", "embeddings"):
        return parse_mode
    # "auto" (and "prompt", which the UI should have resolved already)
    budget = float(rag.auto_inline_context_ratio) * int(settings.context_window)
    return "inline" if provider.count_tokens(text) <= budget else "embeddings"


def _file_info(store, collection: str, file_id: str) -> dict:
    info = dict(store.get_file(collection, file_id))
    info.pop("inline_text", None)
    return info


def ingest_file(store, provider, settings: Settings, rag: RagSettings, *, collection: str, path: str,
                name: str, file_type: str, parse_mode: str | None = None) -> dict:
    size = os.path.getsize(path)
    limit = int(rag.max_file_size_mb) * 1024 * 1024
    if size > limit:
        raise FileTooLarge(f"File is {size / 1048576:.1f} MB; the limit is {rag.max_file_size_mb} MB")

    text = parse_document(path, file_type)
    mode = decide_mode(text, parse_mode or rag.parse_mode, provider, settings, rag)

    if mode == "inline":
        fid = store.add_file(collection, name=name, path=path, type=file_type, size=size,
                             mode="inline", inline_text=text)
        return _file_info(store, collection, fid)

    chunks = chunk_text(text, int(rag.chunk_size_chars), int(rag.overlap_chars))
    if not chunks:
        if store.get_collection(collection) is None:
            # No embedding dim known yet and nothing to embed: keep it inline (empty).
            fid = store.add_file(collection, name=name, path=path, type=file_type, size=size,
                                 mode="inline", inline_text="")
        else:
            fid = store.add_file(collection, name=name, path=path, type=file_type, size=size,
                                 mode="embeddings")
        return _file_info(store, collection, fid)

    embeddings = _embed_all(provider, chunks)

    store.create_collection(collection, len(embeddings[0]), provider.embed_model)
    fid = store.add_file(collection, name=name, path=path, type=file_type, size=size, mode="embeddings")
    try:
        store.insert_chunks(collection, fid, chunks, embeddings)
    except Exception:
        store.delete_file(collection, fid)
        raise
    return _file_info(store, collection, fid)


def _embed_all(provider, texts: list[str]) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        embeddings.extend(provider.embed(texts[i:i + EMBED_BATCH]))
    if len(embeddings) != len(texts) or not embeddings or not embeddings[0]:
        raise ValueError("embedding is empty")
    return embeddings


def ingest_kb_file(store, provider, settings: Settings, *, collection: str, path: str, name: str) -> dict:
    """Ingest one knowledge-base file, always in embeddings mode.

    PDFs are chunked page by page so every chunk knows its page ({"page": n} in its meta) and
    citations can link to it. Other types are chunked as a whole with an empty meta.
    """
    size = os.path.getsize(path)
    file_type = mimetypes.guess_type(name)[0] or ""
    chunks: list[str] = []
    metas: list[dict] = []
    if Path(path).suffix.lower() == ".pdf":
        file_type = "application/pdf"
        for page, text in parse_pdf_pages(path):
            for c in chunk_text(text, int(settings.kb_chunk_size), int(settings.kb_overlap)):
                chunks.append(c)
                metas.append({"page": page})
    else:
        text = parse_document(path, file_type or None)
        chunks = chunk_text(text, int(settings.kb_chunk_size), int(settings.kb_overlap))
        metas = [{} for _ in chunks]
    if not chunks:
        raise ValueError("no extractable text")  # e.g. a scanned PDF without OCR

    embeddings = _embed_all(provider, chunks)
    store.create_collection(collection, len(embeddings[0]), provider.embed_model)
    fid = store.add_file(collection, name=name, path=path, type=file_type, size=size, mode="embeddings")
    try:
        store.insert_chunks(collection, fid, chunks, embeddings, metas)
    except Exception:
        store.delete_file(collection, fid)
        raise
    return _file_info(store, collection, fid)


def _escape_attr(s: str) -> str:
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def build_messages(store, provider, settings: Settings, rag: RagSettings, *, collection: str,
                   history: list[dict], user_message: str) -> tuple[list[dict], list[dict]]:
    system_parts = [SYSTEM_PROMPT]
    citations: list[dict] = []

    coll = store.get_collection(collection) if rag.enabled else None
    if coll is not None:
        files = store.list_files(collection)

        docs = []
        for f in files:
            if f.get("mode") != "inline":
                continue
            full = store.get_file(collection, f["id"])
            text = full.get("inline_text") or ""
            if text.strip():
                docs.append(INLINE_DOC_TEMPLATE.format(name=_escape_attr(f["name"]), text=text))
        if docs:
            system_parts.append(INLINE_TEMPLATE.format(documents="\n\n".join(docs)))

        embed_ids = [f["id"] for f in files if f.get("mode") == "embeddings"]
        if int(coll.get("dim") or 0) > 0 and embed_ids and user_message.strip():
            q = provider.embed([user_message])[0]
            hits = store.search(collection, q, limit=int(rag.retrieval_limit),
                                threshold=float(rag.retrieval_threshold), mode=rag.search_mode,
                                file_ids=embed_ids)
            items = []
            for n, h in enumerate(hits, start=1):
                citations.append({"n": n, "file_id": h["file_id"], "file_name": h["file_name"],
                                  "chunk_index": h["chunk_index"], "score": float(h["score"]),
                                  "text": h["text"]})
                items.append(RETRIEVAL_ITEM_TEMPLATE.format(n=n, file_name=h["file_name"],
                                                            chunk_index=h["chunk_index"], text=h["text"]))
            if items:
                system_parts.append(RETRIEVAL_TEMPLATE.format(context="\n\n".join(items)))

    messages = [{"role": "system", "content": "\n\n".join(system_parts)}]
    for m in history:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})
    return messages, citations
