"""SQLite-backed vector store mirroring Jan's documents RAG storage.

Layout (one SQLite file, many collections):
  collections(name PK, dim, embed_model, created_at)
  files(id PK, collection, name, path, type, size, mode, chunk_count, inline_text, created_at)
  chunks(rowid PK, id UNIQUE, collection, file_id, chunk_index, text, embedding BLOB)
  vec_<sha1(collection)[:16]>  -- sqlite-vec vec0 table per collection, rowid = chunks.rowid

When sqlite-vec cannot be loaded (or enable_ann=False) search falls back to a
linear numpy cosine scan over the stored float32 BLOBs, like Jan does.
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
import threading
import time
import uuid
from typing import Any

import numpy as np

try:  # pragma: no cover - import guard
    import sqlite_vec  # type: ignore
except Exception:  # pragma: no cover
    sqlite_vec = None  # type: ignore


class DimensionMismatch(ValueError):
    pass


class DuplicateFile(ValueError):
    pass


class NotFound(KeyError):
    pass


_MODES = {"embeddings", "inline"}
_SEARCH_MODES = {"auto", "ann", "linear"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    name TEXT PRIMARY KEY,
    dim INTEGER NOT NULL,
    embed_model TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    collection TEXT NOT NULL,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT '',
    size INTEGER NOT NULL DEFAULT 0,
    mode TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    inline_text TEXT,
    created_at REAL NOT NULL,
    UNIQUE(collection, name, path)
);
CREATE INDEX IF NOT EXISTS idx_files_collection ON files(collection);
CREATE TABLE IF NOT EXISTS chunks (
    rowid INTEGER PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    collection TEXT NOT NULL,
    file_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks(collection);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id, chunk_index);
"""

_FILE_COLS = "id, collection, name, path, type, size, mode, chunk_count, created_at"


def _vec_table(collection: str) -> str:
    return "vec_" + hashlib.sha1(collection.encode("utf-8")).hexdigest()[:16]


def _validate_embedding(vec: Any, dim: int) -> np.ndarray:
    if vec is None:
        raise ValueError("embedding is empty")
    arr = np.asarray(vec, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("embedding must be a flat list of floats")
    if arr.size == 0:
        raise ValueError("embedding is empty")
    if arr.size != dim:
        raise ValueError(f"embedding has length {arr.size}, expected {dim}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("embedding contains a non-finite value")
    return arr.astype(np.float32)


class VectorStore:
    def __init__(self, db_path: str, enable_ann: bool = True) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.ann_available = False
        if enable_ann and sqlite_vec is not None:
            try:
                self._conn.enable_load_extension(True)
                sqlite_vec.load(self._conn)
                self._conn.enable_load_extension(False)
                self._conn.execute("SELECT vec_version()").fetchone()
                self.ann_available = True
            except Exception:
                self.ann_available = False
        with self._lock:
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.Error:
                pass
            self._conn.executescript(_SCHEMA)
            if self.ann_available:
                for row in self._conn.execute(
                    "SELECT name, dim FROM collections WHERE dim > 0"
                ).fetchall():
                    self._ensure_vec(row["name"], row["dim"])
            self._conn.commit()

    # ------------------------------------------------------------------ helpers
    def _coll(self, name: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT name, dim, embed_model, created_at FROM collections WHERE name = ?",
            (name,),
        ).fetchone()

    def _vec_exists(self, collection: str) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = ?", (_vec_table(collection),)
            ).fetchone()
            is not None
        )

    def _ensure_vec(self, collection: str, dim: int) -> None:
        """Create the vec0 table if needed and backfill it from chunks if out of sync."""
        if not self.ann_available or dim <= 0:
            return
        table = _vec_table(collection)
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0("
            f"embedding float[{int(dim)}] distance_metric=cosine)"
        )
        n_chunks = self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE collection = ?", (collection,)
        ).fetchone()[0]
        n_vec = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if n_chunks != n_vec:
            self._conn.execute(f"DELETE FROM {table}")
            rows = self._conn.execute(
                "SELECT rowid, embedding FROM chunks WHERE collection = ?", (collection,)
            ).fetchall()
            self._conn.executemany(
                f"INSERT INTO {table}(rowid, embedding) VALUES (?, ?)",
                [(r[0], r[1]) for r in rows],
            )

    def _drop_vec_rows(self, collection: str, rowids: list[int]) -> None:
        if not rowids or not self.ann_available or not self._vec_exists(collection):
            return
        table = _vec_table(collection)
        self._conn.executemany(
            f"DELETE FROM {table} WHERE rowid = ?", [(r,) for r in rowids]
        )

    @staticmethod
    def _file_dict(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "collection": row["collection"],
            "name": row["name"],
            "path": row["path"],
            "type": row["type"],
            "size": row["size"],
            "mode": row["mode"],
            "chunk_count": row["chunk_count"],
            "created_at": row["created_at"],
        }

    # ------------------------------------------------------------- collections
    def create_collection(self, name: str, dim: int, embed_model: str) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("collection name is required")
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
            raise ValueError("dim must be a non-negative integer")
        embed_model = embed_model or ""
        with self._lock:
            row = self._coll(name)
            if row is None:
                self._conn.execute(
                    "INSERT INTO collections(name, dim, embed_model, created_at) VALUES (?,?,?,?)",
                    (name, dim, embed_model, time.time()),
                )
            elif row["dim"] == 0 and dim > 0:
                # Inline-only collection gaining embeddings: upgrade in place.
                self._conn.execute(
                    "UPDATE collections SET dim = ?, embed_model = ? WHERE name = ?",
                    (dim, embed_model, name),
                )
            elif dim == 0:
                pass  # asking for an inline-only collection that already exists
            elif row["dim"] != dim or row["embed_model"] != embed_model:
                raise DimensionMismatch(
                    f"collection {name!r} exists with dim={row['dim']} model={row['embed_model']!r}; "
                    f"got dim={dim} model={embed_model!r}"
                )
            if dim > 0:
                self._ensure_vec(name, dim)
            self._conn.commit()

    def get_collection(self, name: str) -> dict | None:
        with self._lock:
            row = self._coll(name)
        return dict(row) if row is not None else None

    def delete_collection(self, name: str) -> None:
        with self._lock:
            self._conn.execute(f"DROP TABLE IF EXISTS {_vec_table(name)}")
            self._conn.execute("DELETE FROM chunks WHERE collection = ?", (name,))
            self._conn.execute("DELETE FROM files WHERE collection = ?", (name,))
            self._conn.execute("DELETE FROM collections WHERE name = ?", (name,))
            self._conn.commit()

    # ------------------------------------------------------------------- files
    def add_file(
        self,
        collection: str,
        *,
        name: str,
        path: str,
        type: str,
        size: int,
        mode: str,
        inline_text: str | None = None,
    ) -> str:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {sorted(_MODES)}")
        if not name:
            raise ValueError("file name is required")
        with self._lock:
            if self._coll(collection) is None:
                if mode != "inline":
                    raise NotFound(f"collection {collection!r} not found")
                self._conn.execute(
                    "INSERT INTO collections(name, dim, embed_model, created_at) VALUES (?,?,?,?)",
                    (collection, 0, "", time.time()),
                )
            dup = self._conn.execute(
                "SELECT id FROM files WHERE collection = ? AND name = ? AND path = ?",
                (collection, name, path),
            ).fetchone()
            if dup is not None:
                self._conn.rollback()
                raise DuplicateFile(f"file {name!r} already exists in collection")
            file_id = uuid.uuid4().hex
            self._conn.execute(
                "INSERT INTO files(id, collection, name, path, type, size, mode, chunk_count,"
                " inline_text, created_at) VALUES (?,?,?,?,?,?,?,0,?,?)",
                (file_id, collection, name, path, type or "", int(size or 0), mode,
                 inline_text, time.time()),
            )
            self._conn.commit()
            return file_id

    def insert_chunks(
        self,
        collection: str,
        file_id: str,
        texts: list[str],
        embeddings: list[list[float]],
    ) -> int:
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have the same length")
        with self._lock:
            coll = self._coll(collection)
            if coll is None:
                raise NotFound(f"collection {collection!r} not found")
            frow = self._conn.execute(
                "SELECT id FROM files WHERE id = ? AND collection = ?", (file_id, collection)
            ).fetchone()
            if frow is None:
                raise NotFound(f"file {file_id!r} not found")
            if not texts:
                return 0
            dim = coll["dim"]
            if dim <= 0:
                raise ValueError(f"collection {collection!r} has no embedding dimension")
            blobs = [_validate_embedding(e, dim).tobytes() for e in embeddings]
            start = self._conn.execute(
                "SELECT COALESCE(MAX(chunk_index), -1) + 1 FROM chunks WHERE file_id = ?",
                (file_id,),
            ).fetchone()[0]
            use_vec = self.ann_available and self._vec_exists(collection)
            table = _vec_table(collection)
            try:
                for i, (text, blob) in enumerate(zip(texts, blobs)):
                    cur = self._conn.execute(
                        "INSERT INTO chunks(id, collection, file_id, chunk_index, text, embedding)"
                        " VALUES (?,?,?,?,?,?)",
                        (uuid.uuid4().hex, collection, file_id, start + i, str(text), blob),
                    )
                    if use_vec:
                        self._conn.execute(
                            f"INSERT INTO {table}(rowid, embedding) VALUES (?, ?)",
                            (cur.lastrowid, blob),
                        )
                self._conn.execute(
                    "UPDATE files SET chunk_count = chunk_count + ? WHERE id = ?",
                    (len(texts), file_id),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            return len(texts)

    def list_files(self, collection: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_FILE_COLS} FROM files WHERE collection = ? ORDER BY created_at, rowid",
                (collection,),
            ).fetchall()
        return [self._file_dict(r) for r in rows]

    def get_file(self, collection: str, file_id: str) -> dict:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_FILE_COLS}, inline_text FROM files WHERE id = ? AND collection = ?",
                (file_id, collection),
            ).fetchone()
        if row is None:
            raise NotFound(f"file {file_id!r} not found")
        d = self._file_dict(row)
        d["inline_text"] = row["inline_text"]
        return d

    def delete_file(self, collection: str, file_id: str) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM files WHERE id = ? AND collection = ?", (file_id, collection)
            ).fetchone()
            if row is None:
                raise NotFound(f"file {file_id!r} not found")
            rowids = [
                r[0]
                for r in self._conn.execute(
                    "SELECT rowid FROM chunks WHERE file_id = ?", (file_id,)
                ).fetchall()
            ]
            self._drop_vec_rows(collection, rowids)
            self._conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
            self._conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
            self._conn.commit()

    def get_chunks(
        self, collection: str, file_id: str, start: int = 0, end: int | None = None
    ) -> list[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM files WHERE id = ? AND collection = ?", (file_id, collection)
            ).fetchone()
            if row is None:
                raise NotFound(f"file {file_id!r} not found")
            sql = "SELECT id, file_id, chunk_index, text FROM chunks WHERE file_id = ? AND chunk_index >= ?"
            params: list[Any] = [file_id, int(start)]
            if end is not None:
                sql += " AND chunk_index < ?"
                params.append(int(end))
            sql += " ORDER BY chunk_index"
            rows = self._conn.execute(sql, params).fetchall()
        return [
            {"id": r["id"], "file_id": r["file_id"], "chunk_index": r["chunk_index"], "text": r["text"]}
            for r in rows
        ]

    # ------------------------------------------------------------------ search
    def search(
        self,
        collection: str,
        query_embedding: list[float],
        limit: int = 3,
        threshold: float = 0.3,
        mode: str = "auto",
        file_ids: list[str] | None = None,
    ) -> list[dict]:
        if mode not in _SEARCH_MODES:
            raise ValueError(f"mode must be one of {sorted(_SEARCH_MODES)}")
        if mode == "ann" and not self.ann_available:
            raise RuntimeError("ANN search unavailable: sqlite-vec is not loaded")
        with self._lock:
            coll = self._coll(collection)
            if coll is None or coll["dim"] <= 0:
                return []  # nothing embedded yet
            q = _validate_embedding(query_embedding, coll["dim"])
            if limit <= 0 or (file_ids is not None and len(file_ids) == 0):
                return []
            use_ann = mode == "ann" or (mode == "auto" and self.ann_available)
            if use_ann:
                self._ensure_vec(collection, coll["dim"])
                scored = self._search_ann(collection, q, limit, threshold, file_ids)
            else:
                scored = self._search_linear(collection, q, limit, threshold, file_ids)
            return self._hydrate(scored)

    def _search_linear(self, collection, q, limit, threshold, file_ids):
        sql = "SELECT rowid, embedding FROM chunks WHERE collection = ?"
        params: list[Any] = [collection]
        if file_ids is not None:
            sql += f" AND file_id IN ({','.join('?' * len(file_ids))})"
            params.extend(file_ids)
        rows = self._conn.execute(sql, params).fetchall()
        if not rows:
            return []
        mat = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows]).astype(np.float64)
        qd = q.astype(np.float64)
        norms = np.linalg.norm(mat, axis=1) * np.linalg.norm(qd)
        with np.errstate(divide="ignore", invalid="ignore"):
            sims = np.where(norms > 0, (mat @ qd) / np.where(norms > 0, norms, 1.0), 0.0)
        out = [(rows[i][0], float(s)) for i, s in enumerate(sims) if s >= threshold]
        out.sort(key=lambda t: t[1], reverse=True)
        return out[:limit]

    def _search_ann(self, collection, q, limit, threshold, file_ids):
        table = _vec_table(collection)
        total = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if total == 0:
            return []
        allowed: set[int] | None = None
        if file_ids is not None:
            allowed = {
                r[0]
                for r in self._conn.execute(
                    f"SELECT rowid FROM chunks WHERE collection = ? AND file_id IN "
                    f"({','.join('?' * len(file_ids))})",
                    [collection, *file_ids],
                ).fetchall()
            }
            if not allowed:
                return []
        blob = sqlite_vec.serialize_float32(q.tolist())
        k = min(total, limit if allowed is None else max(limit * 10, 50))
        while True:
            rows = self._conn.execute(
                f"SELECT rowid, distance FROM {table} WHERE embedding MATCH ? AND k = ?"
                " ORDER BY distance",
                (blob, k),
            ).fetchall()
            out = []
            below = False
            for rowid, dist in rows:
                if dist is None or not math.isfinite(dist):
                    continue
                score = 1.0 - float(dist)
                if score < threshold:
                    below = True  # results are ordered; everything after is worse
                    break
                if allowed is not None and rowid not in allowed:
                    continue
                out.append((rowid, score))
                if len(out) >= limit:
                    break
            if len(out) >= limit or below or k >= total:
                return out[:limit]
            k = min(total, k * 4)

    def _hydrate(self, scored: list[tuple[int, float]]) -> list[dict]:
        if not scored:
            return []
        ids = [s[0] for s in scored]
        rows = self._conn.execute(
            "SELECT c.rowid AS rid, c.id, c.file_id, c.chunk_index, c.text, f.name AS file_name "
            f"FROM chunks c JOIN files f ON f.id = c.file_id WHERE c.rowid IN ({','.join('?' * len(ids))})",
            ids,
        ).fetchall()
        by_rid = {r["rid"]: r for r in rows}
        hits = []
        for rid, score in scored:
            r = by_rid.get(rid)
            if r is None:
                continue
            hits.append(
                {
                    "chunk_id": r["id"],
                    "file_id": r["file_id"],
                    "file_name": r["file_name"],
                    "chunk_index": r["chunk_index"],
                    "text": r["text"],
                    "score": max(-1.0, min(1.0, score)),
                }
            )
        return hits

    # ------------------------------------------------------------------- misc
    def status(self) -> dict:
        with self._lock:
            n = self._conn.execute("SELECT COUNT(*) FROM collections").fetchone()[0]
        return {"ann_available": self.ann_available, "collections": n}

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
