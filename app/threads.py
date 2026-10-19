"""Chat threads and messages persisted in SQLite."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    owner TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    citations TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    seq INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, seq);
"""


def _now() -> float:
    return time.time()


class ThreadStore:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.executescript(_SCHEMA)
            # Databases from before per-visitor threads: add the column; their threads keep owner NULL.
            cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(threads)")}
            if "owner" not in cols:
                self._conn.execute("ALTER TABLE threads ADD COLUMN owner TEXT")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_threads_owner ON threads(owner, updated_at)")
            self._conn.commit()

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _thread(row: sqlite3.Row) -> dict:
        return {"id": row["id"], "title": row["title"],
                "created_at": row["created_at"], "updated_at": row["updated_at"]}

    @staticmethod
    def _message(row: sqlite3.Row) -> dict:
        try:
            citations = json.loads(row["citations"] or "[]")
        except json.JSONDecodeError:
            citations = []
        return {"id": row["id"], "role": row["role"], "content": row["content"],
                "citations": citations, "created_at": row["created_at"]}

    def _get_thread_row(self, thread_id: str, owner: str | None = None) -> sqlite3.Row:
        row = self._conn.execute("SELECT * FROM threads WHERE id = ?", (thread_id,)).fetchone()
        # Someone else's thread looks exactly like a missing one.
        if row is None or (owner is not None and row["owner"] != owner):
            raise KeyError(f"thread not found: {thread_id}")
        return row

    # -- API ---------------------------------------------------------------
    # `owner` (an anonymous visitor id) scopes threads to one browser in hosted KB mode; None = no scoping.
    def create(self, title: str = "New chat", owner: str | None = None) -> dict:
        tid = uuid.uuid4().hex
        now = _now()
        title = (title or "").strip() or "New chat"
        with self._lock:
            self._conn.execute("INSERT INTO threads (id, title, owner, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                               (tid, title, owner, now, now))
            self._conn.commit()
        return {"id": tid, "title": title, "created_at": now, "updated_at": now}

    def list(self, owner: str | None = None) -> list[dict]:
        with self._lock:
            if owner is None:
                rows = self._conn.execute(
                    "SELECT * FROM threads ORDER BY updated_at DESC, created_at DESC").fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM threads WHERE owner = ? ORDER BY updated_at DESC, "
                                          "created_at DESC", (owner,)).fetchall()
        return [self._thread(r) for r in rows]

    def get(self, thread_id: str, owner: str | None = None) -> dict:
        with self._lock:
            t = self._thread(self._get_thread_row(thread_id, owner))
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE thread_id = ? ORDER BY seq", (thread_id,)).fetchall()
        t["messages"] = [self._message(r) for r in rows]
        return t

    def rename(self, thread_id: str, title: str) -> dict:
        title = (title or "").strip()
        if not title:
            raise ValueError("title must not be empty")
        with self._lock:
            self._get_thread_row(thread_id)
            self._conn.execute("UPDATE threads SET title = ?, updated_at = ? WHERE id = ?",
                               (title, _now(), thread_id))
            self._conn.commit()
            return self._thread(self._get_thread_row(thread_id))

    def delete(self, thread_id: str) -> None:
        with self._lock:
            self._get_thread_row(thread_id)
            self._conn.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
            self._conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
            self._conn.commit()

    def add_message(self, thread_id: str, role: str, content: str, citations: list | None = None) -> dict:
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"invalid role: {role}")
        mid = uuid.uuid4().hex
        now = _now()
        cits = list(citations or [])
        with self._lock:
            self._get_thread_row(thread_id)
            seq = self._conn.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE thread_id = ?",
                                     (thread_id,)).fetchone()[0]
            self._conn.execute(
                "INSERT INTO messages (id, thread_id, role, content, citations, created_at, seq) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (mid, thread_id, role, content, json.dumps(cits), now, seq))
            self._conn.execute("UPDATE threads SET updated_at = ? WHERE id = ?", (now, thread_id))
            self._conn.commit()
        return {"id": mid, "role": role, "content": content, "citations": cits, "created_at": now}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
