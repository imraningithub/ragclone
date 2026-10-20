"""KB-mode cost tracking: per-question usage meter, price table, cost store and spend summaries.

Costs are estimates: reported token counts x the price table below (USD per 1M tokens). A model missing from
the table gets cost None ("unknown"), never 0.
"""
from __future__ import annotations

import json
import math
import sqlite3
import statistics
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

# Standard tier, docs/research/openai-pricing.md. (input, cached input, output) per 1M tokens.
PRICES: dict[str, tuple[float, float | None, float | None]] = {
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "text-embedding-3-small": (0.02, None, None),
}
PRICES_AS_OF = "2026-10-20"
PRICES_SOURCE = "https://developers.openai.com/api/docs/pricing"

TZ = ZoneInfo("Europe/Berlin")
RANGES = ("today", "7d", "30d", "all")


def call_cost(model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float | None:
    """USD for one call, or None when the model is not in the price table."""
    price = PRICES.get(model)
    if price is None:
        return None
    inp, cached_price, out = price
    cached = min(cached_tokens, input_tokens)
    usd = (input_tokens - cached) * inp + cached * (inp if cached_price is None else cached_price)
    usd += output_tokens * (out or 0.0)
    return usd / 1_000_000


class UsageMeter:
    """Collects the paid calls of one question. One meter per request: never shared between requests."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def step(self, step: str) -> Callable[..., None]:
        """A recorder for one pipeline step, handed to the provider as its `usage` callback."""

        def record(model: str, input_tokens: int = 0, cached_tokens: int = 0, output_tokens: int = 0,
                   estimated: bool = False) -> None:
            self.calls.append({"step": step, "model": model, "input": int(input_tokens or 0),
                               "cached": int(cached_tokens or 0), "output": int(output_tokens or 0),
                               "usd": call_cost(model, int(input_tokens or 0), int(cached_tokens or 0),
                                                int(output_tokens or 0)),
                               "estimated": bool(estimated)})

        return record

    @property
    def total_usd(self) -> float:
        return sum(c["usd"] for c in self.calls if c["usd"] is not None)

    @property
    def unknown(self) -> bool:
        return any(c["usd"] is None for c in self.calls)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    visitor TEXT,
    reason TEXT,
    question TEXT,
    timing TEXT NOT NULL DEFAULT '{}',
    total_usd REAL NOT NULL DEFAULT 0,
    estimated INTEGER NOT NULL DEFAULT 0,
    unknown INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_questions_ts ON questions(ts);
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id INTEGER REFERENCES questions(id) ON DELETE CASCADE,
    ts REAL NOT NULL,
    step TEXT NOT NULL,
    model TEXT NOT NULL,
    input INTEGER NOT NULL,
    cached INTEGER NOT NULL,
    output INTEGER NOT NULL,
    usd REAL,
    estimated INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_calls_question ON calls(question_id);
"""


def range_start(name: str, now: float) -> float | None:
    """Start (epoch seconds) of a dashboard range in Europe/Berlin calendar days; None for "all"."""
    if name == "all":
        return None
    days = {"today": 1, "7d": 7, "30d": 30}[name]
    midnight = datetime.fromtimestamp(now, TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    return (midnight - timedelta(days=days - 1)).timestamp()


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]  # nearest-rank


class CostStore:
    def __init__(self, db_path: str, clock: Callable[[], float] = time.time) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def record_question(self, meter: UsageMeter, *, visitor: str | None, reason: str | None,
                        question: str, timing: dict | None = None) -> int:
        now = self._clock()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO questions (ts, visitor, reason, question, timing, total_usd, estimated, unknown) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (now, visitor, reason, question, json.dumps(timing or {}), meter.total_usd,
                 int(any(c["estimated"] for c in meter.calls)), int(meter.unknown)))
            qid = cur.lastrowid
            self._conn.executemany(
                "INSERT INTO calls (question_id, ts, step, model, input, cached, output, usd, estimated) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(qid, now, c["step"], c["model"], c["input"], c["cached"], c["output"], c["usd"],
                  int(c["estimated"])) for c in meter.calls])
            self._conn.commit()
        return qid

    def summary(self, range_name: str) -> dict:
        if range_name not in RANGES:
            raise ValueError(f"range must be one of {', '.join(RANGES)}")
        start = range_start(range_name, self._clock())
        where, args = ("WHERE ts >= ?", (start,)) if start is not None else ("", ())
        with self._lock:
            rows = self._conn.execute(f"SELECT total_usd, unknown FROM questions {where}", args).fetchall()
        costs = [r["total_usd"] for r in rows]
        return {
            "range": range_name,
            "questions": len(costs),
            "spend_usd": sum(costs),
            "per_question": {"mean": statistics.fmean(costs) if costs else None,
                             "median": statistics.median(costs) if costs else None,
                             "p95": _p95(costs) if costs else None},
            "unknown_cost_questions": sum(r["unknown"] for r in rows),
            "pricing": {"as_of": PRICES_AS_OF, "source": PRICES_SOURCE, "currency": "USD", "estimated": True},
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
