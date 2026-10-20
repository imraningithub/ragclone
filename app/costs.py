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
RANGES = ("today", "7d", "30d", "semester", "all")
STEPS = ("planner", "embeddings", "answer")
CHAT_STEPS = ("planner", "answer")  # chat-completion calls; only these can be served from the prompt cache
# Question outcomes: ANSWERED (stored as reason NULL), the refusals no_context / no_answer / uncited, and
# ABORTED (the client went away mid-answer) / ERROR (the model call failed): both still cost money.
ANSWERED, ABORTED, ERROR = "answered", "aborted", "error"


class Pricing:
    """USD per 1M tokens per model: the built-in table, optionally overridden or extended by a JSON file
    {"as_of": "...", "source": "...", "models": {"<model>": {"input": .., "cached_input": .., "output": ..}}}.
    A broken file never stops the app: the built-in prices stay in use and `override_error` says why."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.table: dict[str, tuple[float, float | None, float | None]] = dict(PRICES)
        self.as_of, self.source = PRICES_AS_OF, PRICES_SOURCE
        self.override_error: str | None = None
        if path is not None and Path(path).exists():
            try:
                data = json.loads(Path(path).read_text("utf-8"))
                models = {name: (float(p["input"]),
                                 None if p.get("cached_input") is None else float(p["cached_input"]),
                                 None if p.get("output") is None else float(p["output"]))
                          for name, p in (data.get("models") or {}).items()}
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                self.override_error = f"{Path(path).name} ignored: {e.__class__.__name__}: {e}"
            else:
                self.table.update(models)
                self.as_of = str(data.get("as_of") or self.as_of)
                self.source = str(data.get("source") or Path(path).name)

    def cost(self, model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float | None:
        """USD for one call, or None when the model is not in the price table."""
        price = self.table.get(model)
        if price is None:
            return None
        inp, cached_price, out = price
        cached = min(cached_tokens, input_tokens)
        usd = (input_tokens - cached) * inp + cached * (inp if cached_price is None else cached_price)
        usd += output_tokens * (out or 0.0)
        return usd / 1_000_000

    def describe(self) -> dict:
        return {"as_of": self.as_of, "source": self.source, "currency": "USD", "estimated": True,
                "override_error": self.override_error}


class UsageMeter:
    """Collects the paid calls of one question. One meter per request: never shared between requests."""

    def __init__(self, pricing: Pricing | None = None) -> None:
        self.pricing = pricing or Pricing()
        self.calls: list[dict] = []

    def step(self, step: str) -> Callable[..., None]:
        """A recorder for one pipeline step, handed to the provider as its `usage` callback."""

        def record(model: str, input_tokens: int = 0, cached_tokens: int = 0, output_tokens: int = 0,
                   estimated: bool = False) -> None:
            self.calls.append({"step": step, "model": model, "input": int(input_tokens or 0),
                               "cached": int(cached_tokens or 0), "output": int(output_tokens or 0),
                               "usd": self.pricing.cost(model, int(input_tokens or 0), int(cached_tokens or 0),
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


def range_start(name: str, now: float, semester_start: str | None = None) -> float | None:
    """Start (epoch seconds) of a dashboard range in Europe/Berlin calendar days; None for "all"."""
    if name == "all":
        return None
    if name == "semester":
        if not semester_start:
            raise ValueError("range 'semester' needs SEMESTER_START (e.g. 2026-10-01) in the settings")
        return datetime.fromisoformat(semester_start).replace(tzinfo=TZ).timestamp()
    days = {"today": 1, "7d": 7, "30d": 30}[name]
    midnight = datetime.fromtimestamp(now, TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    return (midnight - timedelta(days=days - 1)).timestamp()


def _berlin_day(ts: float) -> str:
    return datetime.fromtimestamp(ts, TZ).date().isoformat()


def _daily(calls: list, first_day: str, last_day: str) -> list[dict]:
    """Spend per step for every Europe/Berlin day from first_day to last_day, oldest first, zeros included."""
    days: dict[str, dict] = {}
    day = datetime.fromisoformat(first_day).date()
    end = datetime.fromisoformat(last_day).date()
    while day <= end:
        days[day.isoformat()] = {"date": day.isoformat(), **{s: 0 for s in STEPS}, "total": 0}
        day += timedelta(days=1)
    for c in calls:
        entry = days.get(_berlin_day(c["ts"]))
        if entry is not None and c["step"] in STEPS:
            entry[c["step"]] += c["usd"] or 0.0
            entry["total"] += c["usd"] or 0.0
    return list(days.values())


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]  # nearest-rank


def _group(calls: list, key: str) -> list[dict]:
    """Tokens and spend of the given calls per step or per model, most expensive first."""
    groups: dict[str, dict] = {}
    for c in calls:
        g = groups.setdefault(c[key], {key: c[key], "calls": 0, "input": 0, "cached": 0, "output": 0,
                                       "spend_usd": 0.0})
        g["calls"] += 1
        g["input"] += c["input"]
        g["cached"] += c["cached"]
        g["output"] += c["output"]
        g["spend_usd"] += c["usd"] or 0.0
    return sorted(groups.values(), key=lambda g: -g["spend_usd"])


class CostStore:
    def __init__(self, db_path: str, clock: Callable[[], float] = time.time,
                 semester_start: str | None = None, pricing: Pricing | None = None) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._semester_start = semester_start
        self.pricing = pricing or Pricing()  # reported on the dashboard; costs are priced when recorded
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
        now = self._clock()
        start = range_start(range_name, now, self._semester_start)
        where, args = ("WHERE ts >= ?", (start,)) if start is not None else ("", ())
        with self._lock:
            rows = self._conn.execute(f"SELECT ts, total_usd, unknown, estimated, reason FROM questions {where}",
                                      args).fetchall()
            calls = self._conn.execute(
                "SELECT q.ts, c.step, c.model, c.input, c.cached, c.output, c.usd FROM calls c "
                f"JOIN questions q ON q.id = c.question_id {where.replace('ts', 'q.ts')}", args).fetchall()
        first_day = _berlin_day(start if start is not None else min((r["ts"] for r in rows), default=now))
        chat = [c for c in calls if c["step"] in CHAT_STEPS]
        chat_input, chat_cached = sum(c["input"] for c in chat), sum(c["cached"] for c in chat)
        costs = [r["total_usd"] for r in rows]
        by_reason: dict[str, dict] = {}
        for r in rows:
            entry = by_reason.setdefault(r["reason"] or ANSWERED, {"reason": r["reason"] or ANSWERED,
                                                                   "questions": 0, "spend_usd": 0.0})
            entry["questions"] += 1
            entry["spend_usd"] += r["total_usd"]
        return {
            "range": range_name,
            "questions": len(costs),
            "spend_usd": sum(costs),
            "per_question": {"mean": statistics.fmean(costs) if costs else None,
                             "median": statistics.median(costs) if costs else None,
                             "p95": _p95(costs) if costs else None},
            "unknown_cost_questions": sum(r["unknown"] for r in rows),
            "estimated_questions": sum(r["estimated"] for r in rows),
            # answered first, then refusals and failures by count
            "by_reason": sorted(by_reason.values(), key=lambda e: (e["reason"] != ANSWERED, -e["questions"])),
            "by_step": _group(calls, "step"),
            "by_model": _group(calls, "model"),
            "daily": _daily(calls, first_day, _berlin_day(now)),
            "cache": {"input": chat_input, "cached": chat_cached,
                      "share": chat_cached / chat_input if chat_input else None},
            "pricing": self.pricing.describe(),
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
