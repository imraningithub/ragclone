"""Latency benchmark for a running KB-mode server, measured like a browser sees it.

Sends the "answer" questions of kb/dtm_eval.json (one new thread each) and reports, per question and as
medians, the time until the first answer text arrives and until the stream is done.

Usage: .venv/Scripts/python scripts/bench_kb.py [--url http://127.0.0.1:8001] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx

EVAL_FILE = Path(__file__).resolve().parent.parent / "kb" / "dtm_eval.json"


def ask(client: httpx.Client, question: str) -> tuple[float, float, str | None]:
    tid = client.post("/api/threads", json={}).json()["id"]
    t0 = time.perf_counter()
    first = None
    refused = None
    event = None
    with client.stream("POST", f"/api/threads/{tid}/chat", json={"message": question}) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                if event in ("token", "reset") and first is None:
                    first = time.perf_counter() - t0
                elif event == "done":
                    refused = json.loads(line[6:]).get("refused")
    total = time.perf_counter() - t0
    return first if first is not None else total, total, refused


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    items = [i for i in json.loads(EVAL_FILE.read_text("utf-8"))["items"] if i["type"] == "answer"]
    items = items[: args.limit] if args.limit else items
    rows = []
    with httpx.Client(base_url=args.url, timeout=120) as client:
        for item in items:
            first, total, refused = ask(client, item["q"])
            rows.append((first, total, refused))
            print(f"{item['id']:<18} first {first:5.1f}s  total {total:5.1f}s  refused={refused}", flush=True)
    answered = [r for r in rows if not r[2]]
    print(f"\n{len(answered)}/{len(rows)} answered; medians over answered: "
          f"first text {statistics.median(r[0] for r in answered):.1f}s, "
          f"complete {statistics.median(r[1] for r in answered):.1f}s")


if __name__ == "__main__":
    main()
