"""End-to-end eval of the DTM Studienassistent (KB mode) through the real HTTP API.

Copies the KB index (data/rag.db) into a temp dir, runs every item of kb/dtm_eval.json through
POST /api/threads/{id}/chat with the real provider, and reports retrieval hits (from the KB log),
refusal correctness, citation presence, answer correctness and language match.

Usage: .venv/Scripts/python scripts/eval_kb.py [--only ID_PREFIX] [--top-k N] [--threshold T] [--show]
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import tempfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.retrieval import guess_language  # noqa: E402

EVAL_FILE = ROOT / "kb" / "dtm_eval.json"


def sse_events(body: str) -> list[tuple[str, object]]:
    out = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            out.append((fields["event"], json.loads(fields.get("data", "null"))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--show", action="store_true", help="print every answer")
    args = ap.parse_args()

    base = get_settings()
    items = [i for i in json.loads(EVAL_FILE.read_text("utf-8"))["items"] if i["id"].startswith(args.only)]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # Windows keeps the DB open
        # Backup API, not a file copy: recent writes may still be in the WAL file next to rag.db.
        src, dst = sqlite3.connect(base.db_path), sqlite3.connect(Path(tmp) / "rag.db")
        src.backup(dst)
        src.close()
        dst.close()
        settings = replace(base, app_mode="kb", data_dir=Path(tmp), rate_limit_per_min=0,
                           kb_top_k=args.top_k or base.kb_top_k,
                           kb_threshold=base.kb_threshold if args.threshold is None else args.threshold)
        print(f"model={settings.kb_chat_model} embed={settings.embed_model} top_k={settings.kb_top_k} "
              f"threshold={settings.kb_threshold} items={len(items)}\n")
        results = []
        with TestClient(create_app(settings)) as client:
            for item in items:
                item = {**item, "_expected": item.get("answer", "")}
                tid = client.post("/api/threads", json={}).json()["id"]
                # Seed follow-up history through the thread store the app reads from.
                for m in item.get("history", []):
                    client.app.state.rag_app.threads.add_message(tid, m["role"], m["content"])
                r = client.post(f"/api/threads/{tid}/chat", json={"message": item["q"]})
                if r.status_code != 200:
                    results.append({**item, "status": r.status_code, "answer": r.text, "refused": "http"})
                    continue
                ev = sse_events(r.text)
                answer = "".join(d["text"] for e, d in ev if e == "token")
                done = next((d for e, d in ev if e == "done"), {}) or {}
                cites = next((d for e, d in ev if e == "citations"), []) or []
                log = [json.loads(line) for line in settings.kb_log_path.read_text("utf-8").splitlines()]
                hits = log[-1].get("hits", []) if log else []
                results.append({**item, "answer": answer, "refused": done.get("refused"), "cites": cites,
                                "hits": hits, "plan": log[-1].get("plan") if log else None})

    score = defaultdict(lambda: [0, 0])
    for res in results:
        t = res["type"]
        should_answer = t in ("answer", "followup")
        refused = bool(res["refused"])
        ok_refusal = refused != should_answer
        if not should_answer and not refused and res.get("allow_answer"):
            ok_refusal = bool(re.search(res["allow_answer"], res["answer"], re.I))  # a correct grounded answer is fine
        score[f"{t}: correct refuse/answer"][0] += ok_refusal
        score[f"{t}: correct refuse/answer"][1] += 1
        if should_answer:
            doc, pages = res.get("expect_doc", ""), set(res.get("expect_pages", []))
            hit = any(doc in h.get("file", "") and (not pages or h.get("page") in pages) for h in res["hits"])
            score["retrieval hit@K"][0] += hit
            score["retrieval hit@K"][1] += 1
            correct = (not refused) and bool(re.search(res["_expected"], res["answer"], re.I))
            score["answer correct"][0] += correct
            score["answer correct"][1] += 1
            if not refused:
                score["has citation"][0] += bool(re.search(r"\[\d+\]", res["answer"]))
                score["has citation"][1] += 1
            res["_hit"], res["_correct"] = hit, correct
        q_lang = "en" if guess_language(res["q"]) == "en" else "de"
        plain = re.sub(r"\[\d+\]|[\"“„”'‚‘][^\"“„”'‚‘]{0,200}[\"“”'‘]", " ", res["answer"])  # drop quotes/citations
        a_lang = "en" if guess_language(plain) == "en" else "de"
        score["language match"][0] += q_lang == a_lang
        score["language match"][1] += 1
        res["_ok"] = ok_refusal and (not should_answer or res.get("_correct"))

    for res in results:
        if args.show or not res["_ok"]:
            mark = "OK  " if res["_ok"] else "FAIL"
            top = ", ".join(f"{Path(h['file']).name[:18]}:p{h.get('page')}:{h['score']:.2f}" for h in res["hits"][:4])
            print(f"[{mark}] {res['id']:<18} refused={res['refused']!s:<10} hit={res.get('_hit', '-')!s:<5} | {res['q']}")
            print(f"        hits: {top or '-'}")
            print(f"        answer: {' '.join(res['answer'].split())[:900]}\n")

    print("summary:")
    for k, (a, n) in score.items():
        print(f"  {k:<36} {a}/{n}" if n else f"  {k:<36} {a}")


if __name__ == "__main__":
    main()
