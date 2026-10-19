"""Measure retrieval quality on samples/ and suggest a similarity threshold.

Ingests every sample file in embeddings mode (real provider from .env), runs labeled questions,
and reports, per threshold: hit rate (the right chunk is in the top-k) and how often off-topic
questions still return excerpts.

Usage: .venv/Scripts/python scripts/eval_retrieval.py [--k 3]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import rag  # noqa: E402
from app.config import RagSettings, get_settings  # noqa: E402
from app.providers import get_provider  # noqa: E402
from app.store import VectorStore  # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def norm(s: str) -> str:
    # PDF text hard-wraps lines, so compare with collapsed whitespace.
    return " ".join(s.split())

# (question, file the answer lives in, text that must appear in the retrieved chunk)
QUESTIONS = [
    ("How many vacation days do full-time staff get?", "employee_handbook.pdf", "27 days"),
    ("Until when can I carry over unused holidays?", "employee_handbook.pdf", "March 31"),
    ("When do I need a doctor's note?", "employee_handbook.pdf", "third day"),
    ("Can I work from home on Fridays?", "employee_handbook.pdf", "Fridays are core office days"),
    ("How long can I work from another country?", "employee_handbook.pdf", "20 working"),
    ("What do I do if my laptop gets stolen?", "employee_handbook.pdf", "within 24 hours"),
    ("What is the minimum password length?", "employee_handbook.pdf", "14 characters"),
    ("How much is the hotel limit in big cities?", "employee_handbook.pdf", "150 EUR"),
    ("What is my yearly training budget?", "employee_handbook.pdf", "1,500 EUR"),
    ("When are salary raises effective?", "employee_handbook.pdf", "January 1"),
    ("What is the notice period after probation?", "employee_handbook.pdf", "three months"),
    ("Do I have to declare a gift from a supplier?", "employee_handbook.pdf", "50 EUR"),
    ("How long does it take to fully charge the R7?", "r7_product_spec.docx", "90 minutes"),
    ("What safety standard is the robot certified to?", "r7_product_spec.docx", "ISO 3691-4"),
    ("How heavy a load can the R7 carry?", "r7_product_spec.docx", "600 kg"),
    ("How often should the LiDAR be cleaned?", "r7_product_spec.docx", "cleaned weekly"),
    ("How long is the battery warranty?", "r7_product_spec.docx", "3,000 charge cycles"),
    ("How many units did the West region sell in Q4 2025?", "sales_and_prices.xlsx", "West | Q4 2025 | 700"),
    ("What was East region revenue in Q1 2025?", "sales_and_prices.xlsx", "East | Q1 2025 | 150"),
    ("What does the R7 robot cost?", "sales_and_prices.xlsx", "54000"),
    ("How much is the fleet manager licence?", "sales_and_prices.xlsx", "Fleet manager"),
]

OFF_TOPIC = [
    "What is the capital of Australia?",
    "Give me a recipe for banana bread.",
    "Who won the 2014 football world cup?",
    "Explain how photosynthesis works.",
    "What's a good name for a cat?",
]

THRESHOLDS = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    settings = get_settings()
    provider = get_provider(settings)
    rs = replace(RagSettings(), parse_mode="embeddings")
    print(f"provider={provider.name} embed_model={provider.embed_model} k={args.k}\n")

    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(str(Path(tmp) / "eval.db"))
        coll = "eval"
        for f in sorted(SAMPLES.iterdir()):
            info = rag.ingest_file(store, provider, settings, rs, collection=coll, path=str(f), name=f.name,
                                   file_type="application/octet-stream", parse_mode="embeddings")
            print(f"ingested {f.name}: {info['chunk_count']} chunks")
        total = sum(f["chunk_count"] for f in store.list_files(coll))

        q_vecs = provider.embed([q for q, _, _ in QUESTIONS] + OFF_TOPIC)
        # Score every chunk for every question once; thresholds are applied afterwards.
        ranked = [store.search(coll, v, limit=total, threshold=-1.0, mode="linear") for v in q_vecs]
        store.close()

    relevant_scores, best_wrong, misses = [], [], []
    for (q, fname, needle), hits in zip(QUESTIONS, ranked):
        good = [h for h in hits if h["file_name"] == fname and norm(needle) in norm(h["text"])]
        rank = hits.index(good[0]) + 1 if good else None
        relevant_scores.append(good[0]["score"] if good else float("nan"))
        best_wrong.append(max((h["score"] for h in hits if h not in good), default=0.0))
        if rank is None or rank > args.k:
            misses.append((q, rank, good[0]["score"] if good else None, hits[0]["file_name"], hits[0]["text"][:70]))
    off_top = [hits[0]["score"] for hits in ranked[len(QUESTIONS):]]

    rel = [s for s in relevant_scores if s == s]
    print(f"\nrelevant chunk score: min {min(rel):.3f}  median {statistics.median(rel):.3f}  max {max(rel):.3f}")
    print(f"off-topic best score: min {min(off_top):.3f}  median {statistics.median(off_top):.3f}  "
          f"max {max(off_top):.3f}")

    print(f"\n{'threshold':>9}  {'hit@k':>6}  {'off-topic w/ hits':>17}  {'avg excerpts/q':>14}")
    for t in THRESHOLDS:
        hit = sum(1 for (q, f, n), hits in zip(QUESTIONS, ranked)
                  if any(h["file_name"] == f and norm(n) in norm(h["text"]) and h["score"] >= t for h in hits[:args.k]))
        off = sum(1 for s in off_top if s >= t)
        avg = statistics.mean(min(args.k, sum(1 for h in hits if h["score"] >= t)) for hits in ranked[:len(QUESTIONS)])
        print(f"{t:>9.2f}  {hit:>3}/{len(QUESTIONS):<2}  {off:>10}/{len(OFF_TOPIC):<6}  {avg:>14.2f}")

    if misses:
        print(f"\nnot in top-{args.k} (rank, score, top hit instead):")
        for q, rank, score, top_file, top_text in misses:
            print(f"  - {q}  rank={rank} score={score}  top={top_file}: {top_text!r}")


if __name__ == "__main__":
    main()
