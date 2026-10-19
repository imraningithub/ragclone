"""Compare Jan's real retrieval with the clone's on the same documents and questions.

Needs: Jan's Local API Server running (http://127.0.0.1:1337/v1) and a snapshot of Jan's
attachments DB in compare/jan_snapshot/, plus compare/clone_results.json from compare_clone.py.
Usage: .venv/Scripts/python scripts/compare_jan.py -> compare/jan_vs_clone.json + .md
"""
from __future__ import annotations

import glob
import json
import re
import sqlite3
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import rag  # noqa: E402
from app.config import RagSettings, get_settings  # noqa: E402
from app.store import VectorStore  # noqa: E402

JAN_API = "http://127.0.0.1:1337/v1"
JAN_EMBED = "sentence-transformer-mini"
JAN_K, JAN_T = 3, 0.3

# Per question: regexes that must all match somewhere in the retrieved chunks (whitespace-normalized).
EXPECT = [
    [r"27 days", r"March 31"],
    [r"core office days"],
    [r"150 EUR"],
    [r"14 characters"],
    [r"90 minutes"],
    [r"3,000 charge cycles"],
    [r"West\W+(Quarter: )?Q4 2025\W+(Units sold: )?700", r"R7 warehouse robot\W+(List price \(EUR\): )?54000"],
    [],  # off-topic: nothing should be retrieved
]


def norm(s: str) -> str:
    return " ".join(s.split())


def covered(texts: list[str], patterns: list[str]) -> bool:
    if not patterns:
        return not texts
    blob = norm(" \n ".join(texts))
    return all(re.search(p, blob) for p in patterns)


class JanEmbedProvider:
    """Clone provider that embeds with Jan's own MiniLM model through Jan's local API."""
    name, chat_model, embed_model = "jan-local", "-", JAN_EMBED

    def __init__(self) -> None:
        self.client = OpenAI(base_url=JAN_API, api_key="not-needed")

    def embed(self, texts):
        resp = self.client.embeddings.create(model=JAN_EMBED, input=texts)
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    def count_tokens(self, text):
        return len(text) // 4


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (b @ a) / (np.linalg.norm(b, axis=1) * np.linalg.norm(a) + 1e-12)


def main() -> None:
    clone = json.loads((ROOT / "compare" / "clone_results.json").read_text("utf-8"))
    questions = [r["question"] for r in clone["profiles"]["clone_tuned"]]
    jan_emb = JanEmbedProvider()

    db = glob.glob(str(ROOT / "compare" / "jan_snapshot" / "*.db"))[0]
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute("SELECT f.name, c.chunk_file_order, c.text, c.embedding FROM chunks c "
                       "JOIN files f ON f.id = c.file_id ORDER BY f.name, c.chunk_file_order").fetchall()
    con.close()
    names = [r[0] for r in rows]
    orders = [r[1] for r in rows]
    texts = [r[2] for r in rows]
    mat = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])

    # Fidelity check: the API must reproduce the vectors Jan stored at ingest time.
    probe = np.array(jan_emb.embed(texts[:3]), dtype=np.float32)
    fidelity = [float(cos(probe[i], mat[i:i + 1])[0]) for i in range(3)]
    print("fidelity (re-embedded vs stored):", [round(f, 4) for f in fidelity])

    # Cache Jan's query embeddings so reruns don't need Jan's API server.
    cache = ROOT / "compare" / "jan_query_embeddings.json"
    cached = json.loads(cache.read_text("utf-8")) if cache.exists() else {}
    if set(questions) - cached.keys():
        cached.update(zip(questions, jan_emb.embed(questions)))
        cache.write_text(json.dumps(cached), "utf-8")
    q_vecs = np.array([cached[q] for q in questions], dtype=np.float32)

    # Clone chunking + Jan's embedding model, same k/threshold as Jan.
    settings = get_settings()
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(str(Path(tmp) / "x.db"), enable_ann=False)
        rs = replace(RagSettings(), parse_mode="embeddings")
        for f in sorted((ROOT / "samples").iterdir()):
            rag.ingest_file(store, jan_emb, settings, rs, collection="x", path=str(f), name=f.name,
                            file_type="application/octet-stream", parse_mode="embeddings")
        clone_minilm = [store.search("x", list(map(float, v)), limit=JAN_K, threshold=JAN_T, mode="linear")
                        for v in q_vecs]
        clone_chunk_counts = {f["name"]: f["chunk_count"] for f in store.list_files("x")}
        store.close()

    report = {"fidelity": fidelity, "jan_chunk_counts": {}, "clone_chunk_counts": clone_chunk_counts,
              "questions": []}
    for n in names:
        report["jan_chunk_counts"][n] = report["jan_chunk_counts"].get(n, 0) + 1

    for qi, q in enumerate(questions):
        scores = cos(q_vecs[qi], mat)
        order = np.argsort(-scores)
        jan_hits = [{"file": names[i], "chunk": orders[i], "score": float(scores[i]), "text": texts[i]}
                    for i in order[:JAN_K] if scores[i] >= JAN_T]
        jan_best = [{"file": names[i], "chunk": orders[i], "score": float(scores[i])} for i in order[:3]]
        cm = [{"file": h["file_name"], "chunk": h["chunk_index"], "score": h["score"], "text": h["text"]}
              for h in clone_minilm[qi]]
        variants = {
            "jan": jan_hits,
            "clone_minilm_k3_t0.3": cm,
            "clone_openai_k3_t0.3": [{"file": c["file_name"], "chunk": c["chunk_index"], "score": c["score"],
                                      "text": c["text"]} for c in clone["profiles"]["jan_defaults"][qi]["chunks"]],
            "clone_openai_k5_t0.2": [{"file": c["file_name"], "chunk": c["chunk_index"], "score": c["score"],
                                      "text": c["text"]} for c in clone["profiles"]["clone_tuned"][qi]["chunks"]],
        }
        entry = {"question": q, "jan_top3_unfiltered": jan_best, "variants": {},
                 "clone_answers": {p: clone["profiles"][p][qi]["answer"] for p in clone["profiles"]}}
        for name, hits in variants.items():
            entry["variants"][name] = {"answer_in_context": covered([h["text"] for h in hits], EXPECT[qi]),
                                       "hits": hits}
        report["questions"].append(entry)

    out = ROOT / "compare" / "jan_vs_clone.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")

    cols = list(report["questions"][0]["variants"])
    print("\nanswer-bearing chunk retrieved?")
    print(f"{'question':<62}" + "".join(f"{c:>24}" for c in cols))
    for e in report["questions"]:
        cells = []
        for c in cols:
            v = e["variants"][c]
            hits = ",".join(f"{h['file'][:4]}#{h['chunk']}:{h['score']:.2f}" for h in v["hits"]) or "-"
            cells.append(f"{'Y' if v['answer_in_context'] else 'N'} {hits}"[:23])
        print(f"{e['question'][:60]:<62}" + "".join(f"{c:>24}" for c in cells))
    totals = {c: sum(e["variants"][c]["answer_in_context"] for e in report["questions"]) for c in cols}
    print("\ntotals:", totals, "of", len(questions))
    print("jan chunks:", report["jan_chunk_counts"], "| clone chunks:", clone_chunk_counts)
    print("saved", out)


if __name__ == "__main__":
    main()
