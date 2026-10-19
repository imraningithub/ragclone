"""Run the Jan-comparison question set through the clone and save answers + retrieved chunks.

Two profiles: Jan's defaults (top-k 3, threshold 0.3) and the clone's tuned defaults (5, 0.2).
Usage: .venv/Scripts/python scripts/compare_clone.py  -> compare/clone_results.json
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import rag  # noqa: E402
from app.config import RagSettings, get_settings  # noqa: E402
from app.providers import get_provider  # noqa: E402
from app.store import VectorStore  # noqa: E402

QUESTIONS = [
    "How many vacation days do full-time staff get, and until when can unused days be carried over?",
    "Can I work from home on Fridays?",
    "How much is the hotel limit in big cities?",
    "What is the minimum password length?",
    "How long does a full charge of the R7 take?",
    "How long is the battery warranty?",
    "How many units did the West region sell in Q4 2025, and what does one R7 cost?",
    "What is the capital of Australia?",
]

PROFILES = {
    "jan_defaults": dict(retrieval_limit=3, retrieval_threshold=0.3),
    "clone_tuned": dict(retrieval_limit=5, retrieval_threshold=0.2),
}


def main() -> None:
    settings = get_settings()
    provider = get_provider(settings)
    out = {"chat_model": provider.chat_model, "embed_model": provider.embed_model, "profiles": {}}
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(str(Path(tmp) / "c.db"))
        base = replace(RagSettings(), parse_mode="embeddings")
        for f in sorted((ROOT / "samples").iterdir()):
            rag.ingest_file(store, provider, settings, base, collection="c", path=str(f), name=f.name,
                            file_type="application/octet-stream", parse_mode="embeddings")
        out["files"] = [{k: f[k] for k in ("name", "chunk_count")} for f in store.list_files("c")]
        for pname, overrides in PROFILES.items():
            rs = base.update(overrides)
            rows = []
            for q in QUESTIONS:
                messages, cites = rag.build_messages(store, provider, settings, rs, collection="c",
                                                     history=[], user_message=q)
                answer = "".join(provider.chat_stream(messages)).strip()
                rows.append({"question": q, "answer": answer,
                             "chunks": [{k: c[k] for k in ("n", "file_name", "chunk_index", "score", "text")}
                                        for c in cites]})
                print(f"[{pname}] {q}\n  -> {answer}\n  chunks: "
                      + ", ".join(f"{c['file_name']}#{c['chunk_index']}({c['score']:.2f})" for c in cites))
            out["profiles"][pname] = rows
        store.close()
    dest = ROOT / "compare" / "clone_results.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), "utf-8")
    print("\nsaved", dest)


if __name__ == "__main__":
    main()
