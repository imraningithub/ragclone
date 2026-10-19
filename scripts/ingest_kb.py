"""Ingest the knowledge-base folder (the DTM PDFs) into the KB collection.

Every supported file in settings.kb_dir is embedded into settings.kb_collection. A manifest
(data/kb_manifest.json, file name -> sha256) makes reruns incremental: unchanged files are skipped,
changed files are re-ingested and files that left the folder are deleted from the store.

Usage: .venv/Scripts/python scripts/ingest_kb.py [--rebuild] [--dir PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pypdf  # noqa: E402

from app import rag  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app.parsing import SUPPORTED_EXTENSIONS  # noqa: E402
from app.providers import get_provider  # noqa: E402
from app.store import VectorStore  # noqa: E402


def manifest_path(settings: Settings) -> Path:
    return settings.data_dir / "kb_manifest.json"


def load_manifest(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def save_manifest(path: Path, manifest: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(manifest.items())), indent=2, ensure_ascii=False), "utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _kb_files(folder: Path) -> list[Path]:
    # Skip hidden files and Office lock files ("~$Modulhandbuch.docx").
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith((".", "~$"))
    )


def _page_count(path: Path) -> int | None:
    if path.suffix.lower() != ".pdf":
        return None
    try:
        return len(pypdf.PdfReader(str(path)).pages)
    except Exception:
        return None


def _print_row(row: dict) -> None:
    pages = "-" if row["pages"] is None else row["pages"]
    line = f"{row['name']:<50} pages={pages!s:<4} chunks={row['chunks']:<5} {row['status']}"
    if row.get("error"):
        line += f": {row['error']}"
    print(line)


def run(settings: Settings, provider, *, rebuild: bool = False, kb_dir: Path | str | None = None,
        store: VectorStore | None = None) -> list[dict]:
    """Sync the KB folder into the store. Returns one row per file:
    {"name", "pages" (PDFs, else None), "chunks", "status": new|updated|unchanged|removed|error, "error"?}.
    """
    folder = Path(kb_dir) if kb_dir is not None else Path(settings.kb_dir)
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Put the DTM PDFs into {folder} and run again.")
        return []

    own_store = store is None
    if own_store:
        store = VectorStore(settings.db_path)
    coll = settings.kb_collection
    mpath = manifest_path(settings)
    manifest = {} if rebuild else load_manifest(mpath)
    try:
        if rebuild:
            store.delete_collection(coll)

        # The store is the source of truth for what is ingested; the manifest only remembers hashes.
        stored: dict[str, list[dict]] = {}
        for f in store.list_files(coll):
            stored.setdefault(f["name"], []).append(f)

        files = _kb_files(folder)
        if not files:
            print(f"Put the DTM PDFs into {folder} and run again.")
        rows: list[dict] = []
        present = {p.name for p in files}

        for name in sorted((set(manifest) | set(stored)) - present):
            for f in stored.get(name, []):
                store.delete_file(coll, f["id"])
            manifest.pop(name, None)
            rows.append({"name": name, "pages": None, "chunks": 0, "status": "removed"})
            _print_row(rows[-1])

        for path in files:
            digest = sha256(path)
            row = {"name": path.name, "pages": _page_count(path), "chunks": 0}
            old = stored.get(path.name, [])
            if old and manifest.get(path.name) == digest:
                row.update(chunks=sum(f["chunk_count"] for f in old), status="unchanged")
            else:
                row["status"] = "updated" if old else "new"
                for f in old:
                    store.delete_file(coll, f["id"])
                manifest.pop(path.name, None)
                try:
                    info = rag.ingest_kb_file(store, provider, settings, collection=coll,
                                              path=str(path), name=path.name)
                except Exception as exc:  # one bad file must not stop the rest
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                else:
                    row["chunks"] = info["chunk_count"]
                    manifest[path.name] = digest
            rows.append(row)
            _print_row(row)
        return rows
    finally:
        save_manifest(mpath, manifest)  # also after a crash, so finished files are not redone
        if own_store:
            store.close()


def summary(rows: list[dict]) -> str:
    counts = {s: sum(r["status"] == s for r in rows) for s in ("new", "updated", "unchanged", "removed", "error")}
    total = sum(r["chunks"] for r in rows)
    return (f"{len(rows) - counts['removed']} files, {total} chunks: " +
            ", ".join(f"{n} {s}" for s, n in counts.items()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rebuild", action="store_true", help="drop the KB collection and ingest everything again")
    ap.add_argument("--dir", type=Path, default=None, help="KB folder (default: settings.kb_dir / KB_DIR)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # umlauts in file names on a cp1252 console
    settings = get_settings()
    provider = get_provider(settings)
    rows = run(settings, provider, rebuild=args.rebuild, kb_dir=args.dir)
    print(summary(rows))
    return 1 if any(r["status"] == "error" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
