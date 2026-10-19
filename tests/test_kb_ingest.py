import importlib.util
import json
from pathlib import Path

import pytest

from app.config import Settings
from app.providers import FakeProvider
from app.store import VectorStore

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "ingest_kb.py"
_spec = importlib.util.spec_from_file_location("ingest_kb", _SCRIPT)
ingest_kb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ingest_kb)


@pytest.fixture
def settings(tmp_path: Path):
    return Settings(provider="fake", openai_api_key=None, openai_base_url=None, chat_model="fake-chat",
                    embed_model="fake-embed-64", context_window=1000, data_dir=tmp_path / "data",
                    app_mode="kb", kb_dir=tmp_path / "kb", kb_chunk_size=200, kb_overlap=20)


def _pdf(path: Path, pages: list[str]) -> None:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_font("Helvetica", size=11)
    for text in pages:
        pdf.add_page()
        pdf.multi_cell(0, 6, text)
    pdf.output(str(path))


def _run(settings, **kw):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return {r["name"]: r for r in ingest_kb.run(settings, FakeProvider(), **kw)}


def _stored(settings) -> dict[str, int]:
    s = VectorStore(settings.db_path)
    try:
        return {f["name"]: f["chunk_count"] for f in s.list_files(settings.kb_collection)}
    finally:
        s.close()


def test_missing_folder_is_created_with_hint(settings, capsys):
    assert ingest_kb.run(settings, FakeProvider()) == []
    assert settings.kb_dir.is_dir()
    assert f"Put the DTM PDFs into {settings.kb_dir} and run again." in capsys.readouterr().out


def test_new_unchanged_updated_removed(settings, capsys):
    kb = settings.kb_dir
    kb.mkdir()
    _pdf(kb / "Prüfungsordnung.pdf", ["Die Prüfung dauert 90 Minuten.", "Wiederholung einmal möglich."])
    (kb / "faq.txt").write_text("Die Mensa öffnet um 11 Uhr.", "utf-8")
    (kb / "notes.exe").write_bytes(b"MZ")  # unsupported: ignored
    (kb / "~$lock.docx").write_bytes(b"x")  # Office lock file: ignored

    rows = _run(settings)
    assert set(rows) == {"Prüfungsordnung.pdf", "faq.txt"}
    assert rows["Prüfungsordnung.pdf"]["status"] == "new" and rows["Prüfungsordnung.pdf"]["pages"] == 2
    assert rows["faq.txt"]["pages"] is None and rows["faq.txt"]["chunks"] == 1
    out = capsys.readouterr().out
    assert "Prüfungsordnung.pdf" in out and "new" in out
    manifest = json.loads(ingest_kb.manifest_path(settings).read_text("utf-8"))
    assert set(manifest) == {"Prüfungsordnung.pdf", "faq.txt"} and len(manifest["faq.txt"]) == 64
    first = _stored(settings)

    rows = _run(settings)
    assert {r["status"] for r in rows.values()} == {"unchanged"}
    assert rows["Prüfungsordnung.pdf"]["chunks"] == first["Prüfungsordnung.pdf"]
    assert _stored(settings) == first

    (kb / "faq.txt").write_text("Die Mensa öffnet um 11 Uhr. Sie schließt um 15 Uhr.", "utf-8")
    (kb / "Prüfungsordnung.pdf").unlink()
    rows = _run(settings)
    assert rows["faq.txt"]["status"] == "updated"
    assert rows["Prüfungsordnung.pdf"]["status"] == "removed"
    assert _stored(settings) == {"faq.txt": 1}
    assert set(json.loads(ingest_kb.manifest_path(settings).read_text("utf-8"))) == {"faq.txt"}


def test_rebuild_and_store_out_of_sync(settings):
    settings.kb_dir.mkdir()
    (settings.kb_dir / "a.md").write_text("# Titel\n\nText über das Studium.", "utf-8")
    _run(settings)
    rows = _run(settings, rebuild=True)
    assert rows["a.md"]["status"] == "new"
    assert _stored(settings) == {"a.md": 1}
    # Manifest says unchanged but the DB was wiped: the file is ingested again.
    Path(settings.db_path).unlink()
    assert _run(settings)["a.md"]["status"] == "new"
    assert _stored(settings) == {"a.md": 1}


def test_bad_file_is_reported_and_others_continue(settings, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    _pdf(other / "scan.pdf", [""])  # no text, like a scanned PDF without OCR
    (other / "ok.txt").write_text("Semesterbeitrag zahlen.", "utf-8")
    rows = _run(settings, kb_dir=other)
    assert rows["scan.pdf"]["status"] == "error" and "no extractable text" in rows["scan.pdf"]["error"]
    assert rows["ok.txt"]["status"] == "new"
    assert "scan.pdf" not in json.loads(ingest_kb.manifest_path(settings).read_text("utf-8"))
    assert "error" in ingest_kb.summary(list(rows.values()))


def test_main_uses_settings_and_dir(settings, monkeypatch, capsys):
    settings.kb_dir.mkdir()
    (settings.kb_dir / "x.txt").write_text("Hallo Welt.", "utf-8")
    monkeypatch.setattr(ingest_kb, "get_settings", lambda: settings)
    monkeypatch.setattr(ingest_kb, "get_provider", lambda s: FakeProvider())
    settings.data_dir.mkdir()
    assert ingest_kb.main([]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1].startswith("1 files, 1 chunks: 1 new")
