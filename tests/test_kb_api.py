"""KB mode (DTM Studienassistent) HTTP API tests. Refusal reason when retrieval finds nothing: "no_context"."""
import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

from app import guardrails
from app.config import Settings
from app.main import create_app
from app.providers import FakeProvider
from app.store import VectorStore

CONTACT = "das Studienbüro DTM"
CHUNK = "Die Anmeldung zur Prüfung erfolgt online im Campusportal bis zum 15. Mai."


class CountingProvider(FakeProvider):
    """FakeProvider that counts answer calls (json_mode=False) separately from planner calls."""

    def __init__(self):
        self.answer_calls = 0
        self.plan_calls = 0

    def complete(self, messages, *, json_mode=False, temperature=None, model=None):
        if json_mode:
            self.plan_calls += 1
        else:
            self.answer_calls += 1
        return super().complete(messages, json_mode=json_mode, temperature=temperature)


def _settings(tmp_path, **kw):
    base = Settings(provider="fake", openai_api_key=None, openai_base_url=None, chat_model="fake-chat",
                    embed_model="fake-embed-64", context_window=128000, data_dir=tmp_path / "data")
    base.data_dir.mkdir(parents=True, exist_ok=True)
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir(exist_ok=True)
    (kb_dir / "pruefungsordnung.pdf").write_bytes(b"%PDF-1.4\n% test\n")
    (tmp_path / "secret.txt").write_text("top secret", "utf-8")
    return dataclasses.replace(base, app_mode="kb", kb_dir=kb_dir, refusal_contact=CONTACT,
                               admin_token="admin-secret", **kw)


def _seed(settings):
    (settings.data_dir).mkdir(parents=True, exist_ok=True)
    store = VectorStore(settings.db_path)
    p = FakeProvider()
    store.create_collection(settings.kb_collection, FakeProvider.dim, FakeProvider.embed_model)
    fid = store.add_file(settings.kb_collection, name="pruefungsordnung.pdf",
                         path=str(settings.kb_dir / "pruefungsordnung.pdf"), type="application/pdf",
                         size=16, mode="embeddings")
    store.insert_chunks(settings.kb_collection, fid, [CHUNK], p.embed([CHUNK]), metas=[{"page": 3}])
    store.close()


def _make(tmp_path, **kw):
    settings = _settings(tmp_path, **kw)
    _seed(settings)
    provider = CountingProvider()
    return TestClient(create_app(settings, provider)), provider, settings


@pytest.fixture
def kb(tmp_path):
    client, provider, settings = _make(tmp_path)
    with client:
        yield client, provider, settings


def _events(body: str) -> list[tuple[str, object]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def _thread(client):
    return client.post("/api/threads", json={}).json()["id"]


def test_status_has_mode_and_documents(kb):
    client, _, _ = kb
    s = client.get("/api/status").json()
    assert s["mode"] == "kb" and s["kb_name"] == "dtm" and s["refusal_contact"] == CONTACT
    assert s["documents"] == [{"name": "pruefungsordnung.pdf", "chunk_count": 1}]


def test_uploads_forbidden(kb):
    client, _, _ = kb
    tid = _thread(client)
    r = client.post(f"/api/threads/{tid}/files", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 403 and r.json()["detail"] == "Uploads are disabled in this assistant"
    assert client.delete(f"/api/threads/{tid}/files/abc").status_code == 403


def test_settings_put_requires_admin_token(kb):
    client, _, _ = kb
    assert client.get("/api/settings").status_code == 200
    assert client.put("/api/settings", json={"retrieval_limit": 7}).status_code == 403
    assert client.put("/api/settings", json={"retrieval_limit": 7},
                      headers={"X-Admin-Token": "wrong"}).status_code == 403
    r = client.put("/api/settings", json={"retrieval_limit": 7}, headers={"X-Admin-Token": "admin-secret"})
    assert r.status_code == 200 and r.json()["retrieval_limit"] == 7


def test_settings_put_forbidden_without_configured_token(tmp_path):
    settings = dataclasses.replace(_settings(tmp_path), admin_token=None)
    with TestClient(create_app(settings, FakeProvider())) as c:
        assert c.put("/api/settings", json={}, headers={"X-Admin-Token": ""}).status_code == 403


def test_kb_document_served_and_traversal_blocked(kb):
    client, _, _ = kb
    r = client.get("/kb/pruefungsordnung.pdf")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/pdf")
    assert r.content.startswith(b"%PDF")
    assert client.get("/kb/missing.pdf").status_code == 404
    assert client.get("/kb/..%2Fsecret.txt").status_code == 404
    assert client.get("/kb/..%5Csecret.txt").status_code == 404


def test_chat_answer_with_citations(kb):
    client, provider, settings = kb
    tid = _thread(client)
    q = "Wann erfolgt die Anmeldung zur Prüfung im Campusportal?"
    r = client.post(f"/api/threads/{tid}/chat", json={"message": q})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = _events(r.text)
    assert events[0][0] == "citations"
    cits = events[0][1]
    assert cits and cits[0]["page"] == 3 and cits[0]["file_name"] == "pruefungsordnung.pdf"
    tokens = [d["text"] for e, d in events if e == "token"]
    assert tokens and "[1]" in "".join(tokens)
    assert events[-1][0] == "done"
    done = events[-1][1]
    assert done["refused"] is None and done["title"]
    assert provider.answer_calls == 1

    thread = client.get(f"/api/threads/{tid}").json()
    assert [m["role"] for m in thread["messages"]] == ["user", "assistant"]
    assert thread["messages"][0]["content"] == q
    assert thread["messages"][1]["content"] == "".join(tokens)
    assert thread["messages"][1]["citations"][0]["page"] == 3

    log = settings.kb_log_path.read_text("utf-8").strip().splitlines()
    rec = json.loads(log[-1])
    assert rec["thread_id"] == tid and rec["reason"] is None and rec["hits"][0]["page"] == 3


def test_chat_refuses_without_context_and_skips_model(tmp_path):
    client, provider, settings = _make(tmp_path, kb_threshold=0.9)
    with client:
        tid = _thread(client)
        r = client.post(f"/api/threads/{tid}/chat", json={"message": "zebra banana trumpet volcano"})
        assert r.status_code == 200
        events = _events(r.text)
        assert events[0] == ("citations", [])
        text = "".join(d["text"] for e, d in events if e == "token")
        assert CONTACT in text
        assert events[-1][0] == "done" and events[-1][1]["refused"] == "no_context"
        assert provider.answer_calls == 0  # the planner may run, the answer model must not
        msgs = client.get(f"/api/threads/{tid}").json()["messages"]
        assert msgs[1]["content"] == text and not msgs[1]["citations"]


def test_chat_question_validation(kb):
    client, provider, _ = kb
    tid = _thread(client)
    assert client.post(f"/api/threads/{tid}/chat", json={"message": "   "}).status_code == 400
    long_q = "x" * (guardrails.MAX_QUESTION_CHARS + 1)
    assert client.post(f"/api/threads/{tid}/chat", json={"message": long_q}).status_code == 400
    assert provider.answer_calls == 0 and provider.plan_calls == 0
    assert client.get(f"/api/threads/{tid}").json()["messages"] == []


def test_chat_rate_limit(tmp_path):
    client, _, _ = _make(tmp_path, rate_limit_per_min=1)
    with client:
        tid = _thread(client)
        assert client.post(f"/api/threads/{tid}/chat", json={"message": "Prüfung Anmeldung"}).status_code == 200
        r = client.post(f"/api/threads/{tid}/chat", json={"message": "Prüfung Anmeldung"})
        assert r.status_code == 429


def test_chat_mode_status_and_done_refused_null(tmp_path):
    settings = Settings(provider="fake", openai_api_key=None, openai_base_url=None, chat_model="fake-chat",
                        embed_model="fake-embed-64", context_window=128000, data_dir=tmp_path)
    with TestClient(create_app(settings, FakeProvider())) as c:
        s = c.get("/api/status").json()
        assert s["mode"] == "chat" and "documents" not in s
        assert c.get("/kb/anything.pdf").status_code == 404
        tid = c.post("/api/threads", json={}).json()["id"]
        events = _events(c.post(f"/api/threads/{tid}/chat", json={"message": "hi"}).text)
        assert events[-1][0] == "done" and events[-1][1]["refused"] is None
