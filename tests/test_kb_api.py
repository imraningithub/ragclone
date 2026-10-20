"""KB mode (DTM Studienassistent) HTTP API tests. Refusal reason when retrieval finds nothing: "no_context"."""
import dataclasses
import json
import re

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
    """FakeProvider that counts (streamed) answer calls separately from planner calls and records models."""

    def __init__(self, answer=None):
        self.answer_calls = 0
        self.plan_calls = 0
        self.models = {}
        self.answer = answer  # fixed answer text to stream instead of the fake one

    def complete(self, messages, *, json_mode=False, temperature=None, model=None, usage=None):
        if json_mode:
            self.plan_calls += 1
            self.models["plan"] = model
        return super().complete(messages, json_mode=json_mode, temperature=temperature, model=model, usage=usage)

    def chat_stream(self, messages, temperature=None, model=None, usage=None):
        self.answer_calls += 1
        self.models["answer"] = model
        if self.answer is not None:
            yield from re.findall(r"\s*\S+", self.answer)
            return
        yield from super().chat_stream(messages, temperature=temperature, model=model, usage=usage)


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


def _make(tmp_path, answer=None, **kw):
    settings = _settings(tmp_path, **kw)
    _seed(settings)
    provider = CountingProvider(answer)
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
    # tokens stream first; the cited sources follow once the answer is complete
    assert [e for e, _ in events][-2:] == ["citations", "done"]
    cits = events[-2][1]
    assert cits and cits[0]["page"] == 3 and cits[0]["file_name"] == "pruefungsordnung.pdf"
    tokens = [d["text"] for e, d in events if e == "token"]
    assert tokens and "[1]" in "".join(tokens)
    assert "reset" not in [e for e, _ in events]
    done = events[-1][1]
    assert done["refused"] is None and done["title"]
    assert provider.answer_calls == 1
    assert provider.models == {"plan": settings.kb_plan_model, "answer": settings.kb_chat_model}

    thread = client.get(f"/api/threads/{tid}").json()
    assert [m["role"] for m in thread["messages"]] == ["user", "assistant"]
    assert thread["messages"][0]["content"] == q
    assert thread["messages"][1]["content"] == "".join(tokens)
    assert thread["messages"][1]["citations"][0]["page"] == 3

    log = settings.kb_log_path.read_text("utf-8").strip().splitlines()
    rec = json.loads(log[-1])
    assert rec["thread_id"] == tid and rec["reason"] is None and rec["hits"][0]["page"] == 3
    assert {"plan_s", "retrieve_s", "first_token_s", "total_s"} <= set(rec["timing"])


def test_chat_refuses_without_context_and_skips_model(tmp_path):
    client, provider, settings = _make(tmp_path, kb_threshold=0.9)
    with client:
        tid = _thread(client)
        r = client.post(f"/api/threads/{tid}/chat", json={"message": "zebra banana trumpet volcano"})
        assert r.status_code == 200
        events = _events(r.text)
        assert events[-2] == ("citations", [])
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


@pytest.mark.parametrize("answer,reason", [
    ("NO_ANSWER", "no_answer"),
    ("Die Anmeldung erfolgt irgendwann im Mai, ganz sicher.", "uncited"),
    ("Die Anmeldung erfolgt online [7].", "uncited"),
])
def test_refused_answers_never_stream_model_text(tmp_path, answer, reason):
    client, _, _ = _make(tmp_path, answer=answer)
    with client:
        tid = _thread(client)
        events = _events(client.post(f"/api/threads/{tid}/chat", json={"message": "Prüfung Anmeldung"}).text)
        text = "".join(d["text"] for e, d in events if e == "token")
        assert text == guardrails.refusal_text("de", CONTACT)
        assert "reset" not in [e for e, _ in events]
        assert events[-2] == ("citations", []) and events[-1][1]["refused"] == reason


def test_late_refusal_replaces_streamed_text(tmp_path):
    # Cited at first, then mostly NO_ANSWER: the final check refuses and the client must drop what it showed.
    client, _, _ = _make(tmp_path, answer="Anmeldung [1] " + "NO_ANSWER " * 10)
    with client:
        tid = _thread(client)
        events = _events(client.post(f"/api/threads/{tid}/chat", json={"message": "Prüfung Anmeldung"}).text)
        names = [e for e, _ in events]
        assert names.index("token") < names.index("reset")
        assert events[names.index("reset")][1]["text"] == guardrails.refusal_text("de", CONTACT)
        assert events[-1][1]["refused"] == "no_answer"
        msgs = client.get(f"/api/threads/{tid}").json()["messages"]
        assert msgs[1]["content"] == guardrails.refusal_text("de", CONTACT) and not msgs[1]["citations"]


def test_invalid_citation_is_cleaned_while_streaming(tmp_path):
    client, _, _ = _make(tmp_path, answer="Online im Campusportal [1] bis 15. Mai [9]. Weiteres [1, 9].")
    with client:
        tid = _thread(client)
        events = _events(client.post(f"/api/threads/{tid}/chat", json={"message": "Prüfung Anmeldung"}).text)
        text = "".join(d["text"] for e, d in events if e == "token")
        assert "reset" not in [e for e, _ in events]
        assert text == "Online im Campusportal [1] bis 15. Mai. Weiteres [1]."
        # released word by word once the first valid citation arrived, not buffered to the end
        assert len([e for e, _ in events if e == "token"]) >= 3
        assert [c["n"] for c in events[-2][1]] == [1]


def test_provider_error_before_first_token_is_502(tmp_path):
    client, provider, _ = _make(tmp_path)

    def broken(*a, **kw):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    provider.chat_stream = broken
    with client:
        tid = _thread(client)
        assert client.post(f"/api/threads/{tid}/chat", json={"message": "Prüfung Anmeldung"}).status_code == 502
        assert client.get(f"/api/threads/{tid}").json()["messages"] == []


def test_threads_are_private_per_visitor(kb):
    client, _, _ = kb
    tid = _thread(client)
    cookie = client.cookies.get("dtm_client")
    assert cookie and len(cookie) >= 32
    assert client.post(f"/api/threads/{tid}/chat", json={"message": "Prüfung Anmeldung"}).status_code == 200
    assert [t["id"] for t in client.get("/api/threads").json()] == [tid]
    with TestClient(client.app) as other:
        assert other.get("/api/threads").json() == []
        assert other.cookies.get("dtm_client") not in (None, cookie)
        assert other.get(f"/api/threads/{tid}").status_code == 404
        assert other.patch(f"/api/threads/{tid}", json={"title": "x"}).status_code == 404
        assert other.post(f"/api/threads/{tid}/chat", json={"message": "Prüfung"}).status_code == 404
        assert other.delete(f"/api/threads/{tid}").status_code == 404
        # a malformed cookie gets a fresh id, not someone else's threads
        other.cookies.clear()
        other.cookies.set("dtm_client", "short")
        assert other.get("/api/threads").json() == []
    assert len(client.get(f"/api/threads/{tid}").json()["messages"]) == 2
    assert client.delete(f"/api/threads/{tid}").status_code == 200


def test_client_cookie_flags(kb):
    client, _, _ = kb
    header = client.get("/api/status").headers["set-cookie"].lower()
    assert "dtm_client=" in header and "httponly" in header and "samesite=lax" in header
    assert "set-cookie" not in client.get("/api/status").headers  # an existing cookie is kept


def test_rate_limit_is_per_visitor(tmp_path):
    client, _, _ = _make(tmp_path, rate_limit_per_min=1)
    with client:
        tid = _thread(client)
        assert client.post(f"/api/threads/{tid}/chat", json={"message": "Prüfung Anmeldung"}).status_code == 200
        assert client.post(f"/api/threads/{tid}/chat", json={"message": "Prüfung Anmeldung"}).status_code == 429
        with TestClient(client.app) as other:  # same IP, different browser
            tid2 = other.post("/api/threads", json={}).json()["id"]
            assert other.post(f"/api/threads/{tid2}/chat", json={"message": "Prüfung Anmeldung"}).status_code == 200


def test_ip_rate_limit_catches_cookie_dropping(tmp_path):
    client, _, _ = _make(tmp_path, rate_limit_per_min=5, rate_limit_ip_per_min=1)
    with client:
        tid = _thread(client)
        assert client.post(f"/api/threads/{tid}/chat", json={"message": "Prüfung Anmeldung"}).status_code == 200
        with TestClient(client.app) as other:
            tid2 = other.post("/api/threads", json={}).json()["id"]
            assert other.post(f"/api/threads/{tid2}/chat", json={"message": "Prüfung Anmeldung"}).status_code == 429


def test_chat_mode_threads_are_not_scoped(tmp_path):
    settings = Settings(provider="fake", openai_api_key=None, openai_base_url=None, chat_model="fake-chat",
                        embed_model="fake-embed-64", context_window=128000, data_dir=tmp_path)
    app = create_app(settings, FakeProvider())
    with TestClient(app) as a, TestClient(app) as b:
        tid = a.post("/api/threads", json={}).json()["id"]
        assert "set-cookie" not in a.get("/api/threads").headers
        assert [t["id"] for t in b.get("/api/threads").json()] == [tid]
