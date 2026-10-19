import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.providers import FakeProvider


@pytest.fixture
def client(tmp_path):
    settings = Settings(provider="fake", openai_api_key=None, openai_base_url=None, chat_model="fake-chat",
                        embed_model="fake-embed-64", context_window=128000, data_dir=tmp_path)
    with TestClient(create_app(settings, FakeProvider())) as c:
        yield c


def _events(body: str) -> list[tuple[str, object]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def _upload(client, tid, name, text, parse_mode=None):
    data = {"parse_mode": parse_mode} if parse_mode else {}
    return client.post(f"/api/threads/{tid}/files", files={"file": (name, text.encode(), "text/plain")}, data=data)


def test_index_and_status(client):
    assert client.get("/").status_code == 200
    s = client.get("/api/status").json()
    assert {"provider", "chat_model", "embed_model", "ann_available", "context_window"} <= s.keys()


def test_full_flow(client):
    tid = client.post("/api/threads", json={}).json()["id"]

    inline = _upload(client, tid, "notes.txt", "The meeting is on Friday.")
    assert inline.status_code == 200 and inline.json()["mode"] == "inline"
    doc = "\n\n".join(f"Section {i}: giraffes eat acacia leaves number {i}." for i in range(50))
    emb = _upload(client, tid, "zoo.txt", doc, parse_mode="embeddings")
    assert emb.status_code == 200 and emb.json()["mode"] == "embeddings" and emb.json()["chunk_count"] > 1
    assert _upload(client, tid, "zoo.txt", doc).status_code == 409

    files = client.get(f"/api/threads/{tid}/files").json()
    assert [f["name"] for f in files] == ["notes.txt", "zoo.txt"]
    chunks = client.get(f"/api/threads/{tid}/files/{emb.json()['id']}/chunks").json()
    assert chunks[0]["chunk_index"] == 0

    r = client.post(f"/api/threads/{tid}/chat", json={"message": "What do giraffes eat?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = _events(r.text)
    assert events[0][0] == "citations" and events[0][1][0]["file_name"] == "zoo.txt"
    assert any(e == "token" for e, _ in events)
    assert events[-1][0] == "done" and events[-1][1]["title"] == "What do giraffes eat?"

    thread = client.get(f"/api/threads/{tid}").json()
    assert [m["role"] for m in thread["messages"]] == ["user", "assistant"]
    assert thread["messages"][1]["citations"][0]["file_name"] == "zoo.txt"
    assert thread["title"] == "What do giraffes eat?"

    assert client.delete(f"/api/threads/{tid}/files/{inline.json()['id']}").json() == {"ok": True}
    assert len(client.get(f"/api/threads/{tid}/files").json()) == 1
    assert client.delete(f"/api/threads/{tid}").json() == {"ok": True}
    assert client.get(f"/api/threads/{tid}").status_code == 404


def test_errors(client):
    tid = client.post("/api/threads", json={"title": "x"}).json()["id"]
    assert client.post(f"/api/threads/{tid}/files",
                       files={"file": ("evil.exe", b"MZ", "application/octet-stream")}).status_code == 415
    assert client.post(f"/api/threads/{tid}/chat", json={"message": "  "}).status_code == 400
    assert client.post("/api/threads/nope/chat", json={"message": "hi"}).status_code == 404
    assert client.delete(f"/api/threads/{tid}/files/nope").status_code == 404
    assert client.patch(f"/api/threads/{tid}", json={"title": "New"}).json()["title"] == "New"


def test_settings_roundtrip(client):
    s = client.get("/api/settings").json()
    assert s["chunk_size_chars"] == 512 and s["retrieval_limit"] == 5 and s["retrieval_threshold"] == 0.2
    assert client.put("/api/settings", json={"retrieval_limit": 5}).json()["retrieval_limit"] == 5
    assert client.put("/api/settings", json={"retrieval_limit": 99}).status_code == 400
    assert client.put("/api/settings", json={"overlap_chars": 600}).status_code == 400
    assert client.get("/api/settings").json()["retrieval_limit"] == 5


def test_upload_disabled_and_too_large(client):
    tid = client.post("/api/threads", json={}).json()["id"]
    client.put("/api/settings", json={"max_file_size_mb": 1})
    assert _upload(client, tid, "big.txt", "x" * (1024 * 1024 + 1)).status_code == 413
    client.put("/api/settings", json={"enabled": False})
    assert _upload(client, tid, "a.txt", "hi").status_code == 400
