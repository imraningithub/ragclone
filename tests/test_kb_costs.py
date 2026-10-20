"""Cost tracking through the HTTP API: ask questions, read GET /api/admin/costs.

Expected dollar amounts are worked out by hand from the published prices (USD per 1M tokens, Standard tier,
docs/research/openai-pricing.md): gpt-4.1-mini 0.40 in / 0.10 cached / 1.60 out, gpt-4.1 2.00 / 0.50 / 8.00,
text-embedding-3-small 0.02 in.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.providers import FakeProvider
from tests.test_kb_api import _seed, _settings

ADMIN = {"X-Admin-Token": "admin-secret"}
Q = "Wann erfolgt die Anmeldung zur Prüfung im Campusportal?"


class UsageProvider(FakeProvider):
    """Fake provider reporting fixed usage: planner 1000 in / 100 out, embeddings 300, answer 2000 in
    (1024 cached) / `answer_output` out."""

    embed_model = "text-embedding-3-small"

    def __init__(self, answer_output=200):
        self.plan_usage = (1000, 0, 100)
        self.embed_tokens = 300
        self.answer_usage = (2000, 1024, answer_output)


def _client(tmp_path, provider=None, **kw):
    settings = _settings(tmp_path, **kw)
    _seed(settings)
    return TestClient(create_app(settings, provider or UsageProvider())), settings


def _ask(client, question=Q):
    tid = client.post("/api/threads", json={}).json()["id"]
    r = client.post(f"/api/threads/{tid}/chat", json={"message": question})
    assert r.status_code == 200
    return r


def test_answered_question_cost_by_step(tmp_path):
    client, _ = _client(tmp_path)
    with client:
        _ask(client)
        costs = client.get("/api/admin/costs?range=all", headers=ADMIN).json()
    assert costs["questions"] == 1
    # planner (1000*0.40 + 100*1.60) = 560e-6; embeddings 300*0.02 = 6e-6;
    # answer (976*2.00 + 1024*0.50 + 200*8.00) = 4064e-6
    assert costs["spend_usd"] == pytest.approx(0.00463)
    assert costs["per_question"]["mean"] == pytest.approx(0.00463)


def test_mean_median_p95_per_question(tmp_path):
    provider = UsageProvider()
    client, _ = _client(tmp_path, provider)
    with client:
        for output in (200, 400, 1000):
            provider.answer_usage = (2000, 1024, output)
            _ask(client)
        costs = client.get("/api/admin/costs?range=all", headers=ADMIN).json()
    assert costs["questions"] == 3
    assert costs["spend_usd"] == pytest.approx(0.02189)
    assert costs["per_question"]["mean"] == pytest.approx(0.0072966667)
    assert costs["per_question"]["median"] == pytest.approx(0.00623)
    assert costs["per_question"]["p95"] == pytest.approx(0.01103)


def _utc(text):
    from datetime import datetime, timezone
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()


def test_ranges_use_berlin_calendar_days(tmp_path):
    now = {"t": 0.0}
    settings = _settings(tmp_path)
    _seed(settings)
    client = TestClient(create_app(settings, UsageProvider(), clock=lambda: now["t"]))
    with client:
        # Berlin is UTC+2 until 2026-10-25.
        for asked_at in ("2026-10-19T22:30",   # 00:30 Berlin on Oct 20: today
                         "2026-10-19T21:30",   # 23:30 Berlin on Oct 19: yesterday
                         "2026-10-13T21:30",   # 23:30 Berlin on Oct 13: 8th day back, outside 7d
                         "2026-09-01T10:00"):  # outside 30d
            now["t"] = _utc(asked_at)
            _ask(client)
        now["t"] = _utc("2026-10-20T10:00")
        counts = {r: client.get(f"/api/admin/costs?range={r}", headers=ADMIN).json()["questions"]
                  for r in ("today", "7d", "30d", "all")}
        assert counts == {"today": 1, "7d": 2, "30d": 3, "all": 4}
        assert client.get("/api/admin/costs?range=decade", headers=ADMIN).status_code == 400


def test_cost_api_requires_admin_token(tmp_path):
    client, _ = _client(tmp_path)
    with client:
        assert client.get("/api/admin/costs").status_code == 403
        assert client.get("/api/admin/costs", headers={"X-Admin-Token": "wrong"}).status_code == 403
        assert client.get("/api/admin/costs", headers=ADMIN).status_code == 200
        page = client.get("/admin/costs")
        assert page.status_code == 200 and "admin-secret" not in page.text


def test_model_missing_from_price_table_is_unknown_not_free(tmp_path):
    provider = UsageProvider()
    provider.embed_model = "fake-embed-64"  # not in the price table
    client, _ = _client(tmp_path, provider)
    with client:
        _ask(client)
        costs = client.get("/api/admin/costs?range=all", headers=ADMIN).json()
    assert costs["unknown_cost_questions"] == 1
    assert costs["spend_usd"] == pytest.approx(0.00056 + 0.004064)  # the known steps only


def test_concurrent_questions_keep_their_own_usage(tmp_path):
    import threading

    class SlowProvider(UsageProvider):
        def __init__(self):
            super().__init__()
            self.barrier = threading.Barrier(2, timeout=10)

        def complete(self, messages, **kw):
            if kw.get("json_mode"):
                self.barrier.wait()  # both questions are in flight at the same time
            return super().complete(messages, **kw)

        def chat_stream(self, messages, temperature=None, model=None, usage=None):
            # Answer size depends on the question, so mixed-up usage would show in the per-question costs.
            # Kept local: shared provider state would race between the two threads.
            output = 1000 if "lange" in messages[-1]["content"] else 200
            yield from super().chat_stream(messages, temperature=temperature, model=model)
            usage(model, 2000, 1024, output)

    provider = SlowProvider()
    client, _ = _client(tmp_path, provider)
    errors = []

    def ask(question):
        try:
            with TestClient(client.app) as c:
                _ask(c, question)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    with client:
        threads = [threading.Thread(target=ask, args=(q,)) for q in
                   ("Prüfung Anmeldung kurze Frage", "Prüfung Anmeldung lange Frage")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert not errors
        costs = client.get("/api/admin/costs?range=all", headers=ADMIN).json()
    assert costs["questions"] == 2
    assert costs["per_question"]["p95"] == pytest.approx(0.01103)  # the long answer alone
    assert costs["spend_usd"] == pytest.approx(0.00463 + 0.01103)


def test_request_log_has_usage_and_cost(tmp_path):
    import json

    client, settings = _client(tmp_path)
    with client:
        _ask(client)
    rec = json.loads(settings.kb_log_path.read_text("utf-8").splitlines()[-1])
    assert [c["step"] for c in rec["usage"]] == ["planner", "embeddings", "answer"]
    assert rec["usage"][2] | {"usd": None} == {"step": "answer", "model": "gpt-4.1", "input": 2000,
                                               "cached": 1024, "output": 200, "usd": None, "estimated": False}
    assert rec["cost_usd"] == pytest.approx(0.00463)
