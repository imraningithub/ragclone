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

    def __init__(self, answer_output=200, answer_text=None):
        self.plan_usage = (1000, 0, 100)
        self.embed_tokens = 300
        self.answer_usage = (2000, 1024, answer_output)
        self.answer_text = answer_text  # fixed model answer instead of the fake one

    def _complete(self, messages, json_mode):
        if self.answer_text is not None and not json_mode:
            return self.answer_text
        return super()._complete(messages, json_mode)


PLAN_AND_EMBED_USD = 0.00056 + 0.000006  # planner (1000*0.40 + 100*1.60)e-6, embeddings 300*0.02e-6
ANSWER_USD = 0.004064                    # answer (976*2.00 + 1024*0.50 + 200*8.00)e-6


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


def _by_reason(costs):
    return {r["reason"]: (r["questions"], pytest.approx(r["spend_usd"])) for r in costs["by_reason"]}


@pytest.mark.parametrize("answer_text,reason", [
    ("NO_ANSWER", "no_answer"),
    ("Die Anmeldung erfolgt online, ganz sicher.", "uncited"),
])
def test_refusals_after_the_answer_call_are_recorded_by_reason(tmp_path, answer_text, reason):
    client, _ = _client(tmp_path, UsageProvider(answer_text=answer_text))
    with client:
        _ask(client)
        _ask(client, "Wann ist die Prüfung?")
        costs = client.get("/api/admin/costs?range=all", headers=ADMIN).json()
    assert _by_reason(costs) == {reason: (2, 2 * (PLAN_AND_EMBED_USD + ANSWER_USD))}


def test_no_context_refusal_costs_planner_and_embeddings_only(tmp_path):
    client, _ = _client(tmp_path, kb_threshold=0.9)
    with client:
        _ask(client, "zebra banana trumpet volcano")
        costs = client.get("/api/admin/costs?range=all", headers=ADMIN).json()
    assert _by_reason(costs) == {"no_context": (1, PLAN_AND_EMBED_USD)}


def _chat_then_disconnect(app, tid, cookie, question, *, send_raises):
    """Drive one chat request through the ASGI interface (spec 2.3, as uvicorn) and close the "browser tab"
    right after the first answer text arrives. send_raises=True: later writes fail with OSError (uvicorn's
    ClientDisconnected is one), which cuts the stream. False: later writes are silently dropped, which is what
    uvicorn does once it has seen the disconnect."""
    import json

    import anyio

    async def run():
        gone = False
        request_sent = False
        statuses = []

        async def receive():
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": json.dumps({"message": question}).encode(),
                        "more_body": False}
            while not gone:
                await anyio.sleep(0.01)
            return {"type": "http.disconnect"}

        async def send(message):
            nonlocal gone
            if message["type"] == "http.response.start":
                statuses.append(message["status"])
            elif gone and send_raises:
                raise OSError("client disconnected")
            elif b"event: token" in message.get("body", b""):
                gone = True

        path = f"/api/threads/{tid}/chat"
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
                 "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
                 "root_path": "", "client": ("127.0.0.1", 50000), "server": ("testserver", 80),
                 "headers": [(b"content-type", b"application/json"), (b"cookie", f"dtm_client={cookie}".encode())]}
        with anyio.fail_after(10):
            try:
                await app(scope, receive, send)
            except OSError:
                pass  # the server's problem, not the app's
        return statuses

    return anyio.run(run)


ANSWER = "Online im Campusportal [1] bis 15. Mai."


def test_stream_cut_mid_answer_is_recorded_as_aborted_with_estimated_tokens(tmp_path):
    import json

    client, settings = _client(tmp_path, UsageProvider(answer_text=ANSWER))
    with client:
        tid = client.post("/api/threads", json={}).json()["id"]
        _chat_then_disconnect(client.app, tid, client.cookies.get("dtm_client"), Q, send_raises=True)
        costs = client.get("/api/admin/costs?range=all", headers=ADMIN).json()
        assert client.get(f"/api/threads/{tid}").json()["messages"] == []  # nothing saved to the chat
    assert [(r["reason"], r["questions"]) for r in costs["by_reason"]] == [("aborted", 1)]
    assert costs["estimated_questions"] == 1
    rec = json.loads(settings.kb_log_path.read_text("utf-8").splitlines()[-1])
    answer = rec["usage"][-1]
    assert rec["reason"] == "aborted" and answer["step"] == "answer" and answer["estimated"] is True
    # ~4 characters per token of the text generated before the cut: at least "Online im Campusportal [1]"
    # (26 chars = 7 tokens); the server may have pulled a word or two more before it noticed.
    assert 7 <= answer["output"] <= 10
    assert answer["input"] > 1000  # estimated from the prompt (system prompt + excerpts)
    assert rec["cost_usd"] > PLAN_AND_EMBED_USD


def test_disconnect_under_uvicorn_finishes_the_answer_with_exact_usage(tmp_path):
    client, _ = _client(tmp_path, UsageProvider(answer_text=ANSWER))
    with client:
        tid = client.post("/api/threads", json={}).json()["id"]
        assert _chat_then_disconnect(client.app, tid, client.cookies.get("dtm_client"), Q,
                                     send_raises=False) == [200]
        costs = client.get("/api/admin/costs?range=all", headers=ADMIN).json()
    assert _by_reason(costs) == {"answered": (1, PLAN_AND_EMBED_USD + ANSWER_USD)}
    assert costs["estimated_questions"] == 0


class BrokenStreamProvider(UsageProvider):
    """The answer stream fails after its first words; OpenAI then never reports usage."""

    def chat_stream(self, messages, temperature=None, model=None, usage=None):
        yield "Online im Campusportal [1]"
        raise RuntimeError("connection reset")


def test_model_error_mid_stream_is_recorded_as_error(tmp_path):
    client, _ = _client(tmp_path, BrokenStreamProvider())
    with client:
        tid = client.post("/api/threads", json={}).json()["id"]
        body = client.post(f"/api/threads/{tid}/chat", json={"message": Q}).text
        assert "event: error" in body
        costs = client.get("/api/admin/costs?range=all", headers=ADMIN).json()
    assert [(r["reason"], r["questions"]) for r in costs["by_reason"]] == [("error", 1)]
    assert costs["estimated_questions"] == 1


def test_failure_before_streaming_records_the_calls_already_billed(tmp_path):
    class EmbedFails(UsageProvider):
        def embed(self, texts, usage=None):
            raise RuntimeError("embeddings down")

    class FirstChunkFails(UsageProvider):
        def chat_stream(self, messages, temperature=None, model=None, usage=None):
            raise RuntimeError("answer model down")
            yield  # pragma: no cover

    for provider, spend in ((EmbedFails(), 0.00056), (FirstChunkFails(), PLAN_AND_EMBED_USD)):
        client, _ = _client(tmp_path / type(provider).__name__, provider)
        with client:
            tid = client.post("/api/threads", json={}).json()["id"]
            assert client.post(f"/api/threads/{tid}/chat", json={"message": Q}).status_code == 502
            costs = client.get("/api/admin/costs?range=all", headers=ADMIN).json()
        assert _by_reason(costs) == {"error": (1, spend)}, type(provider).__name__


def test_rejected_requests_cost_nothing_and_follow_ups_count_individually(tmp_path):
    client, _ = _client(tmp_path, rate_limit_per_min=2)
    with client:
        tid = client.post("/api/threads", json={}).json()["id"]
        assert client.post(f"/api/threads/{tid}/chat", json={"message": "   "}).status_code == 400
        assert client.post(f"/api/threads/{tid}/chat", json={"message": Q}).status_code == 200
        assert client.post(f"/api/threads/{tid}/chat", json={"message": "Und bis wann?"}).status_code == 200
        assert client.post(f"/api/threads/{tid}/chat", json={"message": Q}).status_code == 429
        costs = client.get("/api/admin/costs?range=all", headers=ADMIN).json()
    assert costs["questions"] == 2  # the question and its follow-up in the same chat


# -- #5: where the money goes ---------------------------------------------------------------------------------

def test_cost_by_step_and_by_model(tmp_path):
    client, _ = _client(tmp_path)
    with client:
        _ask(client)
        _ask(client)
        costs = client.get("/api/admin/costs?range=all", headers=ADMIN).json()
    by_step = {r["step"]: r for r in costs["by_step"]}
    assert by_step["planner"] == {"step": "planner", "calls": 2, "input": 2000, "cached": 0, "output": 200,
                                  "spend_usd": pytest.approx(2 * 0.00056)}
    assert by_step["embeddings"] == {"step": "embeddings", "calls": 2, "input": 600, "cached": 0, "output": 0,
                                     "spend_usd": pytest.approx(2 * 0.000006)}
    assert by_step["answer"] == {"step": "answer", "calls": 2, "input": 4000, "cached": 2048, "output": 400,
                                 "spend_usd": pytest.approx(2 * ANSWER_USD)}
    assert [r["step"] for r in costs["by_step"]] == ["answer", "planner", "embeddings"]  # most expensive first
    by_model = {r["model"]: r["spend_usd"] for r in costs["by_model"]}
    assert by_model == {"gpt-4.1": pytest.approx(2 * ANSWER_USD), "gpt-4.1-mini": pytest.approx(2 * 0.00056),
                        "text-embedding-3-small": pytest.approx(2 * 0.000006)}


def _clocked(tmp_path, provider=None, **kw):
    now = {"t": 0.0}
    settings = _settings(tmp_path, **kw)
    _seed(settings)
    return TestClient(create_app(settings, provider or UsageProvider(), clock=lambda: now["t"])), now


def test_daily_series_per_step_in_berlin_days(tmp_path):
    client, now = _clocked(tmp_path)
    with client:
        for asked_at in ("2026-10-19T08:00", "2026-10-20T07:00", "2026-10-20T08:00"):
            now["t"] = _utc(asked_at)
            _ask(client)
        now["t"] = _utc("2026-10-20T10:00")
        daily = client.get("/api/admin/costs?range=7d", headers=ADMIN).json()["daily"]
    assert [d["date"] for d in daily] == [f"2026-10-{n}" for n in range(14, 21)]  # every day, oldest first
    assert daily[0] == {"date": "2026-10-14", "planner": 0, "embeddings": 0, "answer": 0, "total": 0}
    assert daily[5]["answer"] == pytest.approx(ANSWER_USD)
    assert daily[6]["planner"] == pytest.approx(2 * 0.00056)
    assert daily[6]["total"] == pytest.approx(2 * (PLAN_AND_EMBED_USD + ANSWER_USD))


def test_cache_share_of_chat_input(tmp_path):
    client, _ = _client(tmp_path)
    with client:
        _ask(client)
        cache = client.get("/api/admin/costs?range=all", headers=ADMIN).json()["cache"]
    # planner 1000 in / 0 cached + answer 2000 in / 1024 cached; embeddings can't be cached
    assert cache == {"input": 3000, "cached": 1024, "share": pytest.approx(1024 / 3000)}


def test_semester_range_starts_at_configured_date(tmp_path):
    client, now = _clocked(tmp_path, semester_start="2026-10-01")
    with client:
        for asked_at in ("2026-09-30T21:30",    # 23:30 Berlin on Sep 30: before the semester
                         "2026-09-30T22:30"):   # 00:30 Berlin on Oct 1: first semester minute
            now["t"] = _utc(asked_at)
            _ask(client)
        now["t"] = _utc("2026-10-20T10:00")
        assert client.get("/api/admin/costs?range=semester", headers=ADMIN).json()["questions"] == 1


def test_semester_range_without_start_date_is_rejected(tmp_path):
    client, _ = _client(tmp_path)
    with client:
        r = client.get("/api/admin/costs?range=semester", headers=ADMIN)
    assert r.status_code == 400 and "SEMESTER_START" in r.json()["detail"]


def _with_pricing_file(tmp_path, content, provider=None):
    settings = _settings(tmp_path)
    _seed(settings)
    (settings.data_dir / "pricing.json").write_text(content, "utf-8")
    return TestClient(create_app(settings, provider or UsageProvider()))


def test_pricing_file_overrides_and_extends_the_price_table(tmp_path):
    import json

    provider = UsageProvider()
    provider.embed_model = "fake-embed-64"  # unknown to the built-in table, added by the file
    client = _with_pricing_file(tmp_path, json.dumps({
        "as_of": "2026-11-01", "source": "OpenAI invoice November",
        "models": {"gpt-4.1": {"input": 4.0, "cached_input": 1.0, "output": 16.0},
                   "fake-embed-64": {"input": 0.05}}}), provider)
    with client:
        _ask(client)
        costs = client.get("/api/admin/costs?range=all", headers=ADMIN).json()
    by_model = {r["model"]: r["spend_usd"] for r in costs["by_model"]}
    assert by_model["gpt-4.1"] == pytest.approx(0.008128)       # (976*4 + 1024*1 + 200*16)e-6
    assert by_model["gpt-4.1-mini"] == pytest.approx(0.00056)    # built-in price unchanged
    assert by_model["fake-embed-64"] == pytest.approx(300 * 0.05e-6)
    assert costs["unknown_cost_questions"] == 0
    assert costs["pricing"]["as_of"] == "2026-11-01" and costs["pricing"]["source"] == "OpenAI invoice November"


def test_broken_pricing_file_keeps_built_in_prices_and_says_so(tmp_path):
    client = _with_pricing_file(tmp_path, "{not json")
    with client:
        _ask(client)
        costs = client.get("/api/admin/costs?range=all", headers=ADMIN).json()
    assert costs["spend_usd"] == pytest.approx(PLAN_AND_EMBED_USD + ANSWER_USD)
    assert costs["pricing"]["as_of"] == "2026-10-20" and "pricing.json" in costs["pricing"]["override_error"]


# -- #6: most expensive questions, top visitors, 30-day purge -----------------------------------------------

def test_top_questions_most_expensive_first_with_step_breakdown(tmp_path):
    provider = UsageProvider()
    client, _ = _client(tmp_path, provider)
    with client:
        for output, question in ((200, "kurz"), (1000, "lang"), (400, "mittel")):
            provider.answer_usage = (2000, 1024, output)
            _ask(client, f"Prüfung Anmeldung {question}")
        top = client.get("/api/admin/costs?range=all", headers=ADMIN).json()["top_questions"]
    assert [t["question"] for t in top] == ["Prüfung Anmeldung lang", "Prüfung Anmeldung mittel",
                                            "Prüfung Anmeldung kurz"]
    first = top[0]
    assert first["reason"] == "answered" and first["spend_usd"] == pytest.approx(0.01103)
    assert (first["input"], first["cached"], first["output"]) == (3300, 1024, 1100)
    assert first["steps"] == {"planner": pytest.approx(0.00056), "embeddings": pytest.approx(0.000006),
                              "answer": pytest.approx(0.010464)}
    assert first["asked_at"].endswith(("+02:00", "+01:00"))  # Europe/Berlin


def test_top_questions_capped_at_ten(tmp_path):
    client, _ = _client(tmp_path)
    with client:
        for _ in range(11):
            _ask(client)
        costs = client.get("/api/admin/costs?range=all", headers=ADMIN).json()
    assert costs["questions"] == 11 and len(costs["top_questions"]) == 10


def test_top_visitors_are_anonymous_and_ranked_by_spend(tmp_path):
    client, _ = _client(tmp_path)
    with client:
        _ask(client)
        with TestClient(client.app) as other:  # a second browser
            _ask(other)
            _ask(other)
        visitors = client.get("/api/admin/costs?range=all", headers=ADMIN).json()["top_visitors"]
    assert [v["questions"] for v in visitors] == [2, 1]
    assert visitors[0]["spend_usd"] == pytest.approx(2 * 0.00463)
    ids = [v["visitor"] for v in visitors]
    assert all(len(i) == 8 for i in ids) and ids[0] != ids[1]
    assert client.cookies.get("dtm_client")[:8] not in ids  # never the cookie itself


DAY = 86400.0


def _texts(client):
    top = client.get("/api/admin/costs?range=all", headers=ADMIN).json()["top_questions"]
    return {t["id"]: t["question"] for t in top}


def test_question_text_purged_after_30_days_at_most_once_a_day(tmp_path):
    client, now = _clocked(tmp_path)
    with client:
        now["t"] = 3600.0                      # A at 01:00 on day 0
        _ask(client, "Frage A Prüfung")
        now["t"] = 30 * DAY + 1800             # B: purge runs, A is 29 d 23.5 h old -> kept
        _ask(client, "Frage B Prüfung")
        assert "Frage A Prüfung" in _texts(client).values()
        now["t"] = 30 * DAY + 7200             # C: A is now over 30 days old, but the last purge was 1.5 h ago
        _ask(client, "Frage C Prüfung")
        assert "Frage A Prüfung" in _texts(client).values()
        spend_before = client.get("/api/admin/costs?range=all", headers=ADMIN).json()["spend_usd"]
        now["t"] = 31 * DAY + 3600             # D: a day after the last purge -> A's text goes
        _ask(client, "Frage D Prüfung")
        texts = _texts(client)
        costs = client.get("/api/admin/costs?range=all", headers=ADMIN).json()
    assert sorted(t for t in texts.values() if t) == ["Frage B Prüfung", "Frage C Prüfung", "Frage D Prüfung"]
    assert list(texts.values()).count(None) == 1
    assert costs["questions"] == 4  # the row, tokens and cost stay
    assert costs["spend_usd"] == pytest.approx(spend_before + 0.00463)


def test_question_text_purged_on_app_start(tmp_path):
    now = {"t": 0.0}
    settings = _settings(tmp_path)
    _seed(settings)
    with TestClient(create_app(settings, UsageProvider(), clock=lambda: now["t"])) as first:
        _ask(first, "Alte Frage Prüfung")
    now["t"] = 31 * DAY
    with TestClient(create_app(settings, UsageProvider(), clock=lambda: now["t"])) as restarted:
        assert list(_texts(restarted).values()) == [None]  # gone before any new question arrives
