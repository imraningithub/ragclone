import json

from app.providers import FakeProvider


def test_fake_complete_json_mode():
    p = FakeProvider()
    out = p.complete([{"role": "system", "content": "plan"},
                      {"role": "user", "content": "<question>\nHow do I register for exams?\n</question>"}],
                     json_mode=True, temperature=0)
    assert json.loads(out) == {"standalone": "How do I register for exams?", "language": "en",
                               "queries_de": ["How do I register for exams?"],
                               "queries_en": ["How do I register for exams?"]}
    out = json.loads(p.complete([{"role": "user", "content": "Wann ist die Prüfung?"}], json_mode=True))
    assert out["language"] == "de" and out["standalone"] == "Wann ist die Prüfung?"


def test_fake_complete_answer_modes():
    p = FakeProvider()
    with_ctx = [{"role": "system", "content": 'rules <excerpt id="1" doc="a">x</excerpt>'},
                {"role": "user", "content": "Frage?"}]
    assert "[1]" in p.complete(with_ctx, temperature=0.1)
    assert p.complete([{"role": "system", "content": "rules"}, {"role": "user", "content": "Frage?"}]) == "NO_ANSWER"


def test_fake_chat_stream_with_temperature():
    p = FakeProvider()
    msgs = [{"role": "user", "content": "hi"}]
    assert "".join(p.chat_stream(msgs, temperature=0.2)) == "".join(p.chat_stream(msgs))
    assert "hi" in "".join(p.chat_stream(msgs, 0.0))
