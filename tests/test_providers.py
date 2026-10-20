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


# -- OpenAIProvider usage reporting, against a stubbed SDK client --------------------------------------------
from types import SimpleNamespace as NS  # noqa: E402

from app.config import Settings  # noqa: E402
from app.providers import OpenAIProvider  # noqa: E402


class _Create:
    """Stands in for client.chat.completions / client.embeddings: records kwargs, returns a canned response."""

    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _openai(chat=None, embeddings=None):
    p = OpenAIProvider(Settings(provider="openai", openai_api_key="test-key", openai_base_url=None,
                                chat_model="gpt-4.1-mini", embed_model="text-embedding-3-small",
                                context_window=128000, data_dir=None))
    p.client = NS(chat=NS(completions=chat), embeddings=embeddings)
    return p


def _recorder():
    calls = []
    return calls, lambda model, input_tokens=0, cached_tokens=0, output_tokens=0: calls.append(
        (model, input_tokens, cached_tokens, output_tokens))


def _completion(text, usage):
    return NS(choices=[NS(message=NS(content=text))], usage=usage)


def test_openai_complete_records_usage_with_cached_tokens():
    # Example numbers from OpenAI's prompt-caching cookbook: 1548 prompt tokens, 1280 of them cached.
    chat = _Create(_completion("{}", NS(prompt_tokens=1548, completion_tokens=200, total_tokens=1748,
                                       prompt_tokens_details=NS(cached_tokens=1280))))
    calls, usage = _recorder()
    assert _openai(chat).complete([{"role": "user", "content": "q"}], json_mode=True, model="gpt-4.1-mini",
                                  usage=usage) == "{}"
    assert calls == [("gpt-4.1-mini", 1548, 1280, 200)]


def test_openai_complete_missing_cache_details_count_as_zero():
    for details in (None, NS(cached_tokens=None)):
        chat = _Create(_completion("ok", NS(prompt_tokens=50, completion_tokens=5, total_tokens=55,
                                           prompt_tokens_details=details)))
        calls, usage = _recorder()
        _openai(chat).complete([{"role": "user", "content": "q"}], model="gpt-4.1", usage=usage)
        assert calls == [("gpt-4.1", 50, 0, 5)]


def _chunk(text):
    return NS(choices=[NS(delta=NS(content=text))], usage=None)


def test_openai_stream_records_usage_from_final_chunk():
    final = NS(choices=[], usage=NS(prompt_tokens=2000, completion_tokens=3, total_tokens=2003,
                                    prompt_tokens_details=NS(cached_tokens=1024)))
    chat = _Create(iter([_chunk("Die "), _chunk("Antwort"), _chunk(" [1]"), final]))
    calls, usage = _recorder()
    text = "".join(_openai(chat).chat_stream([{"role": "user", "content": "q"}], model="gpt-4.1", usage=usage))
    assert text == "Die Antwort [1]"
    assert calls == [("gpt-4.1", 2000, 1024, 3)]
    assert chat.kwargs["stream"] is True and chat.kwargs["stream_options"] == {"include_usage": True}


def test_openai_stream_without_usage_chunk_records_nothing():
    # An interrupted stream may never deliver the usage chunk; the caller then estimates.
    chat = _Create(iter([_chunk("Die "), _chunk("Antwort")]))
    calls, usage = _recorder()
    assert "".join(_openai(chat).chat_stream([{"role": "user", "content": "q"}], usage=usage)) == "Die Antwort"
    assert calls == []


def test_openai_embed_records_usage():
    emb = _Create(NS(data=[NS(index=1, embedding=[0.0, 1.0]), NS(index=0, embedding=[1.0, 0.0])],
                     usage=NS(prompt_tokens=12, total_tokens=12)))
    calls, usage = _recorder()
    assert _openai(embeddings=emb).embed(["a", "b"], usage=usage) == [[1.0, 0.0], [0.0, 1.0]]
    assert calls == [("text-embedding-3-small", 12, 0, 0)]


def test_openai_calls_without_usage_are_unchanged():
    chat = _Create(iter([_chunk("hi")]))
    assert "".join(_openai(chat).chat_stream([{"role": "user", "content": "q"}])) == "hi"
    assert "stream_options" not in chat.kwargs
    chat = _Create(_completion("ok", NS(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                                        prompt_tokens_details=None)))
    assert _openai(chat).complete([{"role": "user", "content": "q"}]) == "ok"
    emb = _Create(NS(data=[NS(index=0, embedding=[1.0])], usage=NS(prompt_tokens=1, total_tokens=1)))
    assert _openai(embeddings=emb).embed(["a"]) == [[1.0]]
