from dataclasses import replace
from pathlib import Path

import pytest

from app import rag
from app.config import RagSettings, Settings
from app.providers import FakeProvider
from app.store import VectorStore


@pytest.fixture
def env(tmp_path: Path):
    settings = Settings(provider="fake", openai_api_key=None, openai_base_url=None, chat_model="fake-chat",
                        embed_model="fake-embed-64", context_window=1000, data_dir=tmp_path)
    store = VectorStore(settings.db_path)
    yield settings, store, FakeProvider()
    store.close()


def _txt(tmp_path: Path, name: str, text: str) -> str:
    p = tmp_path / name
    p.write_text(text, "utf-8")
    return str(p)


def test_decide_mode_uses_context_ratio(env):
    settings, _, provider = env
    rs = RagSettings(auto_inline_context_ratio=0.5)  # budget = 500 tokens ≈ 2000 chars
    assert rag.decide_mode("a" * 1900, "auto", provider, settings, rs) == "inline"
    assert rag.decide_mode("a" * 2100, "auto", provider, settings, rs) == "embeddings"
    assert rag.decide_mode("a" * 10, "embeddings", provider, settings, rs) == "embeddings"
    assert rag.decide_mode("a" * 99999, "inline", provider, settings, rs) == "inline"
    assert rag.decide_mode("a" * 10, "prompt", provider, settings, rs) == "inline"


def test_ingest_inline_and_embeddings(env, tmp_path):
    settings, store, provider = env
    rs = RagSettings()
    coll = rag.collection_for_thread("t1")
    small = rag.ingest_file(store, provider, settings, rs, collection=coll,
                            path=_txt(tmp_path, "a.txt", "Cats purr."), name="a.txt", file_type="text/plain")
    assert small["mode"] == "inline" and small["chunk_count"] == 0

    big_text = "\n\n".join(f"Paragraph {i} talks about topic number {i}." for i in range(200))
    big = rag.ingest_file(store, provider, settings, rs, collection=coll, path=_txt(tmp_path, "b.txt", big_text),
                          name="b.txt", file_type="text/plain", parse_mode="embeddings")
    assert big["mode"] == "embeddings" and big["chunk_count"] > 1
    assert store.get_collection(coll)["dim"] == FakeProvider.dim


def test_ingest_rejects_large_file(env, tmp_path):
    settings, store, provider = env
    path = _txt(tmp_path, "big.txt", "x" * (1024 * 1024 + 10))
    with pytest.raises(rag.FileTooLarge):
        rag.ingest_file(store, provider, settings, RagSettings(max_file_size_mb=1), collection="c",
                        path=path, name="big.txt", file_type="text/plain")


def test_build_messages_inline_and_citations(env, tmp_path):
    settings, store, provider = env
    rs = RagSettings(retrieval_threshold=0.0)
    coll = rag.collection_for_thread("t2")
    rag.ingest_file(store, provider, settings, rs, collection=coll,
                    path=_txt(tmp_path, "note.txt", "The password hint is blue whale."),
                    name="note.txt", file_type="text/plain", parse_mode="inline")
    rag.ingest_file(store, provider, settings, rs, collection=coll,
                    path=_txt(tmp_path, "zoo.txt", "Giraffes eat acacia leaves. Penguins eat fish."),
                    name="zoo.txt", file_type="text/plain", parse_mode="embeddings")

    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    messages, cites = rag.build_messages(store, provider, settings, rs, collection=coll,
                                         history=history, user_message="What do giraffes eat?")
    system = messages[0]["content"]
    assert messages[0]["role"] == "system"
    assert '<document name="note.txt">' in system and "blue whale" in system
    assert "[1] (zoo.txt, chunk 0)" in system
    assert cites[0]["n"] == 1 and cites[0]["file_name"] == "zoo.txt"
    assert [m["role"] for m in messages[1:]] == ["user", "assistant", "user"]


def test_build_messages_disabled_adds_no_context(env, tmp_path):
    settings, store, provider = env
    coll = rag.collection_for_thread("t3")
    rag.ingest_file(store, provider, settings, RagSettings(), collection=coll,
                    path=_txt(tmp_path, "n.txt", "secret"), name="n.txt", file_type="text/plain")
    messages, cites = rag.build_messages(store, provider, settings, replace(RagSettings(), enabled=False),
                                         collection=coll, history=[], user_message="secret?")
    assert "secret" not in messages[0]["content"] and cites == []
