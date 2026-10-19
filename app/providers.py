"""LLM + embedding providers. Anything OpenAI-compatible works via OPENAI_BASE_URL."""
from __future__ import annotations

import hashlib
import math
import re
from typing import Iterator, Protocol

from .config import Settings


class Provider(Protocol):
    name: str
    chat_model: str
    embed_model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def chat_stream(self, messages: list[dict]) -> Iterator[str]: ...
    def count_tokens(self, text: str) -> int: ...


def approx_tokens(text: str) -> int:
    # ~4 chars per token for English; good enough for the inline-vs-embed decision.
    return max(1, math.ceil(len(text) / 4)) if text else 0


class OpenAIProvider:
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        if not settings.openai_api_key and not settings.openai_base_url:
            raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
        # Explicit default: an empty OPENAI_BASE_URL in .env would otherwise be picked up by the SDK.
        base_url = settings.openai_base_url or "https://api.openai.com/v1"
        self.client = OpenAI(api_key=settings.openai_api_key or "not-needed", base_url=base_url)
        self.chat_model = settings.chat_model
        self.embed_model = settings.embed_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self.client.embeddings.create(model=self.embed_model, input=texts)
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    def chat_stream(self, messages: list[dict]) -> Iterator[str]:
        stream = self.client.chat.completions.create(model=self.chat_model, messages=messages, stream=True)
        for event in stream:
            if event.choices and event.choices[0].delta and event.choices[0].delta.content:
                yield event.choices[0].delta.content

    def count_tokens(self, text: str) -> int:
        return approx_tokens(text)


class FakeProvider:
    """Deterministic offline provider for tests and UI work without an API key."""

    name = "fake"
    chat_model = "fake-chat"
    embed_model = "fake-embed-64"
    dim = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"\w+", t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out

    def chat_stream(self, messages: list[dict]) -> Iterator[str]:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user = messages[-1]["content"] if messages else ""
        reply = f"(fake) You asked: {user}. Context chars: {len(system)}."
        for word in reply.split(" "):
            yield word + " "

    def count_tokens(self, text: str) -> int:
        return approx_tokens(text)


def get_provider(settings: Settings) -> Provider:
    if settings.provider == "fake":
        return FakeProvider()
    return OpenAIProvider(settings)
