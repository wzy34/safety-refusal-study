"""Offline test double: a scripted `Transport` behind the real `ChatClient`, so tests exercise
the real adapter, cache, limiter, retry and parsing code without network or API keys."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from typing import Any

from srs.config import StudyConfig
from srs.llm.cache import RequestCache
from srs.llm.client import ChatClient
from srs.llm.errors import ProviderError

# An outcome is a raw response dict (returned) or a ProviderError (raised).
Outcome = dict[str, Any] | ProviderError
Responder = Callable[[str, dict[str, Any]], Outcome]


def fake_response(text: str | None = "OK", *, reasoning: str | None = None,
                  finish_reason: str = "stop", model: str = "fake-model",
                  prompt_tokens: int = 10, completion_tokens: int = 20) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "id": "fake", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


def rate_limit_error() -> ProviderError:
    return ProviderError(status_code=429, code="rate_limit_exceeded", message="Rate limit")


def server_error() -> ProviderError:
    return ProviderError(status_code=503, message="Service unavailable")


def block_error(code: str = "1301", message: str = "content flagged as unsafe") -> ProviderError:
    return ProviderError(status_code=400, code=code, message=message)


def fatal_error() -> ProviderError:
    return ProviderError(status_code=401, code="invalid_api_key", message="Unauthorized")


def last_user(body: dict[str, Any]) -> str:
    return next((m["content"] for m in reversed(body["messages"]) if m["role"] == "user"), "")


def echo_responder(provider: str, body: dict[str, Any]) -> Outcome:
    return fake_response(f"Fake reply to: {last_user(body)[:30]}", model=body["model"])


class FakeTransport:
    """Outcomes come from `queue` first (in order), then from `responder`.

    `calls` records every (provider, body) actually sent, so tests can assert on request
    content and on how many network calls were made.
    """

    def __init__(self, responder: Responder | None = None, queue: list[Outcome] | None = None,
                 delay_s: float = 0.0):
        self.responder = responder or echo_responder
        self.queue: deque[Outcome] = deque(queue or [])
        self.delay_s = delay_s
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def create(self, provider, body, extra_body, timeout):
        self.calls.append((provider, body))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        outcome = self.queue.popleft() if self.queue else self.responder(provider, body)
        if isinstance(outcome, ProviderError):
            raise outcome
        return outcome


def make_fake_client(cfg: StudyConfig, responder: Responder | None = None,
                     queue: list[Outcome] | None = None, cache: RequestCache | None = None,
                     delay_s: float = 0.0) -> ChatClient:
    transport = FakeTransport(responder, queue, delay_s)
    return ChatClient(cfg, transport, cache if cache is not None else RequestCache(":memory:"),
                      backoff_initial=0.0, backoff_max=0.0)
