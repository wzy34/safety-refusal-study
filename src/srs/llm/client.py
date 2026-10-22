"""The async chat client (design D1, D3).

`ChatClient.complete()` = adapter -> cache lookup -> per-provider limiter -> retries with
backoff -> parse into a `Completion`. Network access sits behind a `Transport`, so the same
code path runs against the real APIs (`ProviderTransport`, which picks `OpenAITransport` or
`AnthropicTransport` per provider) or scripted fakes (`srs.llm.fake`).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Literal, Protocol

from pydantic import BaseModel
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from srs.config import ConfigError, StudyConfig, get_secret, require_provider_keys
from srs.llm.adapters import GenParams, StructuredSpec, build_request
from srs.llm.cache import RequestCache, request_hash
from srs.llm.errors import (
    ProviderError,
    block_code,
    classify_error,
    from_openai_exception,
    is_blocked_finish,
)
from srs.llm.ratelimit import ProviderLimiter

log = logging.getLogger(__name__)

_THINK = re.compile(r"^\s*<think>(.*?)</think>\s*", re.DOTALL)

# The SDK `timeout` bounds each network read, not the whole call, so a connection the server
# holds open without sending anything never times out (seen with Bailian and Zhipu). Each
# attempt therefore also gets a hard total deadline of `timeout_s * DEADLINE_FACTOR`; the
# slowest successful call in the pilot and main runs took 174 s.
DEADLINE_FACTOR = 3


class Completion(BaseModel):
    status: Literal["ok", "api_blocked", "error"]
    text: str | None = None
    reasoning: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] = {}
    latency_s: float | None = None
    model_version: str | None = None
    block_code: str | None = None
    error: str | None = None
    dropped_params: list[str] = []
    structured_mode: str | None = None
    request_hash: str = ""
    cached: bool = False
    raw: dict[str, Any] | None = None


class Transport(Protocol):
    async def create(self, provider: str, body: dict[str, Any], extra_body: dict[str, Any],
                     timeout: float) -> dict[str, Any]:
        """Send one Chat Completions request; return the raw response as a dict, or raise
        `ProviderError`."""
        ...


class OpenAITransport:
    """Real transport: one `AsyncOpenAI` per provider, SDK retries disabled (we retry)."""

    def __init__(self, cfg: StudyConfig):
        self.cfg = cfg
        self._clients: dict[str, Any] = {}

    def _client(self, provider: str):
        if provider not in self._clients:
            from openai import AsyncOpenAI

            pc = self.cfg.providers[provider]
            self._clients[provider] = AsyncOpenAI(api_key=get_secret(pc.api_key_env),
                                                  base_url=pc.base_url, max_retries=0)
        return self._clients[provider]

    async def create(self, provider, body, extra_body, timeout):
        try:
            resp = await self._client(provider).chat.completions.create(
                **body, extra_body=extra_body or None, timeout=timeout)
        except Exception as e:  # normalized and classified by the caller
            raise from_openai_exception(e) from e
        return resp.model_dump(mode="json")


class ProviderTransport:
    """Routes each request to the transport for its provider's `api` (providers.yaml)."""

    def __init__(self, cfg: StudyConfig):
        from srs.llm.anthropic_api import AnthropicTransport

        self.cfg = cfg
        self._by_api: dict[str, Transport] = {"openai": OpenAITransport(cfg),
                                              "anthropic": AnthropicTransport(cfg)}

    async def create(self, provider, body, extra_body, timeout):
        transport = self._by_api[self.cfg.providers[provider].api]
        return await transport.create(provider, body, extra_body, timeout)


def split_reasoning(message: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (final text, reasoning). Reasoning comes from `reasoning_content`/`reasoning`
    fields or a leading <think> block, and is removed from the final text."""
    text = message.get("content")
    if text is None and message.get("refusal"):
        text = message["refusal"]
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(text, str) and (m := _THINK.match(text)):
        reasoning = reasoning or m.group(1).strip()
        text = text[m.end():]
    return text, reasoning


class ChatClient:
    def __init__(self, cfg: StudyConfig, transport: Transport, cache: RequestCache | None, *,
                 backoff_initial: float = 1.0, backoff_max: float = 60.0):
        self.cfg = cfg
        self.transport = transport
        self.cache = cache
        self.backoff_initial = backoff_initial
        self.backoff_max = backoff_max
        self._limiters: dict[str, ProviderLimiter] = {}

    @classmethod
    def real(cls, cfg: StudyConfig, model_ids: list[str]) -> ChatClient:
        """Client for real API calls. Fails before any request if a model id is still a
        placeholder or a needed API key is missing."""
        unset = [m for m in model_ids if cfg.models[m].is_placeholder]
        if unset:
            raise ConfigError(f"models.yaml: model id not set for {unset} (still SET_ME_...)")
        require_provider_keys(cfg, model_ids)
        return cls(cfg, ProviderTransport(cfg), RequestCache(cfg.path("cache_path")))

    def _limiter(self, provider: str) -> ProviderLimiter:
        if provider not in self._limiters:
            pc = self.cfg.providers[provider]
            self._limiters[provider] = ProviderLimiter(pc.max_concurrency, pc.requests_per_minute)
        return self._limiters[provider]

    async def complete(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        params: GenParams,
        structured: StructuredSpec | None = None,
        *,
        use_cache: bool = True,
        cache_salt: str | None = None,
    ) -> Completion:
        model = self.cfg.models[model_id]
        provider = model.provider
        pc = self.cfg.providers[provider]
        req = build_request(model, messages, params, structured)
        key = request_hash(provider, req.body, req.extra_body, cache_salt)
        common = {"dropped_params": req.dropped, "structured_mode": req.structured_mode,
                  "request_hash": key}

        if use_cache and self.cache is not None and (hit := self.cache.get(key)):
            return self._from_payload(provider, hit["payload"], cached=True, **common)

        def _retryable(e: BaseException) -> bool:
            return isinstance(e, ProviderError) and classify_error(pc, e) == "retry"

        retrying = AsyncRetrying(
            stop=stop_after_attempt(pc.max_retries + 1),
            wait=wait_exponential_jitter(multiplier=self.backoff_initial, max=self.backoff_max,
                                         jitter=self.backoff_initial),
            retry=retry_if_exception(_retryable),
            reraise=True,
        )
        start = time.monotonic()
        try:
            async for attempt in retrying:
                with attempt:
                    async with self._limiter(provider).slot():
                        start = time.monotonic()
                        raw = await self._create_with_deadline(provider, req, pc.timeout_s)
            payload = {"kind": "response", "raw": raw,
                       "latency_s": round(time.monotonic() - start, 3)}
        except ProviderError as e:
            kind = classify_error(pc, e)
            if kind != "blocked":
                log.warning("model=%s request=%s %s error: %s", model_id, key[:12], kind, e)
                return Completion(status="error", error=str(e), raw={"error": e.to_payload()},
                                  **common)
            payload = {"kind": "error", "error": e.to_payload(),
                       "latency_s": round(time.monotonic() - start, 3)}

        comp = self._from_payload(provider, payload, cached=False, **common)
        if comp.status in ("ok", "api_blocked") and use_cache and self.cache is not None:
            self.cache.put(key, {"provider": provider, **req.body, "extra_body": req.extra_body},
                           payload, comp.status)
        log.debug("model=%s request=%s status=%s finish=%s", model_id, key[:12], comp.status,
                  comp.finish_reason)
        return comp

    async def _create_with_deadline(self, provider: str, req: Any,
                                    timeout_s: float) -> dict[str, Any]:
        deadline = timeout_s * DEADLINE_FACTOR
        try:
            return await asyncio.wait_for(
                self.transport.create(provider, req.body, req.extra_body, timeout_s), deadline)
        except TimeoutError:
            raise ProviderError(message=f"no complete response within {deadline:g}s",
                                is_timeout=True) from None

    def _from_payload(self, provider: str, payload: dict[str, Any], *, cached: bool,
                      **common: Any) -> Completion:
        pc = self.cfg.providers[provider]
        latency = payload.get("latency_s")
        if payload["kind"] == "error":
            err = ProviderError(**payload["error"])
            return Completion(status="api_blocked", block_code=block_code(err), latency_s=latency,
                              raw={"error": payload["error"]}, cached=cached, **common)
        raw = payload["raw"]
        choices = raw.get("choices") or []
        if not choices:
            return Completion(status="error", error="response has no choices", raw=raw,
                              latency_s=latency, cached=cached, **common)
        choice = choices[0]
        text, reasoning = split_reasoning(choice.get("message") or {})
        finish = choice.get("finish_reason")
        blocked = is_blocked_finish(pc, finish)
        return Completion(
            status="api_blocked" if blocked else "ok",
            text=text if text is not None else ("" if not blocked else None),
            reasoning=reasoning,
            finish_reason=finish,
            usage=raw.get("usage") or {},
            latency_s=latency,
            model_version=raw.get("model"),
            block_code=finish if blocked else None,
            raw=raw,
            cached=cached,
            **common,
        )

    async def complete_many(self, jobs: list[tuple[str, list[dict[str, str]], GenParams]],
                            structured: StructuredSpec | None = None, *,
                            use_cache: bool = True) -> list[Completion]:
        return await asyncio.gather(*(self.complete(m, msgs, p, structured, use_cache=use_cache)
                                      for m, msgs, p in jobs))
