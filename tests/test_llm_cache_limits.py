import asyncio
import time

import pytest

from srs.config import ProviderConfig
from srs.llm.cache import RequestCache, request_hash
from srs.llm.errors import ProviderError, classify_error, is_blocked_finish
from srs.llm.ratelimit import ProviderLimiter, TokenBucket

# --- cache -----------------------------------------------------------------------------------

def _body(max_tokens=1024):
    return {"model": "m", "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": max_tokens}


def test_cache_hit_and_param_miss(tmp_path):
    c = RequestCache(tmp_path / "c.sqlite")
    k = request_hash("openai", _body(), {})
    assert c.get(k) is None
    c.put(k, _body(), {"kind": "response", "raw": {"x": 1}}, "ok")
    assert c.get(k)["payload"]["raw"] == {"x": 1}
    assert c.get(request_hash("openai", _body(2048), {})) is None
    assert request_hash("deepseek", _body(), {}) != k
    assert request_hash("openai", _body(), {"thinking": 1}) != k


def test_cache_persists_across_reopen(tmp_path):
    path = tmp_path / "c.sqlite"
    c = RequestCache(path)
    k = request_hash("openai", _body(), {})
    c.put(k, _body(), {"kind": "error", "error": {}}, "api_blocked")
    c.close()
    again = RequestCache(path)
    assert again.get(k)["status"] == "api_blocked" and len(again) == 1


def test_cache_rejects_error_status(tmp_path):
    with pytest.raises(ValueError):
        RequestCache(tmp_path / "c.sqlite").put("k", {}, {}, "error")


# --- rate limiting ---------------------------------------------------------------------------

def test_concurrency_never_exceeds_limit():
    async def run():
        lim = ProviderLimiter(max_concurrency=3, requests_per_minute=60_000)
        inflight = peak = 0

        async def job():
            nonlocal inflight, peak
            async with lim.slot():
                inflight += 1
                peak = max(peak, inflight)
                await asyncio.sleep(0.01)
                inflight -= 1

        await asyncio.gather(*(job() for _ in range(20)))
        return peak

    assert asyncio.run(run()) == 3


def test_token_bucket_rate():
    async def run():
        b = TokenBucket(rate_per_s=50, capacity=1)
        t0 = time.monotonic()
        for _ in range(6):
            await b.acquire()
        return time.monotonic() - t0

    assert asyncio.run(run()) >= 5 / 50 * 0.9


# --- error classification --------------------------------------------------------------------

PC = ProviderConfig(base_url="x", api_key_env="K", blocked_error_codes=["1301", "content_filter"],
                    blocked_message_patterns=["Content Exists Risk"],
                    blocked_finish_reasons=["content_filter", "sensitive"])


@pytest.mark.parametrize("err,expected", [
    (ProviderError(status_code=429), "retry"),
    (ProviderError(status_code=500), "retry"),
    (ProviderError(status_code=503), "retry"),
    (ProviderError(is_timeout=True), "retry"),
    (ProviderError(is_connection=True), "retry"),
    (ProviderError(status_code=400, code="1301", message="unsafe"), "blocked"),
    (ProviderError(status_code=400, type="content_filter", message="high risk"), "blocked"),
    (ProviderError(status_code=400, message="Content Exists Risk"), "blocked"),
    (ProviderError(status_code=400, message="content exists risk"), "blocked"),
    (ProviderError(status_code=400, code="invalid_request", message="bad"), "fatal"),
    (ProviderError(status_code=401, message="unauthorized"), "fatal"),
    (ProviderError(status_code=404, message="model not found"), "fatal"),
])
def test_classify_error(err, expected):
    assert classify_error(PC, err) == expected


@pytest.mark.parametrize("finish,blocked", [
    ("content_filter", True), ("sensitive", True), ("stop", False), ("length", False),
    (None, False), ("content_filter: PROHIBITED_CONTENT", True),
    ("content_filter: RECITATION", True), ("length: max_tokens", False),
])
def test_blocked_finish_reason(finish, blocked):
    assert is_blocked_finish(PC, finish) == blocked


def test_from_openai_exception():
    import httpx2 as httpx
    import openai

    from srs.llm.errors import from_openai_exception

    req = httpx.Request("POST", "https://x/v1/chat/completions")
    body = {"error": {"code": "1301", "message": "unsafe content"}}
    exc = openai.BadRequestError("bad", response=httpx.Response(400, request=req), body=body)
    pe = from_openai_exception(exc)
    assert (pe.status_code, pe.code) == (400, "1301") and classify_error(PC, pe) == "blocked"
    assert from_openai_exception(openai.APITimeoutError(request=req)).is_timeout
