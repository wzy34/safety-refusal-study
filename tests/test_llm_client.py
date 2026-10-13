import asyncio

import pytest

from srs.config import ConfigError
from srs.llm.adapters import GenParams, StructuredSpec
from srs.llm.cache import RequestCache
from srs.llm.client import ChatClient, split_reasoning
from srs.llm.fake import (
    block_error,
    fake_response,
    fatal_error,
    make_fake_client,
    rate_limit_error,
    server_error,
)

MSGS = [{"role": "user", "content": "How do I bake bread?"}]
P = GenParams(max_tokens=1024, temperature=0.0, seed=1)


def run(coro):
    return asyncio.run(coro)


def test_success(cfg):
    client = make_fake_client(cfg, queue=[fake_response("Here is how.", model="gemini-x-2026")])
    c = run(client.complete("gemini-a", MSGS, P))
    assert c.status == "ok" and c.text == "Here is how." and c.finish_reason == "stop"
    assert c.usage["completion_tokens"] == 20 and c.model_version == "gemini-x-2026"
    assert c.raw is not None and c.latency_s is not None and not c.cached


def test_reasoning_separated(cfg):
    client = make_fake_client(cfg, queue=[fake_response("Answer.", reasoning="Let me think")])
    c = run(client.complete("deepseek-b", MSGS, P))
    assert c.text == "Answer." and c.reasoning == "Let me think"


def test_think_tags_split():
    text, reasoning = split_reasoning({"content": "<think>hmm</think>\nFinal."})
    assert (text, reasoning) == ("Final.", "hmm")


def test_temperature_dropped_is_recorded(cfg):
    client = make_fake_client(cfg)
    c = run(client.complete("gemini-b", MSGS, P))  # gemini-b: supports_temperature false
    assert "temperature" in c.dropped_params
    assert "temperature" not in client.transport.calls[0][1]


def test_retry_then_success(cfg):
    client = make_fake_client(cfg, queue=[rate_limit_error(), server_error(), fake_response("ok")])
    c = run(client.complete("gemini-a", MSGS, P))
    assert c.status == "ok" and len(client.transport.calls) == 3


def test_retries_exhausted_not_cached(cfg):
    n = cfg.providers["google"].max_retries + 1
    client = make_fake_client(cfg, queue=[rate_limit_error()] * n + [fake_response("later")])
    c = run(client.complete("gemini-a", MSGS, P))
    assert c.status == "error" and "429" in c.error and len(client.transport.calls) == n
    assert len(client.cache) == 0
    assert run(client.complete("gemini-a", MSGS, P)).status == "ok"  # next run retries


def test_block_error_not_retried_and_cached(cfg):
    client = make_fake_client(cfg, queue=[block_error("1301")])
    c = run(client.complete("glm-a", MSGS, P))
    assert c.status == "api_blocked" and c.block_code == "1301"
    assert len(client.transport.calls) == 1 and len(client.cache) == 1
    again = run(client.complete("glm-a", MSGS, P))
    assert again.status == "api_blocked" and again.cached and len(client.transport.calls) == 1


def test_block_by_finish_reason(cfg):
    client = make_fake_client(cfg, queue=[fake_response("partial", finish_reason="sensitive")])
    c = run(client.complete("glm-a", MSGS, P))
    assert c.status == "api_blocked" and c.block_code == "sensitive"


def test_fatal_error(cfg):
    client = make_fake_client(cfg, queue=[fatal_error()])
    c = run(client.complete("gemini-a", MSGS, P))
    assert c.status == "error" and len(client.transport.calls) == 1


def test_cache_hit_makes_no_call(cfg, tmp_path):
    cache = RequestCache(tmp_path / "c.sqlite")
    client = make_fake_client(cfg, cache=cache)
    first = run(client.complete("kimi-a", MSGS, P))
    second = run(client.complete("kimi-a", MSGS, P))
    assert len(client.transport.calls) == 1 and second.cached and second.text == first.text
    run(client.complete("kimi-a", MSGS, GenParams(max_tokens=2048, temperature=0.0)))
    assert len(client.transport.calls) == 2


def test_interrupted_run_resumes_from_cache(cfg, tmp_path):
    path = tmp_path / "c.sqlite"
    run(make_fake_client(cfg, cache=RequestCache(path)).complete("kimi-a", MSGS, P))
    fresh = make_fake_client(cfg, cache=RequestCache(path))  # a "restarted" process
    assert run(fresh.complete("kimi-a", MSGS, P)).cached and fresh.transport.calls == []


def test_structured_json_object(cfg):
    spec = StructuredSpec("v", {"type": "object"})
    client = make_fake_client(cfg, queue=[fake_response('{"a": 1}')])
    c = run(client.complete("judge-deepseek", MSGS, P, spec))
    assert c.structured_mode == "json_object"
    assert client.transport.calls[0][1]["response_format"] == {"type": "json_object"}


def test_concurrency_respected_through_client(cfg):
    cfg.providers["moonshot"].max_concurrency = 2
    cfg.providers["moonshot"].requests_per_minute = 60_000
    client = make_fake_client(cfg, delay_s=0.02)
    inflight = peak = 0
    orig = client.transport.create

    async def tracking(*a):
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        try:
            return await orig(*a)
        finally:
            inflight -= 1

    client.transport.create = tracking
    jobs = [("kimi-a", [{"role": "user", "content": f"q{i}"}], P) for i in range(10)]
    run(client.complete_many(jobs))
    assert peak == 2


def test_real_client_refuses_placeholders(cfg):
    cfg.models["gemini-a"] = cfg.models["gemini-a"].model_copy(update={"model": "SET_ME_x"})
    with pytest.raises(ConfigError, match="SET_ME"):
        ChatClient.real(cfg, ["gemini-a"])


def test_cache_salt_separates_repeated_samples(cfg):
    client = make_fake_client(cfg)
    for s in ("1", "2", "1"):
        run(client.complete("kimi-a", MSGS, P, cache_salt=s))
    assert len(client.transport.calls) == 2
