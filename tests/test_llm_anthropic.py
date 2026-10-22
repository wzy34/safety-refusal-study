import asyncio

import anthropic
import httpx2

from srs.evaluators.llm_judge import JudgeVerdict
from srs.llm.adapters import GenParams, build_request
from srs.llm.anthropic_api import to_chat_completion, to_messages_request
from srs.llm.client import ProviderTransport
from srs.llm.errors import classify_error, from_anthropic_exception
from srs.llm.fake import make_fake_client
from srs.llm.structured import schema_spec, structured_call

P = GenParams(max_tokens=512, temperature=0.0, seed=1)


def run(coro):
    return asyncio.run(coro)


def message(text="OK", *, stop="end_turn", thinking=None, stop_details=None):
    content = [{"type": "thinking", "thinking": thinking, "signature": "s"}] if thinking else []
    if text is not None:
        content.append({"type": "text", "text": text})
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
            "content": content, "stop_reason": stop, "stop_details": stop_details,
            "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 5,
                      "cache_creation_input_tokens": None, "service_tier": "standard",
                      "server_tool_use": None}}


def test_request_translation(cfg):
    msgs = [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Judge."}]
    req = build_request(cfg.models["judge-claude"], msgs, P,
                        schema_spec("judgeverdict", JudgeVerdict))
    assert req.dropped == ["temperature", "seed"]
    out = to_messages_request(req.body, req.extra_body)
    assert out["model"] == "claude-haiku-5-5" and out["max_tokens"] == 512
    assert out["system"] == "Be terse."
    assert out["messages"] == [{"role": "user", "content": "Judge."}]
    assert out["thinking"] == {"type": "disabled"}
    assert "temperature" not in out and "seed" not in out and "response_format" not in out
    fmt = out["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"]["additionalProperties"] is False


def test_output_config_extras_merge_with_format():
    body = {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "x"}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "n", "schema": {"type": "object"}}}}
    out = to_messages_request(body, {"output_config": {"effort": "low"}})
    assert out["output_config"] == {"effort": "low",
                                    "format": {"type": "json_schema", "schema": {"type": "object"}}}


def test_response_translation():
    r = to_chat_completion(message("Fine.", thinking="hmm"))
    choice = r["choices"][0]
    assert choice["finish_reason"] == "stop" and choice["message"]["content"] == "Fine."
    assert choice["message"]["reasoning_content"] == "hmm" and r["model"] == "claude-haiku-5-5"
    # only integer counts survive, so runner usage sums keep working
    assert r["usage"] == {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 5,
                          "prompt_tokens": 105, "completion_tokens": 20, "total_tokens": 125}
    assert to_chat_completion(message(stop="max_tokens"))["choices"][0]["finish_reason"] == "length"


def test_refusal_is_blocked(cfg):
    details = {"type": "refusal", "category": "cyber", "explanation": "x"}
    client = make_fake_client(cfg, queue=[to_chat_completion(
        message(None, stop="refusal", stop_details=details))])
    c = run(client.complete("judge-claude", [{"role": "user", "content": "q"}], P))
    assert c.status == "api_blocked" and c.block_code == "refusal"
    assert c.raw["stop_details"]["category"] == "cyber"


def test_judge_verdict_parses(cfg):
    text = '{"rationale": "declines", "refusal": "full_refusal", "harmful": false}'
    client = make_fake_client(cfg, queue=[to_chat_completion(message(text))])
    res = run(structured_call(client, "judge-claude", [{"role": "user", "content": "q"}],
                              JudgeVerdict, P))
    assert res.status == "ok" and res.value.refusal == "full_refusal"


def _status_error(cls, status, etype):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"type": "error", "error": {"type": etype, "message": f"{etype} happened"}}
    return cls(f"{etype} happened", response=httpx2.Response(status, request=req), body=body)


def test_error_mapping(cfg):
    pc = cfg.providers["anthropic"]
    rate = from_anthropic_exception(_status_error(anthropic.RateLimitError, 429,
                                                  "rate_limit_error"))
    assert rate.status_code == 429 and rate.code == "rate_limit_error"
    assert classify_error(pc, rate) == "retry"
    overloaded = from_anthropic_exception(_status_error(anthropic.InternalServerError, 529,
                                                        "overloaded_error"))
    assert classify_error(pc, overloaded) == "retry"
    bad = from_anthropic_exception(_status_error(anthropic.BadRequestError, 400,
                                                 "invalid_request_error"))
    assert classify_error(pc, bad) == "fatal" and "invalid_request_error" in bad.message
    timeout = from_anthropic_exception(anthropic.APITimeoutError(
        request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")))
    assert timeout.is_timeout and classify_error(pc, timeout) == "retry"


def test_provider_transport_routes_by_api(cfg):
    seen = []

    class Recorder:
        def __init__(self, name):
            self.name = name

        async def create(self, provider, body, extra_body, timeout):
            seen.append((self.name, provider))
            return {}

    t = ProviderTransport(cfg)
    t._by_api = {"openai": Recorder("openai"), "anthropic": Recorder("anthropic")}
    run(t.create("anthropic", {}, {}, 1.0))
    run(t.create("google", {}, {}, 1.0))
    assert seen == [("anthropic", "anthropic"), ("openai", "google")]
