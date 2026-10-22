"""Anthropic Messages API transport (Claude judge).

The rest of the client works on Chat Completions shapes, so this module translates a built
request into a Messages API request and the reply back into a Chat Completions-shaped dict.
Both translations are pure functions; `AnthropicTransport` only adds the SDK call.

A safety-classifier decline (HTTP 200, `stop_reason: "refusal"`) keeps `refusal` as its finish
reason, so providers.yaml can list it under `blocked_finish_reasons`.
"""

from __future__ import annotations

from typing import Any

from srs.config import StudyConfig, get_secret
from srs.llm.errors import from_anthropic_exception

FINISH = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
          "tool_use": "tool_calls"}
USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_read_input_tokens",
                "cache_creation_input_tokens")


def to_messages_request(body: dict[str, Any], extra_body: dict[str, Any]) -> dict[str, Any]:
    """Chat Completions body (+ provider extras) -> Messages API arguments."""
    system = [m["content"] for m in body["messages"] if m["role"] == "system"]
    req: dict[str, Any] = {
        "model": body["model"],
        "max_tokens": body["max_tokens"],
        "messages": [{"role": m["role"], "content": m["content"]}
                     for m in body["messages"] if m["role"] != "system"],
    }
    if system:
        req["system"] = "\n\n".join(system)
    if "temperature" in body:
        req["temperature"] = body["temperature"]
    extras = dict(extra_body)
    output_config = dict(extras.pop("output_config", {}))
    fmt = body.get("response_format")
    if fmt and fmt.get("type") == "json_schema":
        output_config["format"] = {"type": "json_schema", "schema": fmt["json_schema"]["schema"]}
    if output_config:
        req["output_config"] = output_config
    req.update(extras)
    return req


def to_chat_completion(msg: dict[str, Any]) -> dict[str, Any]:
    """Messages API reply -> Chat Completions-shaped dict (text, reasoning, finish, usage)."""
    blocks = msg.get("content") or []
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    reasoning = "\n".join(b.get("thinking") or "" for b in blocks
                          if b.get("type") == "thinking").strip() or None
    stop = msg.get("stop_reason")
    raw_usage = msg.get("usage") or {}
    usage = {k: raw_usage[k] for k in USAGE_FIELDS if isinstance(raw_usage.get(k), int)}
    prompt = sum(usage.get(k, 0) for k in USAGE_FIELDS if k != "output_tokens")
    usage.update(prompt_tokens=prompt, completion_tokens=usage.get("output_tokens", 0),
                 total_tokens=prompt + usage.get("output_tokens", 0))
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning:
        message["reasoning_content"] = reasoning
    return {
        "id": msg.get("id"), "object": "chat.completion", "model": msg.get("model"),
        "choices": [{"index": 0, "finish_reason": FINISH.get(stop, stop), "message": message}],
        "usage": usage,
        "stop_reason": stop,
        "stop_details": msg.get("stop_details"),
    }


class AnthropicTransport:
    """One `AsyncAnthropic` per provider, SDK retries disabled (the client retries)."""

    def __init__(self, cfg: StudyConfig):
        self.cfg = cfg
        self._clients: dict[str, Any] = {}

    def _client(self, provider: str):
        if provider not in self._clients:
            from anthropic import AsyncAnthropic

            pc = self.cfg.providers[provider]
            self._clients[provider] = AsyncAnthropic(api_key=get_secret(pc.api_key_env),
                                                     base_url=pc.base_url, max_retries=0)
        return self._clients[provider]

    async def create(self, provider, body, extra_body, timeout):
        import inspect

        from anthropic.resources.messages import AsyncMessages

        known = inspect.signature(AsyncMessages.create).parameters
        req = to_messages_request(body, extra_body)
        # Fields the SDK has no keyword for (e.g. temperature) still go out as raw body fields.
        rest = {k: req.pop(k) for k in list(req) if k not in known}
        try:
            msg = await self._client(provider).messages.create(
                **req, extra_body=rest or None, timeout=timeout)
        except Exception as e:  # normalized and classified by the caller
            raise from_anthropic_exception(e) from e
        return to_chat_completion(msg.model_dump(mode="json"))
