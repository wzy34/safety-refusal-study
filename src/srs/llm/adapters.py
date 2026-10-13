"""Per-model request building (design D2). `build_request` is a pure function: it sends only
the parameters a model declares supported and records every configured parameter it drops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from srs.config import ModelConfig


@dataclass(frozen=True)
class GenParams:
    max_tokens: int
    temperature: float | None = None
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"max_tokens": self.max_tokens, "temperature": self.temperature, "seed": self.seed}


@dataclass(frozen=True)
class StructuredSpec:
    """A requested JSON output schema (used by the LLM judges and the framing helpers)."""

    name: str
    schema: dict[str, Any]


@dataclass
class BuiltRequest:
    body: dict[str, Any]  # standard Chat Completions arguments, including model and messages
    extra_body: dict[str, Any] = field(default_factory=dict)  # provider-specific fields
    dropped: list[str] = field(default_factory=list)
    structured_mode: str | None = None  # json_schema | json_object | None


def build_request(
    model: ModelConfig,
    messages: list[dict[str, str]],
    params: GenParams,
    structured: StructuredSpec | None = None,
) -> BuiltRequest:
    body: dict[str, Any] = {"model": model.model, "messages": messages}
    dropped: list[str] = []

    body[model.token_param] = params.max_tokens

    if params.temperature is not None:
        if model.supports_temperature:
            body["temperature"] = params.temperature
        else:
            dropped.append("temperature")
    if params.seed is not None:
        if model.supports_seed:
            body["seed"] = params.seed
        else:
            dropped.append("seed")

    mode: str | None = None
    if structured is not None:
        if model.structured_output == "json_schema":
            mode = "json_schema"
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": structured.name, "schema": structured.schema,
                                "strict": True},
            }
        elif model.structured_output == "json_object":
            mode = "json_object"
            body["response_format"] = {"type": "json_object"}
        else:
            dropped.append("response_format")

    return BuiltRequest(body=body, extra_body=dict(model.extra_body), dropped=dropped,
                        structured_mode=mode)
