"""JSON-returning LLM calls with schema validation, corrective retries and refusal detection.

Used by the LLM judges and by the dataset helpers (category assignment, tier screen, framing
rewrites, intent check). A retry appends the invalid reply plus a corrective user turn, so the
request (and its cache key) differs from the first attempt while staying deterministic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ValidationError

from srs.llm.adapters import GenParams, StructuredSpec
from srs.llm.client import ChatClient, Completion

CallStatus = Literal["ok", "refused", "parse_error", "blocked", "error"]

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)
CORRECTION = ("Your previous reply was not a valid JSON object matching the required format. "
              "Reply again with only the JSON object and nothing else.")


@dataclass
class StructuredResult[T: BaseModel]:
    status: CallStatus
    value: T | None
    raw_text: str | None
    attempts: int
    completion: Completion | None


def extract_json(text: str) -> dict | None:
    if not text:
        return None
    m = _FENCE.match(text)
    candidate = m.group(1) if m else text.strip()
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(candidate[start:end + 1])
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def looks_like_refusal(text: str | None) -> bool:
    from srs.evaluators.keyword import looks_like_refusal as _refusal

    return bool(text) and _refusal(text)


def schema_spec(name: str, model: type[BaseModel]) -> StructuredSpec:
    schema = model.model_json_schema()
    schema["additionalProperties"] = False
    schema["required"] = list(schema.get("properties", {}))
    schema.pop("title", None)
    return StructuredSpec(name, schema)


async def structured_call[T: BaseModel](
    client: ChatClient,
    model_id: str,
    messages: list[dict[str, str]],
    output: type[T],
    params: GenParams,
    max_retries: int = 2,
) -> StructuredResult[T]:
    spec = schema_spec(output.__name__.lower(), output)
    convo = list(messages)
    comp: Completion | None = None
    for attempt in range(1, max_retries + 2):
        comp = await client.complete(model_id, convo, params, spec)
        if comp.status == "api_blocked":
            return StructuredResult("blocked", None, comp.text, attempt, comp)
        if comp.status == "error":
            return StructuredResult("error", None, comp.error, attempt, comp)
        obj = extract_json(comp.text or "")
        if obj is not None:
            try:
                return StructuredResult("ok", output.model_validate(obj), comp.text, attempt, comp)
            except ValidationError:
                pass
        elif looks_like_refusal(comp.text):
            return StructuredResult("refused", None, comp.text, attempt, comp)
        convo = [*convo, {"role": "assistant", "content": comp.text or ""},
                 {"role": "user", "content": CORRECTION}]
    return StructuredResult("parse_error", None, comp.text if comp else None, max_retries + 1,
                            comp)


def utility_params(cfg, max_tokens: int = 1024) -> GenParams:
    """Deterministic settings for helper calls (category, tier screen, rewrites, checks)."""
    return GenParams(max_tokens=max_tokens, temperature=0.0, seed=cfg.experiment.seed)
