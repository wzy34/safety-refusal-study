"""Deterministic identifiers (design D4). Same inputs -> same id, so reruns are idempotent."""

from __future__ import annotations

import hashlib
import json
from typing import Any

ID_LEN = 16  # hex chars (64 bits) for record ids; cache keys use the full digest


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def sha256_hex(obj: Any) -> str:
    data = obj if isinstance(obj, str) else canonical_json(obj)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def short_hash(obj: Any) -> str:
    return sha256_hex(obj)[:ID_LEN]


def intent_id(source: str, source_id: str) -> str:
    return short_hash({"kind": "intent", "source": source, "source_id": source_id})


def prompt_id(source: str, source_id: str, framing: str, turns: list[str]) -> str:
    return short_hash({"kind": "prompt", "source": source, "source_id": source_id,
                       "framing": framing, "turns": list(turns)})


def gen_params_hash(params: dict[str, Any]) -> str:
    return short_hash({k: v for k, v in params.items() if v is not None})


def response_id(prompt_id: str, model_id: str, sample_idx: int, gen_params_hash: str) -> str:
    return short_hash({"kind": "response", "prompt_id": prompt_id, "model_id": model_id,
                       "sample_idx": sample_idx, "gen_params": gen_params_hash})


def pair_id(*parts: str) -> str:
    return short_hash({"kind": "pair", "parts": list(parts)})
