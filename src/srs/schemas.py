"""Record schemas for the long-format study tables (design D4).

Tables: prompts, responses, judgments, human_labels, reference_items. Free-form dict fields
(`JSON_FIELDS`) are stored in Parquet as JSON strings; lists of messages stay nested.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from srs.config import Framing

ExpectedBehavior = Literal["refuse", "comply"]
Tier = Literal["clear", "borderline"]
ResponseStatus = Literal["ok", "api_blocked", "error"]
RefusalLabel = Literal["full_compliance", "partial", "full_refusal"]
JudgeStatus = Literal["ok", "judge_refused", "parse_error", "blocked"]
JudgmentTarget = Literal["responses", "external"]

REFUSAL_LABELS: tuple[str, ...] = ("full_compliance", "partial", "full_refusal")
ADJUDICATED_PASS = 0  # HumanLabel.pass_num for the final adjudicated label


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")

    JSON_FIELDS: ClassVar[tuple[str, ...]] = ()

    @field_validator("*", mode="before")
    @classmethod
    def _decode_json(cls, v: Any, info) -> Any:
        if info.field_name in cls.JSON_FIELDS and isinstance(v, str):
            return json.loads(v)
        return v

    def to_row(self) -> dict[str, Any]:
        row = self.model_dump(mode="json")
        for f in self.JSON_FIELDS:
            row[f] = json.dumps(row[f], sort_keys=True, ensure_ascii=False)
        return row


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str
    reasoning: str | None = None  # assistant reasoning trace, never shown to evaluators


class Prompt(Record):
    prompt_id: str
    intent_id: str
    source: str
    source_id: str
    source_category: str | None = None
    category: str
    category_llm_assigned: bool = False
    expected_behavior: ExpectedBehavior
    tier: Tier = "clear"
    tier_reason: str | None = None
    framing: Framing = "direct"
    turns: list[str] = Field(min_length=1)
    pair_id: str | None = None
    split: str | None = None
    template_version: str | None = None

    @field_validator("turns")
    @classmethod
    def _first_turn_nonempty(cls, v: list[str]) -> list[str]:
        if not v[0].strip():
            raise ValueError("first user turn is empty")
        return v


class Response(Record):
    JSON_FIELDS: ClassVar[tuple[str, ...]] = ("gen_params", "usage")

    response_id: str
    prompt_id: str
    model_id: str
    model_version: str | None = None  # the `model` string the API returned
    gen_params: dict[str, Any]
    sample_idx: int = 0
    transcript: list[Message]
    final_text: str | None = None
    reasoning_text: str | None = None
    finish_reason: str | None = None
    status: ResponseStatus
    block_turn: int | None = None  # 1-based user-turn index where a provider block occurred
    error: str | None = None
    dropped_params: list[str] = []
    usage: dict[str, Any] = {}
    latency_s: float | None = None
    run_id: str
    created_at: str = Field(default_factory=utcnow)


class Judgment(Record):
    item_id: str  # response_id for study responses, reference item_id for external sets
    target: JudgmentTarget = "responses"
    evaluator_id: str
    evaluator_version: str
    refusal: RefusalLabel | None = None
    harmful: bool | None = None
    status: JudgeStatus
    raw_output: str | None = None
    run_id: str | None = None
    created_at: str = Field(default_factory=utcnow)


class HumanLabel(Record):
    item_id: str  # response_id
    pass_num: int  # 1, 2, or ADJUDICATED_PASS
    refusal: RefusalLabel | None
    harmful: bool | None = None
    notes: str = ""
    seconds: float | None = None
    labeled_at: str = Field(default_factory=utcnow)


class ReferenceItem(Record):
    item_id: str
    source: str
    source_id: str
    prompt: str
    response: str
    refusal: RefusalLabel | None = None
    harmful: bool | None = None
    prompt_harmful: bool | None = None
    split: str | None = None


TABLES: dict[str, type[Record]] = {
    "prompts": Prompt,
    "responses": Response,
    "judgments": Judgment,
    "human_labels": HumanLabel,
    "reference_items": ReferenceItem,
}
