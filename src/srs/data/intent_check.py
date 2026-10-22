"""Intent-preservation check: a model from a different family than the rewriter confirms that
a framed prompt asks for the same thing as the direct prompt (and, for multi-turn framings,
that only the final turn makes the request)."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from srs.config import StudyConfig
from srs.llm.client import ChatClient
from srs.llm.structured import structured_call, utility_params
from srs.templates import load_template


class IntentVerdict(BaseModel):
    rationale: str
    same_request: bool
    core_request_in_final_turn: bool


@dataclass
class IntentCheckResult:
    passed: bool
    status: str  # ok | refused | parse_error | blocked | error
    reason: str


def render_turns(turns: list[str]) -> str:
    if len(turns) == 1:
        return turns[0]
    return "\n\n".join(f"[User message {i}]\n{t}" for i, t in enumerate(turns, 1))


async def check_intent(original: str, turns: list[str], framing: str, cfg: StudyConfig,
                       client: ChatClient) -> IntentCheckResult:
    tpl = load_template(cfg.path("prompts_dir"), cfg.framings.intent_check_prompt)
    msgs = [{"role": "user", "content": tpl.render(original=original, framing=framing,
                                                   rephrased=render_turns(turns))}]
    res = await structured_call(client, cfg.framings.checker_model, msgs, IntentVerdict,
                                utility_params(cfg))
    if res.status != "ok" or res.value is None:
        return IntentCheckResult(False, res.status, f"checker {res.status}")
    v = res.value
    if not v.same_request:
        return IntentCheckResult(False, "ok", f"different request: {v.rationale[:200]}")
    if len(turns) > 1 and not v.core_request_in_final_turn:
        return IntentCheckResult(False, "ok", f"request before final turn: {v.rationale[:200]}")
    return IntentCheckResult(True, "ok", v.rationale[:200])


def intent_check_version(cfg: StudyConfig) -> str:
    return load_template(cfg.path("prompts_dir"), cfg.framings.intent_check_prompt).version
