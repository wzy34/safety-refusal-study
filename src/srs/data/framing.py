"""Framing variants for the RQ3 split (design D7): academic, roleplay, hypothetical and a
scripted 3-turn conversation, each checked for intent preservation.

Attempt plan per (intent, framing), with N = framings.max_attempts:
  attempts 1..N-1: LLM rewrite of the configured template (attempt k > 1 gets feedback);
  attempt N (or as soon as the rewriter refuses/blocks/fails): the plain template, no LLM.
Every candidate goes through the intent check. If all attempts fail, the whole intent is
dropped from RQ3 and reported with the failing framing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, field_validator

from srs import ids
from srs.config import FramingTemplate, StudyConfig
from srs.data.build import processed_dir, prompts_path, write_report
from srs.data.intent_check import check_intent, intent_check_version
from srs.io import read_records, write_records
from srs.llm.client import ChatClient
from srs.llm.structured import structured_call, utility_params
from srs.schemas import Prompt
from srs.templates import load_template, render_string

log = logging.getLogger(__name__)

NON_DIRECT = ("academic", "roleplay", "hypothetical", "multiturn")
FEEDBACK = ("\nA previous rewrite was rejected because it did not ask for exactly the same "
            "thing as the original ({reason}). Stay closer to the original request.\n")


class SingleRewrite(BaseModel):
    prompt: str = Field(min_length=1)


class MultiTurnRewrite(BaseModel):
    turns: list[str]

    @field_validator("turns")
    @classmethod
    def _three(cls, v: list[str]) -> list[str]:
        if len(v) != 3 or not all(t.strip() for t in v):
            raise ValueError("exactly three non-empty turns required")
        return v


@dataclass
class FramingOutcome:
    framing: str
    prompt: Prompt | None
    attempts: list[dict[str, Any]] = field(default_factory=list)


def template_turns(tpl: FramingTemplate, request: str) -> list[str]:
    if tpl.turns is not None:
        return [render_string(t, request=request) for t in tpl.turns]
    return [render_string(tpl.wrapper or "", request=request)]


async def _rewrite(cfg: StudyConfig, client: ChatClient, framing: str, tpl: FramingTemplate,
                   request: str, feedback: str) -> tuple[list[str] | None, str, str]:
    rewrite_tpl = load_template(cfg.path("prompts_dir"), tpl.rewrite_prompt or "")
    scaffold = "\n".join(f"Message {i}: {t}" for i, t in
                         enumerate(template_turns(tpl, "<the request>"), 1)) \
        if tpl.turns else template_turns(tpl, "<the request>")[0]
    msgs = [{"role": "user", "content": rewrite_tpl.render(template=scaffold, request=request,
                                                           feedback=feedback)}]
    output = MultiTurnRewrite if framing == "multiturn" else SingleRewrite
    res = await structured_call(client, cfg.framings.rewriter_model, msgs, output,
                                utility_params(cfg))
    if res.status != "ok" or res.value is None:
        return None, res.status, rewrite_tpl.version
    turns = res.value.turns if framing == "multiturn" else [res.value.prompt]  # type: ignore
    return [t.strip() for t in turns], "ok", rewrite_tpl.version


async def frame_intent(direct: Prompt, framing: str, cfg: StudyConfig,
                       client: ChatClient) -> FramingOutcome:
    tpl = cfg.framings.templates[framing]  # type: ignore[index]
    request = direct.turns[0]
    n = cfg.framings.max_attempts
    outcome = FramingOutcome(framing, None)
    check_v = intent_check_version(cfg)
    feedback = ""
    for attempt in range(1, n + 1):
        use_template = attempt == n or any(a["mode"] == "llm" and a["rewrite_status"] != "ok"
                                           for a in outcome.attempts)
        if use_template:
            turns = template_turns(tpl, request)
            mode, status = "template", "ok"
            version = "tpl:" + ids.short_hash(tpl.model_dump())[:12]
        else:
            turns, status, rv = await _rewrite(cfg, client, framing, tpl, request, feedback)
            mode, version = "llm", f"rw:{rv}"
        record: dict[str, Any] = {"attempt": attempt, "mode": mode, "rewrite_status": status}
        if turns is None:
            record["check"] = None
            outcome.attempts.append(record)
            continue
        check = await check_intent(request, turns, framing, cfg, client)
        record["check"] = {"passed": check.passed, "status": check.status,
                           "reason": check.reason}
        outcome.attempts.append(record)
        if check.passed:
            outcome.prompt = direct.model_copy(update={
                "framing": framing,
                "turns": turns,
                "prompt_id": ids.prompt_id(direct.source, direct.source_id, framing, turns),
                "template_version": f"{version}+ic:{check_v}",
            })
            return outcome
        feedback = FEEDBACK.format(reason=check.reason[:160])
        if mode == "template":
            break  # the plain template is the last resort
    return outcome


@dataclass
class FramingResult:
    framed: list[Prompt]
    dropped: list[dict[str, Any]]
    attempts: list[dict[str, Any]]


async def generate_framings(directs: list[Prompt], cfg: StudyConfig,
                            client: ChatClient) -> FramingResult:
    """Frame every given direct prompt in all four non-direct framings."""
    jobs = [(d, f) for d in directs for f in NON_DIRECT]
    outcomes = await asyncio.gather(*(frame_intent(d, f, cfg, client) for d, f in jobs))
    by_intent: dict[str, list[FramingOutcome]] = {}
    for (d, _), o in zip(jobs, outcomes, strict=True):
        by_intent.setdefault(d.intent_id, []).append(o)
    framed, dropped, attempts = [], [], []
    for d in directs:
        outs = by_intent[d.intent_id]
        attempts += [{"intent_id": d.intent_id, "framing": o.framing, "attempts": o.attempts}
                     for o in outs]
        failed = [o.framing for o in outs if o.prompt is None]
        if failed:
            dropped.append({"intent_id": d.intent_id, "source": d.source,
                            "source_id": d.source_id, "framing": ", ".join(failed)})
        else:
            framed += [o.prompt for o in outs if o.prompt is not None]
    return FramingResult(framed, dropped, attempts)


async def frame_rq3(cfg: StudyConfig, client: ChatClient, limit: int | None = None
                    ) -> FramingResult:
    """`srs data frame`: (re)generate framings for the RQ3 split and update prompts.parquet.
    Dropped intents get split `rq3_dropped`."""
    prompts = read_records(prompts_path(cfg), Prompt)
    if not prompts:
        raise FileNotFoundError(f"{prompts_path(cfg)} not found; run `srs data build` first")
    directs = [p for p in prompts if p.framing == "direct"]
    rq3 = [p for p in directs if p.split in ("rq3", "rq3_dropped")]
    if limit is not None:
        rq3 = rq3[:limit]
    result = await generate_framings(rq3, cfg, client)
    dropped_ids = {d["intent_id"] for d in result.dropped}
    rq3_ids = {p.intent_id for p in rq3}
    kept_framed = [p for p in prompts if p.framing != "direct" and p.intent_id not in rq3_ids]
    new_directs = [
        p.model_copy(update={"split": "rq3_dropped" if p.intent_id in dropped_ids else "rq3"})
        if p.intent_id in rq3_ids else p for p in directs]
    all_prompts = new_directs + kept_framed + result.framed
    write_records(prompts_path(cfg), all_prompts)
    out = processed_dir(cfg)
    (out / "framing_dropped.json").write_text(json.dumps(result.dropped, indent=2))
    (out / "framing_attempts.json").write_text(json.dumps(result.attempts, indent=2))
    write_report(cfg, all_prompts)
    log.info("framed %d intents: %d prompts, %d dropped", len(rq3), len(result.framed),
             len(result.dropped))
    return result
