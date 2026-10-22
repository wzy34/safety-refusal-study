"""Clarity tiers: `borderline` for prompts whose expected behavior is contested, from
configured source rules plus an LLM screen of configured sources. Everything else is `clear`."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from srs.config import StudyConfig, TaxonomyFile
from srs.llm.client import ChatClient
from srs.llm.structured import structured_call, utility_params
from srs.schemas import Prompt
from srs.templates import load_template


class TierScreen(BaseModel):
    rationale: str
    borderline: bool


def apply_tier_rules(prompts: list[Prompt], tx: TaxonomyFile) -> list[Prompt]:
    out = []
    for p in prompts:
        rule = next((r for r in tx.tier_rules if r.source == p.source
                     and (r.source_categories is None or p.source_category in r.source_categories)
                     and (r.expected_behavior is None
                          or r.expected_behavior == p.expected_behavior)), None)
        if rule is None:
            out.append(p.model_copy(update={"tier": "clear", "tier_reason": None}))
        else:
            out.append(p.model_copy(update={"tier": "borderline",
                                            "tier_reason": f"rule: {rule.reason or rule.source}"}))
    return out


async def screen_tiers(prompts: list[Prompt], cfg: StudyConfig,
                       client: ChatClient) -> list[Prompt]:
    """LLM screen over clear prompts from tier_screen_sources. A failed screen (blocked,
    refused, unparseable) marks the prompt borderline, so it stays out of primary metrics."""
    tx = cfg.taxonomy
    todo = [i for i, p in enumerate(prompts)
            if p.source in tx.tier_screen_sources and p.tier == "clear"]
    if not todo or tx.tier_screen_model is None:
        return prompts
    tpl = load_template(cfg.path("prompts_dir"), "tier_screen.md")
    params = utility_params(cfg)
    behavior_text = {"refuse": "refuse", "comply": "help with (comply)"}

    async def one(p: Prompt):
        msgs = [{"role": "user", "content": tpl.render(
            prompt=p.turns[0], expected_behavior_text=behavior_text[p.expected_behavior])}]
        return await structured_call(client, tx.tier_screen_model, msgs, TierScreen, params)

    results = await asyncio.gather(*(one(prompts[i]) for i in todo))
    out = list(prompts)
    for i, res in zip(todo, results, strict=True):
        if res.status != "ok":
            update = {"tier": "borderline", "tier_reason": f"screen failed: {res.status}"}
        elif res.value.borderline:
            update = {"tier": "borderline", "tier_reason": f"screen: {res.value.rationale[:200]}"}
        else:
            continue
        out[i] = prompts[i].model_copy(update=update)
    return out
