"""Unified risk taxonomy: source categories -> unified categories via taxonomy.yaml, and LLM
assignment (flagged) for sources without risk categories."""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from pydantic import BaseModel, create_model

from srs.config import StudyConfig, TaxonomyFile
from srs.data.sources._common import UNASSIGNED
from srs.llm.client import ChatClient
from srs.llm.structured import structured_call, utility_params
from srs.schemas import Prompt
from srs.templates import load_template

log = logging.getLogger(__name__)


class UnmappedCategoryError(Exception):
    def __init__(self, unmapped: list[tuple[str, str | None]]):
        self.unmapped = unmapped
        listing = "\n".join(f"  {src}: {cat!r}" for src, cat in unmapped)
        super().__init__(f"{len(unmapped)} source categories are not mapped in taxonomy.yaml "
                         f"(source_maps):\n{listing}")


class CategoryAssignmentError(Exception):
    pass


def override_key(p: Prompt) -> str:
    return f"{p.source}/{p.source_id}"


def map_categories(prompts: list[Prompt], tx: TaxonomyFile) -> list[Prompt]:
    """Fill `category` from source_maps. Prompts from llm_assigned_sources stay UNASSIGNED.
    Any unmapped (source, source category) fails the build with the full list."""
    out, unmapped = [], set()
    for p in prompts:
        if p.source in tx.llm_assigned_sources:
            out.append(p)
            continue
        uc = tx.source_maps.get(p.source, {}).get(p.source_category or "")
        if uc is None:
            unmapped.add((p.source, p.source_category))
            continue
        out.append(p.model_copy(update={"category": uc, "category_llm_assigned": False}))
    if unmapped:
        raise UnmappedCategoryError(sorted(unmapped, key=lambda x: (x[0], str(x[1]))))
    return out


def _category_output(categories: list[str]) -> type[BaseModel]:
    return create_model("CategoryAssignment", rationale=(str, ...),
                        category=(Literal[tuple(categories)], ...))  # type: ignore[valid-type]


async def assign_llm_categories(prompts: list[Prompt], cfg: StudyConfig,
                                client: ChatClient) -> list[Prompt]:
    """Assign a category with the configured model to every prompt still UNASSIGNED, and flag
    it `category_llm_assigned`. Manual `category_overrides` win over the LLM."""
    tx = cfg.taxonomy
    todo = [i for i, p in enumerate(prompts) if p.category == UNASSIGNED]
    if not todo:
        return prompts
    if tx.category_model is None:
        raise CategoryAssignmentError("taxonomy.yaml: category_model is not set")
    tpl = load_template(cfg.path("prompts_dir"), "category_assign.md")
    cats = "\n".join(f"- {k}: {v}" for k, v in tx.categories.items())
    output = _category_output(list(tx.categories))
    params = utility_params(cfg)

    async def one(p: Prompt) -> tuple[str | None, str]:
        if (o := tx.category_overrides.get(override_key(p))) is not None:
            return o, "override"
        msgs = [{"role": "user", "content": tpl.render(categories=cats, prompt=p.turns[0])}]
        res = await structured_call(client, tx.category_model, msgs, output, params)
        return (res.value.category if res.value else None), res.status  # type: ignore[attr-defined]

    results = await asyncio.gather(*(one(prompts[i]) for i in todo))
    out = list(prompts)
    failures = []
    for i, (cat, status) in zip(todo, results, strict=True):
        if cat is None:
            failures.append(f"  {override_key(prompts[i])} ({status})")
            continue
        out[i] = prompts[i].model_copy(update={
            "category": cat, "category_llm_assigned": True,
            "template_version": tpl.version if status != "override" else None})
    if failures:
        raise CategoryAssignmentError(
            f"{len(failures)} prompts could not be categorized by {tx.category_model}. Add them "
            f"to taxonomy.yaml category_overrides (key: source/source_id):\n" + "\n".join(failures))
    log.info("LLM-assigned categories for %d prompts", len(todo))
    return out
