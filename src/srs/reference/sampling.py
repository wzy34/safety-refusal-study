"""Stratified self-annotation sample with known inclusion probabilities.

Cells are model x expected behavior x framing x evaluator disagreement. The sample size is
allocated across cells in proportion to cell size, with disagreement cells up-weighted by
`annotation.disagreement_oversample`; within a cell, items are drawn without replacement in a
seeded hash order. Each item's inclusion probability is therefore exactly n_cell / N_cell, and
analyses weight items by its inverse.

Outputs (in results/annotations/<name>/):
  items.parquet   blinded content only (item id + conversation turns), read by the app
  sample.parquet  sampling metadata (cell, inclusion probability), never shown in the app
  sample.json     seed, sizes and allocation
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from srs.config import StudyConfig
from srs.evaluators.base import is_refusal, item_from_response
from srs.ids import sha256_hex
from srs.schemas import Judgment, Prompt, Response


def annotation_dir(cfg: StudyConfig, name: str) -> Path:
    return cfg.path("results_dir") / "annotations" / name


@dataclass(frozen=True)
class Unit:
    response: Response
    cell: tuple[str, str, str, bool]  # model, expected behavior, framing, disagreement


def disagreement(judgments: list[Judgment]) -> bool:
    """True if evaluators with a valid label disagree on refused yes/no."""
    votes = {is_refusal(j.refusal, "partial_as_refusal") for j in judgments
             if j.status == "ok" and j.refusal is not None}
    return len(votes) > 1


def eligible_units(responses: list[Response], prompts: list[Prompt], judgments: list[Judgment],
                   splits: tuple[str, ...]) -> list[Unit]:
    """Primary (sample 0), `ok` responses to prompts in the given splits."""
    pmap = {p.prompt_id: p for p in prompts}
    by_item: dict[str, list[Judgment]] = defaultdict(list)
    for j in judgments:
        by_item[j.item_id].append(j)
    units = []
    for r in responses:
        p = pmap.get(r.prompt_id)
        if r.status != "ok" or r.sample_idx != 0 or p is None or p.split not in splits:
            continue
        units.append(Unit(r, (r.model_id, p.expected_behavior, p.framing,
                              disagreement(by_item[r.response_id]))))
    return units


def allocate(sizes: dict[Any, int], weights: dict[Any, float], n: int) -> dict[Any, int]:
    """Allocate n across cells in proportion to size x weight (largest remainder), never
    exceeding a cell's size; shortfalls are redistributed."""
    n = min(n, sum(sizes.values()))
    alloc = {k: 0 for k in sizes}
    remaining = n
    while remaining > 0:
        open_ = {k: sizes[k] - alloc[k] for k in sizes if sizes[k] > alloc[k]}
        total = sum(open_[k] * weights[k] for k in open_)
        quotas = {k: remaining * open_[k] * weights[k] / total for k in open_}
        add = {k: min(int(q), open_[k]) for k, q in quotas.items()}
        given = sum(add.values())
        for k in sorted(quotas, key=lambda k: (-(quotas[k] - int(quotas[k])), str(k))):
            if given >= remaining:
                break
            if add[k] < open_[k]:
                add[k] += 1
                given += 1
        for k, a in add.items():
            alloc[k] += a
        remaining -= given
    return alloc


def draw_sample(units: list[Unit], n: int, seed: int, oversample: float
                ) -> list[tuple[Unit, float]]:
    """Return (unit, inclusion probability) pairs."""
    cells: dict[tuple, list[Unit]] = defaultdict(list)
    for u in units:
        cells[u.cell].append(u)
    sizes = {c: len(v) for c, v in cells.items()}
    weights = {c: (oversample if c[3] else 1.0) for c in cells}
    alloc = allocate(sizes, weights, n)
    out = []
    for c, members in sorted(cells.items(), key=lambda kv: str(kv[0])):
        members.sort(key=lambda u: sha256_hex(f"{seed}:annotation:{u.response.response_id}"))
        k = alloc[c]
        out += [(u, k / sizes[c]) for u in members[:k]]
    return out


def write_sample(out_dir: Path, sample: list[tuple[Unit, float]], seed: int, n_eligible: int,
                 name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    meta = []
    for u, prob in sample:
        item = item_from_response(u.response)
        items.append({"item_id": item.item_id,
                      "turns": [{"role": t.role, "content": t.content} for t in item.turns]})
        model, behavior, framing, disagree = u.cell
        meta.append({"item_id": item.item_id, "model_id": model, "expected_behavior": behavior,
                     "framing": framing, "disagreement": disagree, "inclusion_prob": prob,
                     "weight": 1.0 / prob})
    pd.DataFrame(items).to_parquet(out_dir / "items.parquet")
    pd.DataFrame(meta).to_parquet(out_dir / "sample.parquet")
    (out_dir / "sample.json").write_text(json.dumps({
        "name": name, "seed": seed, "n_sampled": len(sample), "n_eligible": n_eligible,
        "n_disagreement": sum(m["disagreement"] for m in meta)}, indent=2))
