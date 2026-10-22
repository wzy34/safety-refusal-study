"""Dataset build: load -> map categories -> deduplicate -> LLM categories -> tiers -> splits ->
`prompts.parquet` plus a build report.

Splits are drawn per intent with a seeded hash order, stratified by expected behavior and
unified category. Pilot and RQ3 are balanced between expected behaviors and drawn from
`clear`-tier intents (borderline prompts are excluded from primary metrics, so spending pilot
or 5x framing budget on them would be wasted); `main` takes the remaining intents.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from srs.config import SplitSettings, StudyConfig
from srs.data.sources import benchmark_loaders, missing_raw_error
from srs.data.taxonomy import assign_llm_categories, map_categories
from srs.data.tiers import apply_tier_rules, screen_tiers
from srs.ids import sha256_hex
from srs.io import read_records, write_records
from srs.llm.client import ChatClient
from srs.schemas import Prompt

# Curated / hand-written sources win when the same text appears in several benchmarks.
SOURCE_PRIORITY = ("xstest", "jbb_harmful", "jbb_benign", "harmbench", "strongreject",
                   "sorry_bench", "or_bench_toxic", "or_bench_hard")


def processed_dir(cfg: StudyConfig) -> Path:
    return cfg.path("data_dir") / "processed"


def prompts_path(cfg: StudyConfig) -> Path:
    return processed_dir(cfg) / "prompts.parquet"


# --- deduplication ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def deduplicate(prompts: list[Prompt]) -> tuple[list[Prompt], list[dict[str, Any]]]:
    """Drop repeated texts, keeping the highest-priority source. If duplicates disagree on
    expected behavior, every copy is dropped (the label is contested)."""
    rank = {s: i for i, s in enumerate(SOURCE_PRIORITY)}
    groups: dict[str, list[Prompt]] = defaultdict(list)
    for p in prompts:
        groups[normalize_text(" ".join(p.turns))].append(p)
    kept, removed = [], []
    for group in groups.values():
        group.sort(key=lambda p: (rank.get(p.source, len(rank)), p.source_id))
        if len({p.expected_behavior for p in group}) > 1:
            removed += [{"source": p.source, "source_id": p.source_id, "reason":
                         "duplicate with conflicting expected behavior"} for p in group]
            continue
        kept.append(group[0])
        removed += [{"source": p.source, "source_id": p.source_id,
                     "reason": f"duplicate of {group[0].source}/{group[0].source_id}"}
                    for p in group[1:]]
    order = {id(p): i for i, p in enumerate(prompts)}
    kept.sort(key=lambda p: order[id(p)])
    return kept, removed


# --- splits ----------------------------------------------------------------------------------


def _order_key(seed: int, split: str, intent_id: str) -> str:
    return sha256_hex(f"{seed}:{split}:{intent_id}")


def stratified_pick[T](items: Sequence[T], n: int, stratum: Callable[[T], Hashable],
                       order: Callable[[T], str]) -> list[T]:
    """Pick n items with proportional allocation across strata (largest remainder), taking
    items within each stratum in `order`. Shortfalls in small strata are redistributed."""
    by: dict[Hashable, list[T]] = defaultdict(list)
    for it in items:
        by[stratum(it)].append(it)
    for v in by.values():
        v.sort(key=order)
    n = min(n, len(items))
    alloc = {k: 0 for k in by}
    remaining = n
    while remaining > 0:
        open_ = {k: len(v) - alloc[k] for k, v in by.items() if len(v) > alloc[k]}
        total = sum(open_.values())
        quotas = {k: remaining * c / total for k, c in open_.items()}
        floors = {k: min(int(q), open_[k]) for k, q in quotas.items()}
        given = sum(floors.values())
        for k in sorted(quotas, key=lambda k: (-(quotas[k] - int(quotas[k])), str(k))):
            if given >= remaining:
                break
            if floors[k] < open_[k]:
                floors[k] += 1
                given += 1
        for k, f in floors.items():
            alloc[k] += f
        remaining -= given
    return [it for k, v in by.items() for it in v[:alloc[k]]]


def balanced_pick(pool: list[Prompt], n: int, seed: int, split: str) -> list[Prompt]:
    picked = []
    halves = {"refuse": n - n // 2, "comply": n // 2}
    for behavior, k in halves.items():
        cands = [p for p in pool if p.expected_behavior == behavior]
        picked += stratified_pick(cands, k, lambda p: p.category,
                                  lambda p: _order_key(seed, split, p.intent_id))
    return picked


def assign_splits(prompts: list[Prompt], settings: SplitSettings, seed: int) -> list[Prompt]:
    """Assign each (direct-framing) intent to pilot, rq3, main or None (unused)."""
    split_of: dict[str, str | None] = {}
    clear = [p for p in prompts if p.tier == "clear"]
    for p in balanced_pick(clear, settings.pilot_intents, seed, "pilot"):
        split_of[p.intent_id] = "pilot"
    rest_clear = [p for p in clear if p.intent_id not in split_of]
    for p in balanced_pick(rest_clear, settings.rq3_intents, seed, "rq3"):
        split_of[p.intent_id] = "rq3"
    rest = [p for p in prompts if p.intent_id not in split_of]
    if settings.main_intents is None:
        main = rest
    else:
        main = stratified_pick(rest, settings.main_intents,
                               lambda p: (p.expected_behavior, p.category),
                               lambda p: _order_key(seed, "main", p.intent_id))
    for p in main:
        split_of[p.intent_id] = "main"
    return [p.model_copy(update={"split": split_of.get(p.intent_id)}) for p in prompts]


# --- report ----------------------------------------------------------------------------------


def _counts(prompts: list[Prompt], *fields: str) -> dict[str, int]:
    c = Counter(" / ".join(str(getattr(p, f)) for f in fields) for p in prompts)
    return dict(sorted(c.items()))


def build_report(prompts: list[Prompt], dedup_removed: list[dict], dropped: list[dict]) -> dict:
    direct = [p for p in prompts if p.framing == "direct"]
    return {
        "n_prompts": len(prompts),
        "n_intents": len({p.intent_id for p in prompts}),
        "by_source": _counts(direct, "source"),
        "by_split": _counts(direct, "split"),
        "by_expected_behavior": _counts(direct, "expected_behavior"),
        "by_category": _counts(direct, "category"),
        "by_tier": _counts(direct, "tier"),
        "by_framing": _counts(prompts, "framing"),
        "by_split_behavior": _counts(direct, "split", "expected_behavior"),
        "by_split_tier": _counts(direct, "split", "tier"),
        "llm_assigned_categories": sum(p.category_llm_assigned for p in direct),
        "dedup_removed": {"count": len(dedup_removed), "items": dedup_removed},
        "dropped_intents": {"count": len(dropped), "items": dropped},
    }


def _md_table(title: str, counts: dict[str, int]) -> str:
    rows = "\n".join(f"| {k} | {v} |" for k, v in counts.items())
    return f"### {title}\n\n| value | count |\n|---|---|\n{rows}\n"


def write_report(cfg: StudyConfig, prompts: list[Prompt]) -> Path:
    out = processed_dir(cfg)
    dedup = _read_json(out / "dedup_removed.json")
    dropped = _read_json(out / "framing_dropped.json")
    rep = build_report(prompts, dedup, dropped)
    (out / "build_report.json").write_text(json.dumps(rep, indent=2))
    parts = [f"# Dataset build report\n\n{rep['n_prompts']} prompts, {rep['n_intents']} intents, "
             f"{rep['llm_assigned_categories']} LLM-assigned categories.\n"]
    for key in ("by_source", "by_split", "by_expected_behavior", "by_category", "by_tier",
                "by_framing", "by_split_behavior", "by_split_tier"):
        parts.append(_md_table(key.replace("_", " "), rep[key]))
    parts.append(f"### Deduplication removals: {len(dedup)}\n\n" + "\n".join(
        f"- {d['source']}/{d['source_id']}: {d['reason']}" for d in dedup) + "\n")
    parts.append(f"### Intents dropped by the intent check: {len(dropped)}\n\n" + "\n".join(
        f"- {d['intent_id']} ({d['source']}/{d['source_id']}): failed {d['framing']}"
        for d in dropped) + "\n")
    path = out / "build_report.md"
    path.write_text("\n".join(parts))
    return path


def _read_json(path: Path) -> list:
    return json.loads(path.read_text()) if path.exists() else []


# --- pipeline --------------------------------------------------------------------------------


@dataclass
class BuildResult:
    prompts: list[Prompt]
    dedup_removed: list[dict] = field(default_factory=list)
    report_path: Path | None = None


def load_all(raw_dir: Path) -> list[Prompt]:
    prompts: list[Prompt] = []
    for name, load in benchmark_loaders().items():
        try:
            prompts += load(raw_dir)
        except FileNotFoundError as e:
            raise missing_raw_error(name, e.filename) from e
    return prompts


async def build_dataset(cfg: StudyConfig, client: ChatClient,
                        raw_dir: Path | None = None) -> BuildResult:
    raw_dir = raw_dir or cfg.path("data_dir") / "raw"
    prompts = map_categories(load_all(raw_dir), cfg.taxonomy)
    prompts, removed = deduplicate(prompts)
    prompts = await assign_llm_categories(prompts, cfg, client)
    prompts = await screen_tiers(apply_tier_rules(prompts, cfg.taxonomy), cfg, client)
    prompts = assign_splits(prompts, cfg.experiment.splits, cfg.experiment.seed)

    out = processed_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dedup_removed.json").write_text(json.dumps(removed, indent=2))
    (out / "framing_dropped.json").unlink(missing_ok=True)  # a fresh build has no framings
    write_records(prompts_path(cfg), prompts)
    return BuildResult(prompts, removed, write_report(cfg, prompts))


def load_prompts(cfg: StudyConfig) -> list[Prompt]:
    return read_records(prompts_path(cfg), Prompt)
