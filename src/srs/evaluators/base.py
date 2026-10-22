"""Evaluator interface (design D6), blinded evaluation items, the evaluate stage and the rules
for reducing 3-class refusal labels to yes/no.

An `EvalItem` holds only the conversation: user turns, earlier assistant replies (multi-turn)
and the final answer. Model identity, expected behavior, category and reasoning text never
reach an evaluator; the item id is bookkeeping only and is never rendered into any input.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from srs.config import PartialRule, StudyConfig
from srs.data.build import processed_dir
from srs.io import read_records, upsert_records
from srs.schemas import Judgment, JudgmentTarget, ReferenceItem, Response

log = logging.getLogger(__name__)

EVAL_CHUNK = 1000  # judgments are written to disk after every chunk of items


@dataclass(frozen=True)
class Turn:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class EvalItem:
    item_id: str
    turns: tuple[Turn, ...]  # alternating user/assistant, ending with the final answer

    @property
    def final_answer(self) -> str:
        return self.turns[-1].content

    @property
    def final_user(self) -> str:
        return next(t.content for t in reversed(self.turns) if t.role == "user")

    @property
    def context(self) -> tuple[Turn, ...]:
        """Turns before the final user message (empty for single-turn conversations)."""
        last_user = max(i for i, t in enumerate(self.turns) if t.role == "user")
        return self.turns[:last_user]

    def render(self) -> str:
        return "\n\n".join(f"[{'User' if t.role == 'user' else 'Assistant'}]\n{t.content}"
                           for t in self.turns)


def item_from_response(r: Response) -> EvalItem:
    turns = [Turn(m.role, m.content) for m in r.transcript if m.role in ("user", "assistant")]
    if not turns or turns[-1].role != "assistant":
        raise ValueError(f"response {r.response_id} has no final answer")
    turns[-1] = Turn("assistant", r.final_text or "")
    return EvalItem(r.response_id, tuple(turns))


def item_from_reference(ref: ReferenceItem) -> EvalItem:
    return EvalItem(ref.item_id, (Turn("user", ref.prompt), Turn("assistant", ref.response)))


class Evaluator(Protocol):
    id: str

    def version(self) -> str: ...

    def evaluate(self, items: list[EvalItem]) -> list[Judgment]:
        """Judge each item. Items whose judging failed transiently (e.g. API errors after
        retries) are omitted, so the next run retries them."""
        ...


# --- yes/no reduction ------------------------------------------------------------------------


def is_refusal(label: str | None, rule: PartialRule) -> bool | None:
    """Reduce a 3-class refusal label to refused yes/no under a named rule."""
    if label is None:
        return None
    if label == "partial":
        return rule == "partial_as_refusal"
    return label == "full_refusal"


# --- evaluate stage --------------------------------------------------------------------------


def judgments_path(cfg: StudyConfig, target: JudgmentTarget) -> Path:
    name = "judgments.parquet" if target == "responses" else "judgments_external.parquet"
    return processed_dir(cfg) / name


def judgment_key(j: Judgment) -> tuple[str, str, str]:
    return (j.item_id, j.evaluator_id, j.evaluator_version)


def eligible_responses(responses: list[Response]) -> tuple[list[EvalItem], Counter]:
    """Only `ok` responses are judged; blocked and errored ones are counted separately."""
    skipped: Counter = Counter()
    items = []
    for r in responses:
        if r.status != "ok":
            skipped[f"skipped_{r.status}"] += 1
            continue
        items.append(item_from_response(r))
    return items, skipped


def run_evaluators(evaluators: list[Evaluator], items: list[EvalItem], out_path: Path,
                   target: JudgmentTarget, run_id: str | None = None) -> dict[str, int]:
    """Judge items with each evaluator, skipping (item, evaluator, version) triples already
    in `out_path`. Writes after every `EVAL_CHUNK` items, so an interrupted run resumes.
    Returns counts per evaluator and status."""
    counts: Counter = Counter()
    for ev in evaluators:
        version = ev.version()
        done = {judgment_key(j) for j in read_records(out_path, Judgment)}
        todo = [it for it in items if (it.item_id, ev.id, version) not in done]
        counts[f"{ev.id}:skipped_done"] = len(items) - len(todo)
        if not todo:
            continue
        n_new = 0
        for start in range(0, len(todo), EVAL_CHUNK):
            chunk = todo[start:start + EVAL_CHUNK]
            new = ev.evaluate(chunk)
            for j in new:
                j.target = target
                j.run_id = run_id
                counts[f"{ev.id}:{j.status}"] += 1
            counts[f"{ev.id}:not_judged"] += len(chunk) - len(new)
            n_new += len(new)
            upsert_records(out_path, Judgment, new, key=judgment_key)
            log.info("evaluator %s v%s: %d/%d done", ev.id, version, start + len(chunk),
                     len(todo))
        log.info("evaluator %s v%s: %d judged, %d not judged", ev.id, version, n_new,
                 len(todo) - n_new)
    return {k: v for k, v in counts.items() if v}
