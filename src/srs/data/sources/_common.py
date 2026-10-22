"""Shared helpers for benchmark loaders. Loaders return direct-framing `Prompt` records whose
`category` is UNASSIGNED; the taxonomy step fills it in."""

from __future__ import annotations

from srs import ids
from srs.schemas import ExpectedBehavior, Prompt

UNASSIGNED = "__unassigned__"


def text_id(text: str) -> str:
    """Stable source id for datasets without row ids: a hash of the normalized text."""
    return ids.short_hash(" ".join(text.split()))


def make_prompt(source: str, source_id: str, text: str, expected: ExpectedBehavior,
                source_category: str | None, pair_id: str | None = None) -> Prompt:
    turns = [text.strip()]
    return Prompt(
        prompt_id=ids.prompt_id(source, source_id, "direct", turns),
        intent_id=ids.intent_id(source, source_id),
        source=source,
        source_id=str(source_id),
        source_category=source_category,
        category=UNASSIGNED,
        expected_behavior=expected,
        framing="direct",
        turns=turns,
        pair_id=pair_id,
    )
