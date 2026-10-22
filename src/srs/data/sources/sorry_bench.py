"""SORRY-Bench 2024-06 (Xie et al., 2024): 440 refuse-expected requests over 44 categories.
Only the `base` prompt style is used (the other styles are mutations, not new intents)."""

from __future__ import annotations

import json
from pathlib import Path

from srs.data.sources._common import make_prompt
from srs.schemas import Prompt

SOURCE = "sorry_bench"


def load(raw_dir: Path) -> list[Prompt]:
    out = []
    with open(raw_dir / "sorry_bench" / "question.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            if q.get("prompt_style", "base") != "base":
                continue
            out.append(make_prompt(SOURCE, str(q["question_id"]), q["turns"][0], "refuse",
                                   str(q["category"])))
    return out
