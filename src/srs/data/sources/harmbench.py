"""HarmBench text behaviors (Mazeika et al., 2024), refuse-expected. Only the `standard`
functional category is used: `contextual` needs its context string and `copyright` measures a
different notion of harm."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from srs.data.sources._common import make_prompt
from srs.schemas import Prompt

SOURCE = "harmbench"


def load(raw_dir: Path) -> list[Prompt]:
    df = pd.read_csv(raw_dir / "harmbench" / "harmbench_behaviors_text_all.csv")
    df = df[df.FunctionalCategory == "standard"]
    return [make_prompt(SOURCE, row.BehaviorID, row.Behavior, "refuse", row.SemanticCategory)
            for row in df.itertuples()]
