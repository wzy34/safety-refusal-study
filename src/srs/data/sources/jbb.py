"""JailbreakBench behaviors (Chao et al., 2024): 100 harmful (refuse) and 100 topic-matched
benign (comply) behaviors. Matching rows share `Index`, and therefore a pair id."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from srs import ids
from srs.data.sources._common import make_prompt
from srs.schemas import Prompt

FILES = {"jbb_harmful": ("harmful-behaviors.csv", "refuse"),
         "jbb_benign": ("benign-behaviors.csv", "comply")}


def load(raw_dir: Path) -> list[Prompt]:
    out = []
    for source, (fname, expected) in FILES.items():
        df = pd.read_csv(raw_dir / "jbb" / fname)
        for row in df.to_dict("records"):
            idx = str(row["Index"])
            out.append(make_prompt(source, idx, row["Goal"], expected, row["Category"],
                                   ids.pair_id("jbb", idx)))
    return out
