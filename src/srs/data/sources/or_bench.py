"""OR-Bench (Cui et al., 2024): Hard-1K seemingly-toxic but safe prompts (comply) and Toxic
prompts (refuse). Rows have no ids, so source ids hash the prompt text."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from srs.data.sources._common import make_prompt, text_id
from srs.schemas import Prompt

FILES = {"or_bench_hard": ("or-bench-hard-1k.csv", "comply"),
         "or_bench_toxic": ("or-bench-toxic.csv", "refuse")}


def load(raw_dir: Path) -> list[Prompt]:
    out = []
    for source, (fname, expected) in FILES.items():
        df = pd.read_csv(raw_dir / "or_bench" / fname)
        seen: set[str] = set()
        for row in df.itertuples():
            sid = text_id(row.prompt)
            if sid in seen:  # exact duplicate rows within a file
                continue
            seen.add(sid)
            out.append(make_prompt(source, sid, row.prompt, expected, str(row.category)))
    return out
