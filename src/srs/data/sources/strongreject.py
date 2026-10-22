"""StrongREJECT (Souly et al., 2024): 313 forbidden prompts (refuse) in 6 categories."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from srs.data.sources._common import make_prompt, text_id
from srs.schemas import Prompt

SOURCE = "strongreject"


def load(raw_dir: Path) -> list[Prompt]:
    df = pd.read_csv(raw_dir / "strongreject" / "strongreject_dataset.csv")
    return [make_prompt(SOURCE, text_id(row.forbidden_prompt), row.forbidden_prompt, "refuse",
                        row.category)
            for row in df.itertuples()]
