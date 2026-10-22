"""XSTest (Röttger et al., 2024): 250 safe prompts (comply) and 200 unsafe contrast prompts
(refuse). Types are prompt *styles*, not risk categories, so categories are LLM-assigned.

Contrast prompts mirror safe prompts position-by-position within their type; those pairs share
a pair id. `nons_group_real_discr` and `privacy_public` have no contrast counterpart.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from srs import ids
from srs.data.sources._common import make_prompt
from srs.schemas import Prompt

SOURCE = "xstest"
CONTRAST_OF = {
    "contrast_homonyms": "homonyms",
    "contrast_figurative_language": "figurative_language",
    "contrast_safe_targets": "safe_targets",
    "contrast_safe_contexts": "safe_contexts",
    "contrast_definitions": "definitions",
    "contrast_historical_events": "historical_events",
    "contrast_discr": "real_group_nons_discr",
    "contrast_privacy": "privacy_fictional",
}
_PAIRED_SAFE = {v: k for k, v in CONTRAST_OF.items()}


def load(raw_dir: Path) -> list[Prompt]:
    df = pd.read_csv(raw_dir / "xstest" / "xstest_prompts.csv")
    df["pos"] = df.groupby("type").cumcount()
    out = []
    for row in df.itertuples():
        label = str(row.label).strip()
        if label not in ("safe", "unsafe"):
            raise ValueError(f"xstest id {row.id}: unknown label {label!r}")
        base = CONTRAST_OF.get(row.type, row.type if row.type in _PAIRED_SAFE else None)
        pair = ids.pair_id(SOURCE, base, str(row.pos)) if base else None
        out.append(make_prompt(SOURCE, str(row.id), row.prompt,
                               "comply" if label == "safe" else "refuse", None, pair))
    return out
