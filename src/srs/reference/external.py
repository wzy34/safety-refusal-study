"""External human-labeled reference sets, normalized to `ReferenceItem` with the shared label
vocabulary (refusal: full_compliance / partial / full_refusal; harmful: bool). Labels a set
does not provide are null.

| source               | refusal label                  | harmful label                     |
|----------------------|--------------------------------|-----------------------------------|
| wildguardtest        | response_refusal_label         | response_harm_label               |
| jbb_judge            | -                              | human_majority (jailbroken)       |
| harmbench_val        | -                              | majority of human_0..2            |
| xstest_completions   | final_label (3-class)          | -                                 |
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from srs import ids
from srs.data.sources import XSTEST_COMPLETION_MODELS, FetchError, missing_raw_error
from srs.schemas import ReferenceItem

# Published sizes, used to sanity-check the loaded counts.
PUBLISHED_SIZES = {"wildguardtest": 1725, "jbb_judge": 300, "harmbench_val": 602,
                   "xstest_completions": 450 * len(XSTEST_COMPLETION_MODELS)}

_WG_REFUSAL = {"refusal": "full_refusal", "compliance": "full_compliance"}
_WG_HARM = {"harmful": True, "unharmful": False}
_XS_LABEL = {"1_full_compliance": "full_compliance", "2_full_refusal": "full_refusal",
             "3_partial_refusal": "partial"}


def _item(source: str, source_id: str, prompt: str, response: str, **labels) -> ReferenceItem:
    return ReferenceItem(item_id=ids.short_hash({"kind": "ref", "source": source,
                                                 "source_id": source_id}),
                         source=source, source_id=source_id, prompt=str(prompt),
                         response=str(response), **labels)


def _none_if_na(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


def load_wildguardtest(raw_dir: Path) -> list[ReferenceItem]:
    df = pd.read_parquet(raw_dir / "wildguardmix" / "wildguard_test.parquet")
    out = []
    for i, row in enumerate(df.to_dict("records")):
        response = _none_if_na(row.get("response"))
        if response is None:
            continue
        refusal = _WG_REFUSAL.get(_none_if_na(row.get("response_refusal_label")))
        harm = _WG_HARM.get(_none_if_na(row.get("response_harm_label")))
        prompt_harm = _WG_HARM.get(_none_if_na(row.get("prompt_harm_label")))
        out.append(_item("wildguardtest", str(i), row["prompt"], response, refusal=refusal,
                         harmful=harm, prompt_harmful=prompt_harm))
    return out


def load_jbb_judge(raw_dir: Path) -> list[ReferenceItem]:
    df = pd.read_csv(raw_dir / "jbb" / "judge-comparison.csv")
    return [_item("jbb_judge", str(r["Index"]), r["prompt"], r["target_response"],
                  harmful=bool(int(r["human_majority"])), prompt_harmful=True)
            for r in df.to_dict("records")]


def load_harmbench_val(raw_dir: Path) -> list[ReferenceItem]:
    data = json.loads((raw_dir / "harmbench" / "text_behaviors_val_set.json").read_text())
    out = []
    for behavior_id, rows in data.items():
        for k, r in enumerate(rows):
            votes = [int(r[f"human_{h}"]) for h in range(3)]
            out.append(_item("harmbench_val", f"{behavior_id}:{k}", r["test_case"],
                             r["generation"], harmful=sum(votes) >= 2, prompt_harmful=True))
    return out


def load_xstest_completions(raw_dir: Path) -> list[ReferenceItem]:
    out = []
    for model in XSTEST_COMPLETION_MODELS:
        df = pd.read_csv(raw_dir / "xstest_completions" / f"xstest_v2_completions_{model}.csv")
        for r in df.to_dict("records"):
            label = str(r["final_label"]).strip()
            if label not in _XS_LABEL:
                raise ValueError(f"xstest_completions {model}/{r['id']}: label {label!r}")
            out.append(_item("xstest_completions", f"{model}:{r['id']}", r["prompt"],
                             r["completion"], refusal=_XS_LABEL[label],
                             prompt_harmful=str(r["type"]).startswith("contrast_")))
    return out


REFERENCE_LOADERS: dict[str, tuple[str, Callable[[Path], list[ReferenceItem]]]] = {
    # reference source -> (raw source it comes from, loader)
    "wildguardtest": ("wildguardmix", load_wildguardtest),
    "jbb_judge": ("jbb", load_jbb_judge),
    "harmbench_val": ("harmbench", load_harmbench_val),
    "xstest_completions": ("xstest_completions", load_xstest_completions),
}


def load_reference_items(raw_dir: Path, require_all: bool = True
                         ) -> tuple[list[ReferenceItem], dict[str, str]]:
    """Load every reference set. Returns (items, {source: error}) for sets that could not be
    loaded; with require_all, the first missing set raises its access-step error instead."""
    items: list[ReferenceItem] = []
    missing: dict[str, str] = {}
    for name, (raw_name, load) in REFERENCE_LOADERS.items():
        try:
            items += load(raw_dir)
        except FileNotFoundError as e:
            err: FetchError = missing_raw_error(raw_name, e.filename)
            if require_all:
                raise err from e
            missing[name] = str(err)
    return items, missing


def overlap_key(prompt: str, response: str) -> str:
    """Normalized prompt+response hash used by the classifier leakage guard."""
    norm = lambda s: " ".join(str(s).lower().split())  # noqa: E731
    return ids.sha256_hex({"p": norm(prompt), "r": norm(response)})
