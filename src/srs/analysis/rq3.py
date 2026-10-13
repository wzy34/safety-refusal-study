"""RQ3: does framing change refusal? On the RQ3 split and separately per expected behavior:
refusal rate per model x framing, paired flip rates against `direct` (exact McNemar, Holm),
and intent consistency (same decision under all five framings)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from srs.analysis import plots
from srs.analysis.data import AnalysisData
from srs.analysis.metrics import decisions
from srs.analysis.stats import holm, mcnemar_exact, wilson
from srs.config import FRAMINGS, StudyConfig


def rq3_decisions(data: AnalysisData, cfg: StudyConfig, evaluator: str) -> pd.DataFrame:
    a = cfg.experiment.analysis
    resp = data.resp[(data.resp["split"] == "rq3") & (data.resp["sample_idx"] == 0)]
    dec = decisions(resp, data.judg, evaluator, a.primary_rule)
    return dec[dec["valid"] & ((dec["tier"] == "clear") | a.include_borderline)]


def refusal_rates(dec: pd.DataFrame, level: float) -> pd.DataFrame:
    rows = []
    for (m, exp, fr), g in dec.groupby(["model_id", "expected_behavior", "framing"]):
        k, n = int(g["refused"].astype(bool).sum()), len(g)
        lo, hi = wilson(k, n, level)
        rows.append({"model_id": m, "expected_behavior": exp, "framing": fr, "n": n,
                     "refusal_rate": k / n, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def flip_rates(dec: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, exp), g in dec.groupby(["model_id", "expected_behavior"]):
        wide = g.pivot_table(index="intent_id", columns="framing", values="refused",
                             aggfunc="first")
        if "direct" not in wide:
            continue
        for fr in [f for f in FRAMINGS if f != "direct" and f in wide]:
            pair = wide[["direct", fr]].dropna()
            d, f = pair["direct"].astype(bool), pair[fr].astype(bool)
            res = mcnemar_exact(d.tolist(), f.tolist())
            rows.append({"model_id": m, "expected_behavior": exp, "framing": fr,
                         "n_intents": len(pair),
                         "flip_rate": float((d != f).mean()) if len(pair) else float("nan"),
                         "refused_direct_only": res["a_only"],
                         "refused_framing_only": res["b_only"], "p_value": res["p_value"]})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = out.groupby("expected_behavior")["p_value"].transform(
            lambda p: holm(list(p)))
    return out


def consistency(dec: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, exp), g in dec.groupby(["model_id", "expected_behavior"]):
        wide = g.pivot_table(index="intent_id", columns="framing", values="refused",
                             aggfunc="first").reindex(columns=list(FRAMINGS)).dropna()
        same = wide.astype(bool).nunique(axis=1) == 1
        rows.append({"model_id": m, "expected_behavior": exp, "n_intents": len(wide),
                     "consistency": float(same.mean()) if len(wide) else float("nan")})
    return pd.DataFrame(rows)


def run(data: AnalysisData, cfg: StudyConfig, out: Path) -> list[Path]:
    a = cfg.experiment.analysis
    out.mkdir(parents=True, exist_ok=True)
    written = []
    evaluators = sorted(set(data.judg["evaluator_id"])) if not data.judg.empty else []
    for ev in evaluators:
        tag = "primary" if ev == a.primary_evaluator else ev
        dec = rq3_decisions(data, cfg, ev)
        for name, df in (("refusal_rates", refusal_rates(dec, a.ci_level)),
                         ("flip_rates", flip_rates(dec)), ("consistency", consistency(dec))):
            df.to_csv(p := out / f"rq3_{name}_{tag}.csv", index=False)
            written.append(p)
        if ev == a.primary_evaluator:
            fl = flip_rates(dec)
            for exp in ("refuse", "comply"):
                sub = fl[fl["expected_behavior"] == exp] if not fl.empty else fl
                if sub.empty:
                    continue
                table = sub.pivot(index="model_id", columns="framing", values="flip_rate")
                written += plots.heatmap(table, f"Flip rate vs direct, {exp}-expected "
                                         f"({ev}, {a.primary_rule})",
                                         out / f"rq3_flip_heatmap_{exp}", cbar_label="flip rate")
    return written
