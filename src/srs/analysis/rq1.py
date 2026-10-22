"""RQ1: do models differ in under- and over-refusal? Per-model metrics, pairwise exact
McNemar tests on shared prompts with Holm correction, the FAR-ORR rank correlation, and the
same metric table under every other evaluator (sensitivity)."""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

from srs.analysis.data import AnalysisData
from srs.analysis.metrics import compute_metrics, decisions, primary_rows
from srs.analysis.stats import holm, mcnemar_exact, spearman
from srs.config import StudyConfig

OTHER_RULE = {"partial_as_refusal": "partial_as_compliance",
              "partial_as_compliance": "partial_as_refusal"}


def model_metrics(data: AnalysisData, cfg: StudyConfig, evaluator: str, rule: str,
                  include_borderline: bool | None = None) -> pd.DataFrame:
    a = cfg.experiment.analysis
    dec = decisions(primary_rows(data.resp, a.primary_splits), data.judg, evaluator, rule)  # type: ignore[arg-type]
    return compute_metrics(dec, ("model_id",),
                           a.include_borderline if include_borderline is None
                           else include_borderline,
                           a.ci_level, a.bootstrap_resamples, cfg.experiment.seed)


def pairwise_tests(dec: pd.DataFrame, include_borderline: bool = False) -> pd.DataFrame:
    """Exact McNemar per model pair on shared prompts, for FAR and for ORR."""
    d = dec[dec["valid"]] if include_borderline else dec[dec["valid"] & (dec["tier"] == "clear")]
    rows = []
    for metric, expected, outcome in (("far_comply", "refuse", False), ("orr", "comply", True)):
        sub = d[d["expected_behavior"] == expected]
        wide = sub.pivot_table(index="prompt_id", columns="model_id", values="refused",
                               aggfunc="first")
        for a, b in itertools.combinations(sorted(wide.columns), 2):
            pair = wide[[a, b]].dropna()
            ya = (pair[a].astype(bool) == outcome).tolist()
            yb = (pair[b].astype(bool) == outcome).tolist()
            res = mcnemar_exact(ya, yb)
            rows.append({"metric": metric, "model_a": a, "model_b": b,
                         "rate_a": sum(ya) / len(ya) if ya else float("nan"),
                         "rate_b": sum(yb) / len(yb) if yb else float("nan"), **res})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = out.groupby("metric")["p_value"].transform(lambda p: holm(list(p)))
    return out


def run(data: AnalysisData, cfg: StudyConfig, out: Path) -> list[Path]:
    a = cfg.experiment.analysis
    primary, rule = a.primary_evaluator, a.primary_rule
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for ev in sorted(set(data.judg["evaluator_id"])) if not data.judg.empty else [primary]:
        for r in (rule, OTHER_RULE[rule]):
            m = model_metrics(data, cfg, ev, r)
            tag = "primary" if (ev, r) == (primary, rule) else f"{ev}_{r}"
            path = out / f"rq1_metrics_{tag}.csv"
            m.to_csv(path, index=False)
            written.append(path)
    dec = decisions(primary_rows(data.resp, a.primary_splits), data.judg, primary, rule)
    pw = pairwise_tests(dec, a.include_borderline)
    pw.to_csv(p := out / "rq1_pairwise.csv", index=False)
    written.append(p)
    m = model_metrics(data, cfg, primary, rule)
    ok = m.dropna(subset=["far_comply", "orr"])
    corr = spearman(ok["far_comply"], ok["orr"]) if len(ok) >= 3 else {"rho": float("nan")}
    pd.DataFrame([{"evaluator_id": primary, "rule": rule, **corr}]).to_csv(
        p := out / "rq1_far_orr_correlation.csv", index=False)
    written.append(p)
    return written
