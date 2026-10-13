"""RQ2: are weaknesses category-specific? FAR/ORR per model x category (heatmaps), Kendall's
W across models, and a logistic GLM of refusal on model, expectation, category and their
two-way interactions with standard errors clustered by intent."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from srs.analysis import plots
from srs.analysis.data import AnalysisData
from srs.analysis.metrics import compute_metrics, decisions, primary_rows
from srs.analysis.stats import clustered_logit, kendalls_w
from srs.config import StudyConfig

FORMULA = "y ~ (C(model_id) + C(expected_behavior) + C(category)) ** 2"


def category_metrics(data: AnalysisData, cfg: StudyConfig, evaluator: str) -> pd.DataFrame:
    a = cfg.experiment.analysis
    dec = decisions(primary_rows(data.resp), data.judg, evaluator, a.primary_rule)
    return compute_metrics(dec, ("model_id", "category"), a.include_borderline, a.ci_level,
                           a.bootstrap_resamples, cfg.experiment.seed)


def run(data: AnalysisData, cfg: StudyConfig, out: Path) -> list[Path]:
    a = cfg.experiment.analysis
    out.mkdir(parents=True, exist_ok=True)
    written = []
    evaluators = sorted(set(data.judg["evaluator_id"])) if not data.judg.empty else []
    for ev in evaluators:
        cm = category_metrics(data, cfg, ev)
        tag = "primary" if ev == a.primary_evaluator else ev
        cm.to_csv(p := out / f"rq2_category_metrics_{tag}.csv", index=False)
        written.append(p)
    cm = category_metrics(data, cfg, a.primary_evaluator)
    w_rows = []
    for metric, label in (("far_comply", "FAR"), ("orr", "ORR")):
        table = cm.pivot(index="model_id", columns="category", values=metric)
        written += plots.heatmap(table, f"{label} by model and category "
                                 f"({a.primary_evaluator}, {a.primary_rule})",
                                 out / f"rq2_heatmap_{metric}", cbar_label=label)
        w_rows.append({"metric": metric, **kendalls_w(table)})
    pd.DataFrame(w_rows).to_csv(p := out / "rq2_kendalls_w.csv", index=False)
    written.append(p)

    dec = decisions(primary_rows(data.resp), data.judg, a.primary_evaluator, a.primary_rule)
    d = dec[dec["valid"] & ((dec["tier"] == "clear") | a.include_borderline)].copy()
    d["y"] = d["refused"].astype(int)
    try:
        tab = clustered_logit(d, FORMULA, "intent_id")
        tab.insert(0, "formula", FORMULA)
    except Exception as e:  # e.g. perfect separation in sparse cells
        tab = pd.DataFrame([{"formula": FORMULA, "error": f"{type(e).__name__}: {e}"}])
    tab.to_csv(p := out / "rq2_glm.csv", index=False)
    written.append(p)
    return written
