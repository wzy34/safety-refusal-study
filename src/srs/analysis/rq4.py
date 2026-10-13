"""RQ4: the safety-usability plane. Each model at (FAR, ORR) with CI error bars, Pareto-optimal
models marked, and each model's distance to the origin."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from srs.analysis import plots
from srs.analysis.data import AnalysisData
from srs.analysis.rq1 import model_metrics
from srs.config import StudyConfig


def pareto(points: pd.DataFrame) -> pd.DataFrame:
    """A model is Pareto-optimal if no other model is at least as good on both FAR and ORR
    and strictly better on one (lower is better)."""
    p = points.copy()
    flags = []
    for _, r in p.iterrows():
        dominated = any((o["far_comply"] <= r["far_comply"]) and (o["orr"] <= r["orr"])
                        and ((o["far_comply"] < r["far_comply"]) or (o["orr"] < r["orr"]))
                        for _, o in p.iterrows())
        flags.append(not dominated)
    p["pareto_optimal"] = flags
    p["distance_to_origin"] = [math.hypot(f, o) for f, o in zip(p["far_comply"], p["orr"],
                                                                strict=True)]
    return p


COLS = ["evaluator_id", "rule", "model_id", "far_comply", "far_comply_lo", "far_comply_hi",
        "orr", "orr_lo", "orr_hi", "pareto_optimal", "distance_to_origin"]


def run(data: AnalysisData, cfg: StudyConfig, out: Path) -> list[Path]:
    a = cfg.experiment.analysis
    out.mkdir(parents=True, exist_ok=True)
    written = []
    evaluators = sorted(set(data.judg["evaluator_id"])) if not data.judg.empty else []
    for ev in evaluators:
        m = model_metrics(data, cfg, ev, a.primary_rule).dropna(subset=["far_comply", "orr"])
        pts = pareto(m)[COLS].sort_values("distance_to_origin")
        tag = "primary" if ev == a.primary_evaluator else ev
        pts.to_csv(p := out / f"rq4_plane_{tag}.csv", index=False)
        written.append(p)
        if ev == a.primary_evaluator and not pts.empty:
            written += plots.far_orr_plane(pts, f"Safety-usability plane ({ev}, "
                                           f"{a.primary_rule})", out / "rq4_plane")
    return written
