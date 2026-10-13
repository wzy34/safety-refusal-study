"""Evaluator agreement and accuracy (RQ5): pairwise Cohen's kappa, Krippendorff's alpha,
(weighted) precision/recall/F1/accuracy against a reference, signed bias of FAR/ORR
estimates, Kendall's tau between model rankings, self-preference, and judge status rates."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd

from srs.analysis.stats import bootstrap_ci
from srs.config import PartialRule
from srs.evaluators.base import is_refusal


def refused_matrix(judg: pd.DataFrame, evaluator_ids: Sequence[str], rule: PartialRule
                   ) -> pd.DataFrame:
    """items x evaluators matrix of refused (1/0), NaN where no valid label."""
    ok = judg[(judg["status"] == "ok") & judg["refusal"].notna()
              & judg["evaluator_id"].isin(evaluator_ids)].copy()
    ok["refused"] = [float(is_refusal(x, rule)) for x in ok["refusal"]]
    mat = ok.pivot_table(index="item_id", columns="evaluator_id", values="refused",
                         aggfunc="first")
    return mat.reindex(columns=list(evaluator_ids))


def pairwise_kappa(mat: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import cohen_kappa_score

    rows = []
    for a, b in itertools.combinations(mat.columns, 2):
        pair = mat[[a, b]].dropna()
        k = cohen_kappa_score(pair[a], pair[b]) if len(pair) else float("nan")
        rows.append({"evaluator_a": a, "evaluator_b": b, "n": len(pair),
                     "kappa": float(k) if not pd.isna(k) else float("nan"),
                     "percent_agreement": float((pair[a] == pair[b]).mean()) if len(pair)
                     else float("nan")})
    return pd.DataFrame(rows)


def krippendorff_alpha(mat: pd.DataFrame) -> float:
    import krippendorff

    data = mat.T.to_numpy(float)  # coders x units
    if data.shape[1] == 0 or np.all(np.isnan(data)):
        return float("nan")
    return float(krippendorff.alpha(reliability_data=data, level_of_measurement="nominal"))


def weighted_prf(y_true: Sequence[bool], y_pred: Sequence[bool],
                 weights: Sequence[float] | None = None) -> dict[str, float]:
    """Precision/recall/F1 for the positive class, plus accuracy; optionally weighted (e.g. by
    inverse inclusion probability)."""
    yt = np.asarray(y_true, bool)
    yp = np.asarray(y_pred, bool)
    w = np.ones(len(yt)) if weights is None else np.asarray(weights, float)
    tp = float(w[yt & yp].sum())
    fp = float(w[~yt & yp].sum())
    fn = float(w[yt & ~yp].sum())
    acc = float(w[yt == yp].sum() / w.sum()) if len(w) else float("nan")
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p == p and r == r and p + r else float("nan")
    return {"n": len(yt), "precision": p, "recall": r, "f1": f, "accuracy": acc}


def _rates(d: pd.DataFrame, col: str) -> tuple[float, float]:
    """Weighted FAR (refuse-expected, not refused) and ORR (comply-expected, refused)."""
    ref = d[d["expected_behavior"] == "refuse"]
    com = d[d["expected_behavior"] == "comply"]
    far = float((ref["weight"] * (1 - ref[col])).sum() / ref["weight"].sum()) if len(ref) \
        else float("nan")
    orr = float((com["weight"] * com[col]).sum() / com["weight"].sum()) if len(com) \
        else float("nan")
    return far, orr


def signed_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Per model: evaluator FAR/ORR minus reference FAR/ORR on the same (weighted) items.
    `df` columns: model_id, expected_behavior, ref_refused, eval_refused, weight."""
    rows = []
    for model, d in df.groupby("model_id", sort=True):
        far_e, orr_e = _rates(d, "eval_refused")
        far_r, orr_r = _rates(d, "ref_refused")
        rows.append({"model_id": model, "n": len(d), "far_eval": far_e, "far_ref": far_r,
                     "far_bias": far_e - far_r, "orr_eval": orr_e, "orr_ref": orr_r,
                     "orr_bias": orr_e - orr_r})
    return pd.DataFrame(rows)


def ranking_tau(df: pd.DataFrame, metric: str, n_boot: int, seed: int,
                level: float = 0.95) -> dict[str, float]:
    """Kendall's tau between the model ranking by `metric` (far or orr) under the evaluator
    and under the reference, with a bootstrap CI resampling items within each model."""
    from scipy.stats import kendalltau

    expected = "refuse" if metric == "far" else "comply"
    groups = []
    for _, g in df.groupby("model_id", sort=True):
        g = g[g["expected_behavior"] == expected]
        w = g["weight"].to_numpy(float)
        ev, rf = g["eval_refused"].to_numpy(float), g["ref_refused"].to_numpy(float)
        if metric == "far":  # FAR counts the *not* refused
            ev, rf = 1 - ev, 1 - rf
        groups.append((w, ev, rf))

    def tau_from(idx_per_group: list[np.ndarray] | None) -> float:
        ev_r, rf_r = [], []
        for k, (w, ev, rf) in enumerate(groups):
            if len(w) == 0:
                ev_r.append(np.nan)
                rf_r.append(np.nan)
                continue
            i = idx_per_group[k] if idx_per_group is not None else slice(None)
            ws = w[i].sum()
            ev_r.append((w[i] * ev[i]).sum() / ws)
            rf_r.append((w[i] * rf[i]).sum() / ws)
        e, r = np.array(ev_r), np.array(rf_r)
        ok = ~(np.isnan(e) | np.isnan(r))
        if ok.sum() < 3:
            return float("nan")
        return float(kendalltau(e[ok], r[ok]).statistic)

    def stat(rng: np.random.Generator) -> float:
        return tau_from([rng.integers(0, len(w), len(w)) for w, _, _ in groups])

    lo, hi = bootstrap_ci(stat, n_boot, seed, level)
    return {"metric": metric, "tau": tau_from(None), "tau_lo": lo, "tau_hi": hi,
            "n_models": len(groups)}


def self_preference(df: pd.DataFrame, judge_family: str) -> pd.DataFrame:
    """Agreement of a judge with the reference on own-family vs other-family responses.
    `df` columns: family, ref_refused, eval_refused, weight."""
    from sklearn.metrics import cohen_kappa_score

    rows = []
    for subset, d in (("own_family", df[df["family"] == judge_family]),
                      ("other_family", df[df["family"] != judge_family])):
        prf = weighted_prf(d["ref_refused"].astype(bool), d["eval_refused"].astype(bool),
                           d["weight"]) if len(d) else {"n": 0}
        k = cohen_kappa_score(d["ref_refused"], d["eval_refused"]) \
            if len(d) and d["ref_refused"].nunique() + d["eval_refused"].nunique() > 2 \
            else float("nan")
        rows.append({"judge_family": judge_family, "subset": subset, "kappa": float(k), **prf})
    return pd.DataFrame(rows)


def judge_status_rates(judg: pd.DataFrame) -> pd.DataFrame:
    """Share of each judgment status per evaluator and target."""
    if judg.empty:
        return pd.DataFrame()
    counts = judg.groupby(["evaluator_id", "target", "status"]).size().unstack(fill_value=0)
    for s in ("ok", "judge_refused", "blocked", "parse_error"):
        if s not in counts:
            counts[s] = 0
    total = counts.sum(axis=1)
    out = counts.div(total, axis=0).add_prefix("rate_")
    out["n"] = total
    out["rate_not_ok"] = 1 - out["rate_ok"]
    return out.reset_index()
