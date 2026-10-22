"""Statistical helpers (design D8): Wilson intervals, exact McNemar, Holm correction, seeded
percentile bootstrap, Kendall's W, Spearman correlation and a logistic GLM with standard
errors clustered by intent."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


def wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    from statsmodels.stats.proportion import proportion_confint

    lo, hi = proportion_confint(k, n, alpha=1 - level, method="wilson")
    return float(lo), float(hi)


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> dict[str, float]:
    """Exact McNemar test for paired binary outcomes a[i], b[i]."""
    from statsmodels.stats.contingency_tables import mcnemar

    a_arr, b_arr = np.asarray(a, bool), np.asarray(b, bool)
    n01 = int(np.sum(~a_arr & b_arr))
    n10 = int(np.sum(a_arr & ~b_arr))
    if n01 + n10 == 0:
        p = 1.0
    else:
        table = [[int(np.sum(a_arr & b_arr)), n10], [n01, int(np.sum(~a_arr & ~b_arr))]]
        p = float(mcnemar(table, exact=True).pvalue)
    return {"n_pairs": len(a_arr), "a_only": n10, "b_only": n01, "p_value": p}


def holm(p_values: Sequence[float]) -> list[float]:
    if len(p_values) == 0:
        return []
    from statsmodels.stats.multitest import multipletests

    p = np.asarray(p_values, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out.tolist()


def bootstrap_ci(stat: Callable[[np.random.Generator], float], n_resamples: int, seed: int,
                 level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of `stat(rng)` over seeded resamples; `stat` draws its own resample
    from the generator it is given (so callers control the resampling unit)."""
    rng = np.random.default_rng(seed)
    vals = np.array([stat(rng) for _ in range(n_resamples)], float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def kendalls_w(scores: pd.DataFrame) -> dict[str, float]:
    """Kendall's coefficient of concordance. Rows = raters (e.g. models), columns = objects
    (e.g. categories) scored by each rater; ranks are taken within rows (ties averaged) and
    the tie correction is applied. Also returns the chi-square test p-value."""
    from scipy.stats import chi2, rankdata

    data = scores.dropna(axis=1).to_numpy(float)
    m, n = data.shape
    if m < 2 or n < 2:
        return {"W": float("nan"), "chi2": float("nan"), "p_value": float("nan"), "m": m, "n": n}
    ranks = np.vstack([rankdata(row) for row in data])
    r = ranks.sum(axis=0)
    s = float(np.sum((r - r.mean()) ** 2))
    t = 0.0
    for row in data:
        _, counts = np.unique(row, return_counts=True)
        t += float(np.sum(counts ** 3 - counts))
    denom = m ** 2 * (n ** 3 - n) - m * t
    if denom <= 0:  # every rater tied all objects: concordance is undefined
        return {"W": float("nan"), "chi2": float("nan"), "p_value": float("nan"), "m": m, "n": n}
    w = 12 * s / denom
    stat = m * (n - 1) * w
    return {"W": w, "chi2": stat, "p_value": float(chi2.sf(stat, n - 1)), "m": m, "n": n}


def spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    from scipy.stats import spearmanr

    res = spearmanr(x, y)
    return {"rho": float(res.statistic), "p_value": float(res.pvalue), "n": len(x)}


def clustered_logit(df: pd.DataFrame, formula: str, cluster: str) -> pd.DataFrame:
    """Binomial GLM with cluster-robust standard errors. Returns a coefficient table.

    Design columns that are linear combinations of earlier ones (e.g. an interaction cell
    that does not occur) are dropped via pivoted QR and listed with `dropped=True`."""
    import patsy
    import statsmodels.api as sm
    from scipy.linalg import qr

    y, X = patsy.dmatrices(formula, df, return_type="dataframe")
    _, r, piv = qr(X.to_numpy(float), mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    rank = int((diag > diag.max() * 1e-10).sum()) if len(diag) else 0
    keep = sorted(piv[:rank])
    dropped = [c for i, c in enumerate(X.columns) if i not in set(keep)]
    Xk = X.iloc[:, keep]
    groups = pd.factorize(df.loc[Xk.index, cluster])[0]
    fit = sm.GLM(y.iloc[:, 0], Xk, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    ci = fit.conf_int()
    out = pd.DataFrame({"term": fit.params.index, "coef": fit.params.values,
                        "se_cluster": fit.bse.values, "z": fit.tvalues.values,
                        "p_value": fit.pvalues.values, "ci_lo": ci[0].values,
                        "ci_hi": ci[1].values, "odds_ratio": np.exp(fit.params.values),
                        "dropped": False})
    if dropped:
        out = pd.concat([out, pd.DataFrame({"term": dropped, "dropped": True})],
                        ignore_index=True)
    return out
