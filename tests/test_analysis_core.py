import numpy as np
import pandas as pd
import pytest

from srs.analysis import stats
from srs.analysis.metrics import compute_metrics, decisions, judgments_frame
from srs.schemas import Judgment


def _resp_df(rows):
    return pd.DataFrame([{"response_id": f"r{i}", "prompt_id": f"p{i}", "intent_id": f"i{i}",
                          "model_id": m, "family": "gemini", "category": "privacy",
                          "expected_behavior": exp, "tier": tier, "framing": "direct",
                          "split": "main", "sample_idx": 0, "status": status}
                         for i, (m, exp, tier, status) in enumerate(rows)])


def _judg(labels, evaluator="judge_a"):
    return judgments_frame([Judgment(item_id=f"r{i}", evaluator_id=evaluator,
                                     evaluator_version="v1", refusal=lab, harmful=harm,
                                     status="ok") for i, (lab, harm) in labels.items()])


def test_hand_computed_case():
    # 10 refuse-expected (3 not refused), 10 comply-expected (2 refused)
    rows = [("m", "refuse", "clear", "ok")] * 10 + [("m", "comply", "clear", "ok")] * 10
    labels = {i: ("full_compliance" if i < 3 else "full_refusal", i < 1) for i in range(10)}
    labels.update({i: ("full_refusal" if i < 12 else "full_compliance", False)
                   for i in range(10, 20)})
    dec = decisions(_resp_df(rows), _judg(labels), "judge_a", "partial_as_refusal")
    m = compute_metrics(dec, n_boot=500, seed=1).iloc[0]
    assert m["far_comply"] == pytest.approx(0.3) and m["orr"] == pytest.approx(0.2)
    assert m["j"] == pytest.approx(0.5) and m["far_harm"] == pytest.approx(0.1)
    assert m["n_refuse"] == 10 and m["n_comply"] == 10
    for k in ("far_comply", "orr", "j", "far_harm"):
        assert m[f"{k}_lo"] < m[k] < m[f"{k}_hi"]
    assert m["rule"] == "partial_as_refusal" and m["evaluator_id"] == "judge_a"


def test_partial_rule_and_blocked_and_tier():
    rows = [("m", "refuse", "clear", "ok"), ("m", "refuse", "clear", "ok"),
            ("m", "refuse", "clear", "api_blocked"), ("m", "comply", "clear", "ok"),
            ("m", "comply", "borderline", "ok")]
    labels = {0: ("partial", False), 1: ("full_refusal", False), 3: ("partial", False),
              4: ("full_refusal", False)}
    resp = _resp_df(rows)
    as_ref = compute_metrics(decisions(resp, _judg(labels), "judge_a", "partial_as_refusal"),
                             n_boot=100).iloc[0]
    as_com = compute_metrics(decisions(resp, _judg(labels), "judge_a", "partial_as_compliance"),
                             n_boot=100).iloc[0]
    assert as_ref["far_comply"] == 0.0 and as_com["far_comply"] == 0.5
    assert as_ref["orr"] == 1.0 and as_com["orr"] == 0.0
    assert as_ref["n_refuse"] == 2  # blocked response not in the denominator
    assert as_ref["n_blocked"] == 1 and as_ref["block_rate"] == pytest.approx(1 / 4)
    assert as_ref["n_comply"] == 1  # borderline excluded by default
    both = compute_metrics(decisions(resp, _judg(labels), "judge_a", "partial_as_refusal"),
                           include_borderline=True, n_boot=100).iloc[0]
    assert both["n_comply"] == 2 and both["tiers"] == "clear+borderline"


def test_latest_evaluator_version_used():
    js = [Judgment(item_id="r0", evaluator_id="e", evaluator_version="old", refusal="partial",
                   status="ok", created_at="2026-01-01T00:00:00+00:00"),
          Judgment(item_id="r0", evaluator_id="e", evaluator_version="new",
                   refusal="full_refusal", status="ok", created_at="2026-02-01T00:00:00+00:00")]
    assert judgments_frame(js)["evaluator_version"].tolist() == ["new"]


# --- 8.2 stats ---------------------------------------------------------------------------------

def test_wilson_reference():
    from statsmodels.stats.proportion import proportion_confint
    assert stats.wilson(3, 10) == pytest.approx(proportion_confint(3, 10, method="wilson"))
    assert stats.wilson(3, 10) == pytest.approx((0.1078, 0.6032), abs=1e-4)


def test_mcnemar_exact_known():
    # discordant pairs 1 vs 9 -> exact two-sided binomial p = 2 * P(X <= 1 | n=10, .5)
    a = [True] * 9 + [False] * 1 + [True] * 5
    b = [False] * 9 + [True] * 1 + [True] * 5
    res = stats.mcnemar_exact(a, b)
    assert (res["a_only"], res["b_only"]) == (9, 1)
    assert res["p_value"] == pytest.approx(2 * 11 / 1024)
    assert stats.mcnemar_exact([True, False], [True, False])["p_value"] == 1.0


def test_holm_known():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_bootstrap_seeded():
    data = np.arange(100.0)
    f = lambda rng: rng.choice(data, 100).mean()  # noqa: E731
    a = stats.bootstrap_ci(f, 500, seed=3)
    assert a == stats.bootstrap_ci(f, 500, seed=3) and a[0] < 49.5 < a[1]


def test_kendalls_w_known():
    # perfect agreement -> W = 1; Siegel & Castellan example without ties
    perfect = pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    assert stats.kendalls_w(perfect)["W"] == pytest.approx(1.0)
    opposite = pd.DataFrame([[1, 2, 3, 4], [4, 3, 2, 1]])
    assert stats.kendalls_w(opposite)["W"] == pytest.approx(0.0)
    # hand computed: ranks sums R = [4, 5, 9] for m=3, n=3 -> S = 14, W = 12*14/(9*24) = 0.7778
    w = stats.kendalls_w(pd.DataFrame([[1, 2, 3], [1, 2, 3], [2, 1, 3]]))
    assert w["W"] == pytest.approx(12 * 14 / (9 * 24))


def test_spearman_matches_scipy():
    from scipy.stats import spearmanr
    x, y = [1, 2, 3, 4, 5], [2, 1, 4, 3, 5]
    assert stats.spearman(x, y)["rho"] == pytest.approx(spearmanr(x, y).statistic)


def test_clustered_logit_matches_statsmodels():
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"x": rng.choice(["a", "b"], n), "intent": rng.integers(0, 50, n)})
    eff = rng.normal(0, 1, 50)
    df["y"] = (rng.random(n) < 1 / (1 + np.exp(-(0.5 * (df["x"] == "b") + eff[df["intent"]])))
               ).astype(int)
    tab = stats.clustered_logit(df, "y ~ C(x)", "intent")
    ref = smf.glm("y ~ C(x)", df, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": df["intent"]})
    assert tab["coef"].tolist() == pytest.approx(ref.params.tolist())
    assert tab["se_cluster"].tolist() == pytest.approx(ref.bse.tolist())
    naive = smf.glm("y ~ C(x)", df, family=sm.families.Binomial()).fit()
    assert not np.allclose(tab["se_cluster"], naive.bse)


def test_clustered_logit_drops_collinear_terms():
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame({"a": rng.choice(["p", "q"], n), "intent": rng.integers(0, 40, n)})
    df["b"] = np.where(df["a"] == "p", "u", "v")  # b is fully determined by a
    df["y"] = (rng.random(n) < 0.4).astype(int)
    tab = stats.clustered_logit(df, "y ~ C(a) + C(b)", "intent")
    assert tab["dropped"].sum() == 1 and tab.loc[~tab["dropped"], "se_cluster"].notna().all()
