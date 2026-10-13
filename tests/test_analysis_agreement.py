import numpy as np
import pandas as pd
import pytest

from srs.analysis import agreement as A
from srs.analysis.metrics import judgments_frame
from srs.schemas import Judgment


def _judg(labels: dict[str, list[str]], statuses=None):
    js = []
    for ev, labs in labels.items():
        for i, lab in enumerate(labs):
            st = (statuses or {}).get((ev, i), "ok")
            js.append(Judgment(item_id=f"r{i}", evaluator_id=ev, evaluator_version="v",
                               refusal=lab if st == "ok" else None, status=st))
    return judgments_frame(js)


R, C, P = "full_refusal", "full_compliance", "partial"


def test_kappa_alpha_known():
    labs = [R, R, C, C, R, C, R, C]
    inv = [C if x == R else R for x in labs]
    j = _judg({"a": labs, "b": labs, "c": inv})
    mat = A.refused_matrix(j, ["a", "b", "c"], "partial_as_refusal")
    k = A.pairwise_kappa(mat).set_index(["evaluator_a", "evaluator_b"])["kappa"]
    assert k[("a", "b")] == pytest.approx(1.0) and k[("a", "c")] == pytest.approx(-1.0)
    assert A.krippendorff_alpha(mat[["a", "b"]]) == pytest.approx(1.0)
    assert A.krippendorff_alpha(mat) < 0


def test_rule_changes_partial_and_invalid_excluded():
    j = _judg({"a": [P, R, C], "b": [R, R, C]}, statuses={("b", 2): "parse_error"})
    m1 = A.refused_matrix(j, ["a", "b"], "partial_as_refusal")
    m2 = A.refused_matrix(j, ["a", "b"], "partial_as_compliance")
    assert m1.loc["r0", "a"] == 1 and m2.loc["r0", "a"] == 0
    assert np.isnan(m1.loc["r2", "b"]) and A.pairwise_kappa(m1).iloc[0]["n"] == 2


def test_weighted_prf_hand_computed():
    yt = [True, True, False, False]
    yp = [True, False, True, False]
    plain = A.weighted_prf(yt, yp)
    assert (plain["precision"], plain["recall"], plain["accuracy"]) == (0.5, 0.5, 0.5)
    w = A.weighted_prf(yt, yp, [3, 1, 1, 1])  # tp=3, fn=1, fp=1, tn=1
    assert w["precision"] == pytest.approx(0.75) and w["recall"] == pytest.approx(0.75)
    assert w["accuracy"] == pytest.approx(4 / 6)


def _sample(n_models=4, n=40, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_models):
        for i in range(n):
            exp = "refuse" if i % 2 else "comply"
            p_ref = (0.9 - 0.15 * m) if exp == "refuse" else 0.1 + 0.05 * m
            ref = rng.random() < p_ref
            ev = (not ref) if rng.random() < noise else ref
            rows.append({"model_id": f"m{m}", "family": "gemini" if m < 2 else "glm",
                         "expected_behavior": exp, "ref_refused": float(ref),
                         "eval_refused": float(ev), "weight": 1.0 + (i % 3)})
    return pd.DataFrame(rows)


def test_signed_bias_known():
    df = _sample()
    df["eval_refused"] = 1.0  # always refuses -> FAR 0, ORR 1
    b = A.signed_bias(df).set_index("model_id")
    ref = df[df.model_id == "m0"]
    r = ref[ref.expected_behavior == "refuse"]
    far_ref = (r.weight * (1 - r.ref_refused)).sum() / r.weight.sum()
    assert b.loc["m0", "far_bias"] == pytest.approx(0 - far_ref)
    assert b.loc["m0", "orr_eval"] == 1.0


def test_ranking_tau_perfect_and_ci():
    res = A.ranking_tau(_sample(n_models=5, n=200, seed=1), "far", n_boot=200, seed=2)
    assert res["tau"] == pytest.approx(1.0) and res["tau_lo"] <= res["tau"] <= res["tau_hi"]
    again = A.ranking_tau(_sample(n_models=5, n=200, seed=1), "far", n_boot=200, seed=2)
    assert again == res


def test_self_preference_split():
    df = _sample(n_models=4, n=100, seed=3)
    other = df["family"] != "gemini"
    df.loc[other, "eval_refused"] = 1 - df.loc[other, "ref_refused"]  # wrong on other family
    sp = A.self_preference(df, "gemini").set_index("subset")
    assert sp.loc["own_family", "accuracy"] == 1.0 and sp.loc["other_family", "accuracy"] == 0.0
    assert sp.loc["own_family", "n"] == 200


def test_judge_status_rates():
    j = _judg({"judge_b": [R, R, C, C]}, statuses={("judge_b", 0): "blocked",
                                                    ("judge_b", 1): "judge_refused"})
    j["target"] = "responses"
    rates = A.judge_status_rates(j).iloc[0]
    assert rates["rate_blocked"] == 0.25 and rates["rate_judge_refused"] == 0.25
    assert rates["rate_not_ok"] == 0.5 and rates["n"] == 4
