"""RQ5: how reliable are the evaluators? Agreement among evaluators (pairwise kappa,
Krippendorff's alpha), accuracy against the external human-labeled sets and against the
weighted self-annotation reference, signed FAR/ORR bias per model, Kendall's tau between
model rankings, self-preference of the LLM judges, and judge status rates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from srs.analysis import agreement as A
from srs.analysis.data import AnalysisData
from srs.config import LLMJudgeEvaluatorConfig, StudyConfig
from srs.evaluators.base import is_refusal


def _with_eval(df: pd.DataFrame, judg: pd.DataFrame, ev: str, key: str, rule: str
               ) -> pd.DataFrame:
    j = judg[(judg["evaluator_id"] == ev) & (judg["status"] == "ok")][
        ["item_id", "refusal", "harmful"]].rename(
        columns={"item_id": key, "refusal": "eval_refusal", "harmful": "eval_harmful"})
    m = df.merge(j, on=key)
    m["eval_refused"] = [is_refusal(x, rule) for x in m["eval_refusal"]]
    return m


def external_accuracy(data: AnalysisData, rule: str) -> pd.DataFrame:
    if data.judg_ext.empty or data.refs.empty:
        return pd.DataFrame()
    refs = data.refs.rename(columns={"refusal": "ref_refusal", "harmful": "ref_harmful"})
    rows = []
    for ev in sorted(set(data.judg_ext["evaluator_id"])):
        m = _with_eval(refs, data.judg_ext, ev, "item_id", rule)
        for src, g in m.groupby("source"):
            r = g[g["ref_refusal"].notna() & g["eval_refused"].notna()]
            if len(r):
                rows.append({"evaluator_id": ev, "source": src, "label": "refusal", "rule": rule,
                             **A.weighted_prf([is_refusal(x, rule) for x in r["ref_refusal"]],
                                              r["eval_refused"].astype(bool))})
            h = g[g["ref_harmful"].notna() & g["eval_harmful"].notna()]
            if len(h):
                rows.append({"evaluator_id": ev, "source": src, "label": "harmful", "rule": rule,
                             **A.weighted_prf(h["ref_harmful"].astype(bool),
                                              h["eval_harmful"].astype(bool))})
    return pd.DataFrame(rows)


def self_reference(data: AnalysisData, rule: str) -> pd.DataFrame:
    """Final self-annotation labels with weights and response metadata."""
    if data.human.empty:
        return pd.DataFrame()
    h = data.human.rename(columns={"item_id": "response_id", "refusal": "ref_refusal",
                                   "harmful": "ref_harmful"})
    h = h.merge(data.resp[["response_id", "model_id", "family", "expected_behavior"]],
                on="response_id")
    h["ref_refused"] = [float(is_refusal(x, rule)) for x in h["ref_refusal"]]
    return h


def run(data: AnalysisData, cfg: StudyConfig, out: Path) -> list[Path]:
    a = cfg.experiment.analysis
    rule = a.primary_rule
    seed, n_boot = cfg.experiment.seed, a.bootstrap_resamples
    out.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}
    evaluators = sorted(set(data.judg["evaluator_id"])) if not data.judg.empty else []

    mat = A.refused_matrix(data.judg, evaluators, rule) if evaluators else pd.DataFrame()
    tables["kappa_pairwise"] = A.pairwise_kappa(mat) if evaluators else pd.DataFrame()
    tables["alpha"] = pd.DataFrame([{"rule": rule, "evaluators": ",".join(evaluators),
                                     "n_items": len(mat),
                                     "krippendorff_alpha": A.krippendorff_alpha(mat)
                                     if evaluators else float("nan")}])
    tables["accuracy_external"] = external_accuracy(data, rule)

    ref = self_reference(data, rule)
    acc, bias, taus, selfpref = [], [], [], []
    for ev in evaluators if not ref.empty else []:
        m = _with_eval(ref, data.judg, ev, "response_id", rule)
        m = m[m["eval_refused"].notna()].copy()
        m["eval_refused"] = m["eval_refused"].astype(float)
        if m.empty:
            continue
        acc.append({"evaluator_id": ev, "label": "refusal", "rule": rule, "weighted": True,
                    **A.weighted_prf(m["ref_refused"].astype(bool), m["eval_refused"].astype(bool),
                                     m["weight"])})
        hm = m[m["ref_harmful"].notna() & m["eval_harmful"].notna()]
        if len(hm):
            acc.append({"evaluator_id": ev, "label": "harmful", "rule": rule, "weighted": True,
                        **A.weighted_prf(hm["ref_harmful"].astype(bool),
                                         hm["eval_harmful"].astype(bool), hm["weight"])})
        b = A.signed_bias(m)
        b.insert(0, "evaluator_id", ev)
        bias.append(b)
        for metric in ("far", "orr"):
            taus.append({"evaluator_id": ev, **A.ranking_tau(m, metric, n_boot, seed,
                                                              a.ci_level)})
        ecfg = cfg.evaluators.get(ev)
        if isinstance(ecfg, LLMJudgeEvaluatorConfig):
            sp = A.self_preference(m, cfg.models[ecfg.model].family)
            sp.insert(0, "evaluator_id", ev)
            selfpref.append(sp)
    tables["accuracy_self"] = pd.DataFrame(acc)
    tables["bias"] = pd.concat(bias) if bias else pd.DataFrame()
    tables["ranking_tau"] = pd.DataFrame(taus)
    tables["self_preference"] = pd.concat(selfpref) if selfpref else pd.DataFrame()
    both = pd.concat([d for d in (data.judg, data.judg_ext) if not d.empty]) \
        if not (data.judg.empty and data.judg_ext.empty) else pd.DataFrame()
    tables["judge_status"] = A.judge_status_rates(both)

    written = []
    for name, df in tables.items():
        df.to_csv(p := out / f"rq5_{name}.csv", index=False)
        written.append(p)
    return written
