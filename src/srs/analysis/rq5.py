"""RQ5: how reliable are the evaluators? Agreement among evaluators (pairwise kappa,
Krippendorff's alpha), accuracy against the external human-labeled sets and against the
weighted self-annotation reference, signed FAR/ORR bias per model, Kendall's tau between
model rankings, self-preference of the LLM judges, judge status rates, and each model's
FAR/ORR as every evaluator measures it, the thinking-variant effect on FAR/ORR per family
under every evaluator (tables plus heatmaps), and each model's FAR/ORR rank under every
evaluator compared with the primary evaluator's ranking (table plus bump charts). Study
responses are limited to `analysis.primary_splits`."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pandas as pd

from srs.analysis import agreement as A
from srs.analysis import plots
from srs.analysis.data import AnalysisData
from srs.analysis.metrics import decisions, primary_rows
from srs.analysis.rq1 import model_metrics
from srs.analysis.stats import bootstrap_ci
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


def in_splits(data: AnalysisData, splits: list[str]) -> AnalysisData:
    """Study responses (sample 0, every framing) in the given splits, and only their
    judgments; external reference data is left as is."""
    resp = data.resp[(data.resp["sample_idx"] == 0) & data.resp["split"].isin(splits)]
    judg = data.judg[data.judg["item_id"].isin(resp["response_id"])] \
        if not data.judg.empty else data.judg
    return replace(data, resp=resp, judg=judg)


# Figure column order: simplest evaluator first, LLM judges last.
EVALUATOR_ORDER = ["keyword", "keyword_ext", "classic", "wildguard", "judge_a", "judge_b"]


def rates_by_evaluator(data: AnalysisData, cfg: StudyConfig, evaluators: list[str],
                       rule: str) -> pd.DataFrame:
    """Per model FAR and ORR (with Wilson CIs) as each evaluator measures them, on the same
    primary rows as RQ1."""
    cols = ["evaluator_id", "rule", "model_id", "n_refuse", "far_comply", "far_comply_lo",
            "far_comply_hi", "n_comply", "orr", "orr_lo", "orr_hi"]
    frames = [model_metrics(data, cfg, ev, rule)[cols] for ev in evaluators]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


def rate_figures(rates: pd.DataFrame, rule: str, out: Path) -> list[Path]:
    """Model x evaluator heatmaps of FAR and of ORR: a column that differs from the others
    shows where the evaluator choice changes a model's estimate."""
    if rates.empty:
        return []
    order = _evaluator_order(set(rates["evaluator_id"]))
    written = []
    for metric, title in (("far_comply", "FAR (refuse-expected prompts not refused)"),
                          ("orr", "ORR (comply-expected prompts refused)")):
        table = rates.pivot(index="model_id", columns="evaluator_id", values=metric)[order]
        written += plots.heatmap(table, f"{title} by evaluator, {rule}",
                                 out / f"rq5_{metric}_by_evaluator",
                                 cbar_label=metric.split("_")[0].upper())
    return written


def thinking_pairs(cfg: StudyConfig, model_ids: set[str]) -> dict[str, tuple[str, str]]:
    """family -> (standard model, thinking model), for families with exactly one of each
    among the given models."""
    by_family: dict[str, dict[bool, list[str]]] = {}
    for m in sorted(model_ids):
        if m in cfg.models:
            mc = cfg.models[m]
            by_family.setdefault(mc.family, {}).setdefault(bool(mc.is_reasoning), []).append(m)
    return {f: (v[False][0], v[True][0]) for f, v in sorted(by_family.items())
            if len(v.get(False, [])) == 1 and len(v.get(True, [])) == 1}


def thinking_effect(data: AnalysisData, cfg: StudyConfig, evaluators: list[str],
                    rule: str) -> pd.DataFrame:
    """Per family and evaluator: FAR and ORR of the thinking variant minus the standard one,
    on the prompts both answered validly, with a bootstrap CI that resamples prompts (the
    two variants answer the same prompts, so the difference is paired)."""
    a = cfg.experiment.analysis
    pairs = thinking_pairs(cfg, set(data.resp["model_id"]))
    rows = []
    for ev in evaluators:
        dec = decisions(primary_rows(data.resp, a.primary_splits), data.judg, ev, rule)  # type: ignore[arg-type]
        if not a.include_borderline:
            dec = dec[dec["tier"] == "clear"]
        dec = dec[dec["valid"]]
        for family, (std, thk) in pairs.items():
            for metric, expected in (("far_comply", "refuse"), ("orr", "comply")):
                sub = dec[dec["expected_behavior"] == expected]
                wide = sub.pivot_table(index="prompt_id", columns="model_id", values="refused",
                                       aggfunc="first")
                if not {std, thk} <= set(wide.columns):
                    continue
                wide = wide[[std, thk]].dropna().astype(float)
                if metric == "far_comply":  # FAR counts the *not* refused
                    wide = 1 - wide
                s, t = wide[std].to_numpy(), wide[thk].to_numpy()
                if not len(s):
                    continue

                def stat(rng, s=s, t=t):
                    i = rng.integers(0, len(s), len(s))
                    return float(t[i].mean() - s[i].mean())

                lo, hi = bootstrap_ci(stat, a.bootstrap_resamples, cfg.experiment.seed,
                                      a.ci_level)
                rows.append({"evaluator_id": ev, "rule": rule, "family": family,
                             "standard_model": std, "thinking_model": thk, "metric": metric,
                             "n_prompts": len(s), "standard": s.mean(), "thinking": t.mean(),
                             "delta": t.mean() - s.mean(), "delta_lo": lo, "delta_hi": hi})
    return pd.DataFrame(rows)


def _evaluator_order(present: set[str]) -> list[str]:
    return [e for e in EVALUATOR_ORDER if e in present] + sorted(present - set(EVALUATOR_ORDER))


def thinking_figures(effect: pd.DataFrame, rule: str, ci_level: float, out: Path
                     ) -> list[Path]:
    """Family x evaluator heatmaps of the thinking effect on FAR and on ORR, on one shared
    symmetric scale; '*' marks a CI that excludes zero."""
    if effect.empty:
        return []
    order = _evaluator_order(set(effect["evaluator_id"]))
    lim = max(0.25, math.ceil(effect["delta"].abs().max() * 10) / 10)
    note = f"thinking minus standard; red = higher with thinking; * {ci_level:.0%} CI excludes 0"
    written = []
    for metric, title in (("far_comply", "Thinking effect on FAR"),
                          ("orr", "Thinking effect on ORR")):
        e = effect[effect["metric"] == metric]
        if e.empty:
            continue
        table = e.pivot(index="family", columns="evaluator_id", values="delta")[order]
        sig = (e.assign(sig=(e["delta_lo"] > 0) | (e["delta_hi"] < 0))
               .pivot(index="family", columns="evaluator_id", values="sig")[order])
        written += plots.heatmap(table, f"{title} by evaluator, {rule}",
                                 out / f"rq5_thinking_delta_{metric}",
                                 cbar_label=f"Δ {metric.split('_')[0].upper()}",
                                 vmin=-lim, vmax=lim, diverging=True, marks=sig, note=note)
    return written


def model_ranks(rates: pd.DataFrame, primary: str) -> pd.DataFrame:
    """Each model's rank on FAR and on ORR under every evaluator (1 = lowest rate; ties get
    the average rank), with Kendall's tau-b between that evaluator's ranking and the primary
    evaluator's."""
    from scipy.stats import kendalltau

    rows = []
    for metric in ("far_comply", "orr"):
        wide = rates.pivot(index="model_id", columns="evaluator_id", values=metric)
        ranks = wide.rank(method="average")
        for ev in wide.columns:
            tau = p = float("nan")
            if primary in wide.columns and wide[[ev, primary]].notna().all(axis=None):
                res = kendalltau(wide[ev], wide[primary])
                tau, p = float(res.statistic), float(res.pvalue)
            for model in wide.index:
                rows.append({"evaluator_id": ev, "metric": metric, "model_id": model,
                             "rate": wide.at[model, ev], "rank": ranks.at[model, ev],
                             "primary_evaluator": primary, "tau_vs_primary": tau,
                             "tau_p": p})
    return pd.DataFrame(rows)


def rank_figures(ranks: pd.DataFrame, cfg: StudyConfig, rule: str, out: Path) -> list[Path]:
    """Bump charts of model rank across evaluators, one per metric: crossing lines are models
    that change places when the evaluator changes."""
    if ranks.empty:
        return []
    families = list(dict.fromkeys(cfg.models[m].family for m in cfg.evaluated))
    family_colors = dict(zip(families, plots.CATEGORICAL, strict=False))
    models = sorted(set(ranks["model_id"]))
    family = {m: cfg.models[m].family for m in models}
    thinking = {m: bool(cfg.models[m].is_reasoning) for m in models}
    primary = ranks["primary_evaluator"].iloc[0]
    written = []
    for metric, title in (("far_comply", "Model rank on FAR by evaluator"),
                          ("orr", "Model rank on ORR by evaluator")):
        r = ranks[ranks["metric"] == metric]
        order = _evaluator_order(set(r["evaluator_id"]))
        wide = r.pivot(index="model_id", columns="evaluator_id", values="rank")[order]
        tau = r.groupby("evaluator_id")["tau_vs_primary"].first()
        xlabels = [ev.replace("_", " ") + ("\n(primary)" if ev == primary else
                                           f"\nτ = {tau[ev]:+.2f}" if pd.notna(tau[ev]) else "")
                   for ev in order]
        note = (f"{rule}; rank 1 = lowest rate, ties share the average rank; "
                f"τ = Kendall's tau-b vs. {primary}")
        written += plots.bump_chart(wide, family, family_colors, thinking, xlabels,
                                    title, note, out / f"rq5_rank_{metric}")
    return written


def run(data: AnalysisData, cfg: StudyConfig, out: Path) -> list[Path]:
    a = cfg.experiment.analysis
    data = in_splits(data, a.primary_splits)
    rule = a.primary_rule
    seed, n_boot = cfg.experiment.seed, a.bootstrap_resamples
    out.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}
    evaluators = sorted(set(data.judg["evaluator_id"])) if not data.judg.empty else []

    # Agreement uses the same prompts as the rates: clear-tier only unless borderline prompts
    # are included.
    tiers = "clear+borderline" if a.include_borderline else "clear"
    judg = data.judg
    if not a.include_borderline and not judg.empty:
        clear_ids = set(data.resp.loc[data.resp["tier"] == "clear", "response_id"])
        judg = judg[judg["item_id"].isin(clear_ids)]
    mat = A.refused_matrix(judg, evaluators, rule) if evaluators else pd.DataFrame()
    kappa = A.pairwise_kappa(mat) if evaluators else pd.DataFrame()
    if not kappa.empty:
        kappa.insert(0, "tiers", tiers)
    tables["kappa_pairwise"] = kappa
    tables["alpha"] = pd.DataFrame([{
        "tiers": tiers, "rule": rule, "evaluators": ",".join(evaluators), "n_items": len(mat),
        "krippendorff_alpha": A.krippendorff_alpha(mat) if evaluators else float("nan")}])
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
    tables["rates_by_evaluator"] = rates_by_evaluator(data, cfg, evaluators, rule)
    tables["thinking_effect"] = thinking_effect(data, cfg, evaluators, rule)
    tables["model_ranks"] = model_ranks(tables["rates_by_evaluator"], a.primary_evaluator) \
        if not tables["rates_by_evaluator"].empty else pd.DataFrame()

    written = []
    for name, df in tables.items():
        df.to_csv(p := out / f"rq5_{name}.csv", index=False)
        written.append(p)
    return (written + rate_figures(tables["rates_by_evaluator"], rule, out)
            + thinking_figures(tables["thinking_effect"], rule, a.ci_level, out)
            + rank_figures(tables["model_ranks"], cfg, rule, out))
