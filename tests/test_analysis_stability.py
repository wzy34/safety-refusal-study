import pandas as pd
import pytest

from srs.analysis import stability
from srs.analysis.data import load_analysis_data
from srs.evaluators.base import judgments_path
from srs.generation.runner import responses_path
from srs.io import read_records, write_records
from srs.schemas import Judgment, Message, Response
from tests.synthetic import EVALS, TS, build

# refusals among the 5 sampled responses for four stability prompts, and the greedy decision
PATTERN = {0: ([1, 1, 1, 1, 1], True), 1: ([0, 0, 0, 0, 0], True), 2: ([1, 1, 1, 0, 0], False),
           3: ([0, 1, 0, 0, 0], False)}


def _add_stability(cfg, prompts):
    chosen = [p for p in prompts if p.split == "main" and p.tier == "clear"][:4]
    responses = read_records(responses_path(cfg), Response)
    judgments = read_records(judgments_path(cfg, "responses"), Judgment)
    greedy = {(r.model_id, r.prompt_id): r.response_id for r in responses}
    for m in cfg.evaluated:
        for i, p in enumerate(chosen):
            samples, g = PATTERN[i]
            rid0 = greedy[(m, p.prompt_id)]
            responses = [r.model_copy(update={"status": "ok", "final_text": "x",
                                              "transcript": [Message(role="user", content="q"),
                                                             Message(role="assistant",
                                                                     content="x")]})
                         if r.response_id == rid0 else r for r in responses]
            judgments = [j for j in judgments if j.item_id != rid0]
            for k, refused in enumerate([int(g), *samples]):
                rid = rid0 if k == 0 else f"{rid0}:s{k}"
                if k:
                    responses.append(Response(
                        response_id=rid, prompt_id=p.prompt_id, model_id=m, gen_params={},
                        sample_idx=k, transcript=[Message(role="user", content="q"),
                                                  Message(role="assistant", content="x")],
                        final_text="x", status="ok", run_id="syn"))
                for ev in EVALS:
                    judgments.append(Judgment(
                        item_id=rid, evaluator_id=ev, evaluator_version="v1",
                        refusal="full_refusal" if refused else "full_compliance",
                        harmful=False, status="ok", created_at=TS))
    write_records(responses_path(cfg), responses)
    write_records(judgments_path(cfg, "responses"), judgments)
    return chosen


@pytest.fixture
def world(cfg):
    cfg.experiment.analysis.bootstrap_resamples = 200
    prompts, _ = build(cfg)
    chosen = _add_stability(cfg, prompts)
    return cfg, load_analysis_data(cfg), chosen


def test_prompt_patterns_and_majority(world):
    cfg, data, chosen = world
    t = stability.prompt_table(data, "judge_a", "partial_as_refusal")
    assert len(t) == 4 * len(cfg.evaluated) and (t["n_valid"] == 5).all()
    one = t[t["model_id"] == cfg.evaluated[0]].set_index("prompt_id")
    got = [one.loc[p.prompt_id, "pattern"] for p in chosen]
    assert got == ["always_refuse", "always_comply", "mixed", "mixed"]
    assert [one.loc[p.prompt_id, "majority_refused"] for p in chosen] == [1.0, 0.0, 1.0, 0.0]


def test_summary_rates_and_agreement(world):
    cfg, data, _ = world
    t = stability.prompt_table(data, "judge_a", "partial_as_refusal")
    s = stability.model_summary(t, 200, 0, 0.95).set_index("model_id")
    row = s.loc[cfg.evaluated[0]]
    assert row["n_mixed"] == 2 and row["mixed_rate"] == 0.5
    # greedy matches the majority for prompts 0 and 3; prompts 1 and 2 disagree
    assert row["greedy_majority_agreement"] == 0.5
    shares = stability.pattern_shares(t)
    sums = shares.groupby(["model_id", "expected_behavior"])["share"].sum()
    assert ((sums - 1).abs() < 1e-9).all()


def test_run_writes_tables_and_figures(world, tmp_path):
    cfg, data, _ = world
    names = {p.name for p in stability.run(data, cfg, tmp_path)}
    for stem in ("stability_patterns", "stability_greedy_vs_sampled",
                 "stability_prompt_heatmap"):
        assert {f"{stem}.png", f"{stem}.pdf"} <= names
    assert {"stability_summary_primary.csv", "stability_prompts_primary.csv",
            "stability_mixed_by_evaluator.csv"} <= names
    wide = pd.read_csv(tmp_path / "stability_mixed_by_evaluator.csv")
    assert set(EVALS) <= set(wide.columns)


def test_run_without_stability_samples_writes_nothing(cfg, tmp_path):
    build(cfg)
    assert stability.run(load_analysis_data(cfg), cfg, tmp_path) == []
