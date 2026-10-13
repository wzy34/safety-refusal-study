import filecmp

import pandas as pd
import pytest
from typer.testing import CliRunner

from srs import cli
from srs.analysis import rq1, rq2, rq3, rq4, rq5
from srs.analysis.data import load_analysis_data
from tests.synthetic import EVALS, build


@pytest.fixture
def world(cfg):
    cfg.experiment.analysis.bootstrap_resamples = 200
    build(cfg)
    return cfg, load_analysis_data(cfg)


def test_rq1_pairwise_and_sensitivity(world, tmp_path):
    cfg, data = world
    files = {p.name for p in rq1.run(data, cfg, tmp_path)}
    pw = pd.read_csv(tmp_path / "rq1_pairwise.csv")
    assert pw.groupby("metric").size().to_dict() == {"far_comply": 28, "orr": 28}
    assert (pw["p_holm"] >= pw["p_value"]).all()
    for ev in EVALS:
        if ev != cfg.experiment.analysis.primary_evaluator:
            assert f"rq1_metrics_{ev}_partial_as_refusal.csv" in files
    m = pd.read_csv(tmp_path / "rq1_metrics_primary.csv")
    assert len(m) == 8 and m["rule"].unique().tolist() == ["partial_as_refusal"]
    assert {"far_comply", "orr", "j", "block_rate", "n_refuse"} <= set(m.columns)
    assert "rq1_metrics_judge_a_partial_as_compliance.csv" in files
    corr = pd.read_csv(tmp_path / "rq1_far_orr_correlation.csv")
    assert corr["rho"].iloc[0] > 0  # synthetic FAR and ORR both rise with model index


def test_rq2_outputs(world, tmp_path):
    cfg, data = world
    names = {p.name for p in rq2.run(data, cfg, tmp_path)}
    for n in ("rq2_heatmap_far_comply.png", "rq2_heatmap_far_comply.pdf",
              "rq2_heatmap_orr.png", "rq2_glm.csv", "rq2_kendalls_w.csv",
              "rq2_category_metrics_primary.csv"):
        assert n in names
    glm = pd.read_csv(tmp_path / "rq2_glm.csv")
    assert "se_cluster" in glm.columns and len(glm) > 10
    w = pd.read_csv(tmp_path / "rq2_kendalls_w.csv")
    assert set(w["metric"]) == {"far_comply", "orr"}


def test_rq3_known_flip_rates():
    rows = []
    # model m, refuse-expected: 10 intents; direct always refused; academic flips 4 of 10
    for i in range(10):
        for fr in ("direct", "academic", "roleplay", "hypothetical", "multiturn"):
            refused = not (fr == "academic" and i < 4)
            rows.append({"model_id": "m", "expected_behavior": "refuse", "intent_id": f"i{i}",
                         "framing": fr, "refused": refused})
    dec = pd.DataFrame(rows)
    fl = rq3.flip_rates(dec).set_index("framing")
    assert fl.loc["academic", "flip_rate"] == pytest.approx(0.4)
    assert fl.loc["roleplay", "flip_rate"] == 0.0
    assert fl.loc["academic", "refused_direct_only"] == 4
    assert rq3.consistency(dec)["consistency"].iloc[0] == pytest.approx(0.6)


def test_rq3_outputs(world, tmp_path):
    cfg, data = world
    names = {p.name for p in rq3.run(data, cfg, tmp_path)}
    assert {"rq3_flip_rates_primary.csv", "rq3_consistency_primary.csv",
            "rq3_refusal_rates_primary.csv", "rq3_flip_heatmap_refuse.png"} <= names
    fl = pd.read_csv(tmp_path / "rq3_flip_rates_primary.csv")
    assert set(fl["framing"]) == {"academic", "roleplay", "hypothetical", "multiturn"}
    assert set(fl["expected_behavior"]) == {"refuse", "comply"}


def test_rq4_pareto():
    pts = pd.DataFrame([{"model_id": "a", "far_comply": 0.1, "orr": 0.1},
                        {"model_id": "b", "far_comply": 0.2, "orr": 0.3},  # dominated by a
                        {"model_id": "c", "far_comply": 0.05, "orr": 0.4}])
    out = rq4.pareto(pts).set_index("model_id")
    assert out["pareto_optimal"].to_dict() == {"a": True, "b": False, "c": True}
    assert out.loc["a", "distance_to_origin"] == pytest.approx(0.1414, abs=1e-4)


def test_rq4_outputs(world, tmp_path):
    cfg, data = world
    names = {p.name for p in rq4.run(data, cfg, tmp_path)}
    assert {"rq4_plane_primary.csv", "rq4_plane.png", "rq4_plane.pdf"} <= names


def test_rq5_tables_complete(world, tmp_path):
    cfg, data = world
    rq5.run(data, cfg, tmp_path)
    expect = {"kappa_pairwise": 10, "accuracy_external": None, "accuracy_self": None,
              "bias": 8 * 5, "ranking_tau": 10, "self_preference": 4, "judge_status": None,
              "alpha": 1}
    for name, n in expect.items():
        df = pd.read_csv(tmp_path / f"rq5_{name}.csv")
        assert len(df) > 0, name
        if n is not None:
            assert len(df) == n, name
    ext = pd.read_csv(tmp_path / "rq5_accuracy_external.csv")
    assert set(ext["source"]) == {"wildguardtest", "xstest_completions"}
    assert set(ext[ext["source"] == "wildguardtest"]["label"]) == {"refusal", "harmful"}
    sp = pd.read_csv(tmp_path / "rq5_self_preference.csv")
    assert set(sp["evaluator_id"]) == {"judge_a", "judge_b"}
    js = pd.read_csv(tmp_path / "rq5_judge_status.csv")
    jb = js[(js["evaluator_id"] == "judge_b") & (js["target"] == "responses")].iloc[0]
    assert 0 < jb["rate_blocked"] < 0.1


def test_analyze_all_reproducible(world, monkeypatch):
    cfg, _ = world
    monkeypatch.setitem(cli._state, "cfg", cfg)
    for _ in range(2):
        res = CliRunner().invoke(cli.app, ["analyze", "all"])
        assert res.exit_code == 0, res.output
    runs = sorted((cfg.path("results_dir") / "analysis").iterdir())
    assert len(runs) == 2
    csvs = sorted(p.relative_to(runs[0]) for p in runs[0].rglob("*.csv"))
    assert len(csvs) > 20
    for rel in csvs:
        assert filecmp.cmp(runs[0] / rel, runs[1] / rel, shallow=False), rel
    import json
    manifest = json.loads(next(runs[0].glob("*.json")).read_text())
    assert manifest["extra"]["primary_evaluator"] == "judge_a"
    assert any("responses.parquet" in i for i in manifest["extra"]["inputs"])
    for rq in ("rq1", "rq2", "rq3", "rq4", "rq5"):
        assert (runs[0] / rq).is_dir()
