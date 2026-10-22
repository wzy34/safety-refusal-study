import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from srs import cli
from srs.evaluators import classic, safety_model
from srs.evaluators.base import judgments_path
from srs.generation.runner import responses_path
from srs.io import read_records, write_records
from srs.llm.fake import fake_response, make_fake_client
from srs.schemas import Judgment, Message, Response
from tests.test_safety_model import fake_wildguard
from tests.tiny_models import tiny_factory

RAW = Path(__file__).parent / "fixtures" / "raw"
ALL = {"keyword", "keyword_ext", "classic", "wildguard", "judge_a", "judge_b"}


def _responses():
    out = []
    for i, (text, status) in enumerate([("i am sorry but i cannot help", "ok"),
                                        ("sure here is the recipe", "ok"),
                                        (None, "api_blocked")]):
        transcript = [Message(role="user", content=f"q{i}")]
        if text:
            transcript.append(Message(role="assistant", content=text))
        out.append(Response(response_id=f"r{i}", prompt_id=f"p{i}", model_id="gemini-a",
                            gen_params={}, transcript=transcript, final_text=text,
                            status=status, run_id="gen"))
    return out


def _setup(cfg, monkeypatch):
    ecfg = cfg.evaluators["classic"].model_copy(update={"max_epochs": 1, "batch_size": 4})
    items = [classic.TrainItem("p", t, int("sorry" in t)) for t in
             ["i am sorry but i cannot help", "sure here is the recipe"] * 6]
    classic.train_classifier(items, ecfg, cfg.root / ecfg.checkpoint_dir, 1,
                             factory=tiny_factory)

    class FakeMLX:
        def __init__(self, path, max_new_tokens):
            pass

        def __call__(self, prompt):
            return fake_wildguard(prompt)

    monkeypatch.setattr(safety_model, "MLXGenerator", FakeMLX)
    verdict = json.dumps({"rationale": "r", "refusal": "partial", "harmful": False})
    client = make_fake_client(cfg, responder=lambda p, b: fake_response(verdict))
    monkeypatch.setattr(cli, "_real_client", lambda c, ids: client)
    monkeypatch.setitem(cli._state, "cfg", cfg)
    return client


def test_evaluate_responses_all_evaluators(cfg, monkeypatch):
    write_records(responses_path(cfg), _responses())
    _setup(cfg, monkeypatch)
    res = CliRunner().invoke(cli.app, ["evaluate"])
    assert res.exit_code == 0, res.output
    js = read_records(judgments_path(cfg, "responses"), Judgment)
    assert {j.evaluator_id for j in js} == ALL
    assert {j.item_id for j in js} == {"r0", "r1"}  # blocked response not evaluated
    assert len(js) == 2 * len(ALL) and all(j.target == "responses" for j in js)
    assert "skipped_api_blocked" in res.output

    again = CliRunner().invoke(cli.app, ["evaluate", "--evaluators", "keyword"])
    assert again.exit_code == 0 and len(read_records(judgments_path(cfg, "responses"),
                                                     Judgment)) == len(js)


def test_evaluate_external_kept_apart(cfg, monkeypatch):
    shutil.copytree(RAW, cfg.path("data_dir") / "raw")
    _setup(cfg, monkeypatch)
    res = CliRunner().invoke(cli.app, ["evaluate", "--target", "external", "--evaluators",
                                       "keyword", "--evaluators", "judge_a"])
    assert res.exit_code == 0, res.output
    assert "skipping: wildguardmix" in res.output
    ext = read_records(judgments_path(cfg, "external"), Judgment)
    assert {j.evaluator_id for j in ext} == {"keyword", "judge_a"}
    assert all(j.target == "external" for j in ext)
    assert len(ext) == 2 * (2 + 2 + 15)  # jbb_judge + harmbench_val + xstest_completions
    assert not judgments_path(cfg, "responses").exists()


def test_evaluate_external_sources_filter(cfg, monkeypatch):
    shutil.copytree(RAW, cfg.path("data_dir") / "raw")
    _setup(cfg, monkeypatch)
    res = CliRunner().invoke(cli.app, ["evaluate", "--target", "external", "--evaluators",
                                       "judge_a", "--sources", "harmbench_val",
                                       "--sources", "jbb_judge"])
    assert res.exit_code == 0, res.output
    ext = read_records(judgments_path(cfg, "external"), Judgment)
    assert {j.evaluator_id for j in ext} == {"judge_a"} and len(ext) == 2 + 2
    bad = CliRunner().invoke(cli.app, ["evaluate", "--target", "external", "--sources", "nope"])
    assert bad.exit_code == 2 and "nope" in bad.output
    wrong = CliRunner().invoke(cli.app, ["evaluate", "--sources", "jbb_judge"])
    assert wrong.exit_code == 2
