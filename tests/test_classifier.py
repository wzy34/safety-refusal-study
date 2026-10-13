import json

import pytest

from srs.evaluators.base import EvalItem, Turn
from srs.evaluators.classic import (
    ClassifierEvaluator,
    LeakageError,
    TrainItem,
    apply_leakage_guard,
    encode_middle_truncated,
    train_classifier,
)
from srs.schemas import ReferenceItem
from tests.tiny_models import tiny_factory, tiny_tokenizer


def _ref(prompt, response):
    return ReferenceItem(item_id=prompt, source="wildguardtest", source_id=prompt, prompt=prompt,
                         response=response)


def test_leakage_guard_removes_overlap():
    train = [TrainItem("p1", "r1", 1), TrainItem("P1 ", " r1", 0), TrainItem("p2", "r2", 0)]
    kept, removed = apply_leakage_guard(train, [_ref("p1", "R1"), _ref("zz", "yy")])
    assert removed == 2 and kept == [TrainItem("p2", "r2", 0)]
    with pytest.raises(LeakageError):
        apply_leakage_guard(train, [])


def test_middle_truncation_keeps_start_and_end():
    tok = tiny_tokenizer()
    text = "sorry " + "flour " * 50 + "logs"
    ids = encode_middle_truncated(tok, text, 12)
    assert len(ids) == 12
    toks = tok.convert_ids_to_tokens(ids)
    assert toks[0] == "<s>" and toks[-1] == "</s>" and toks[1] == "sorry" and toks[-2] == "logs"
    short = encode_middle_truncated(tok, "sure here", 12)
    assert tok.convert_ids_to_tokens(short) == ["<s>", "sure", "here", "</s>"]


def _items(n=40):
    refusals = ["i am sorry but i cannot help with that request", "i must decline"]
    complies = ["sure here is the recipe step one", "the answer is to restart the process"]
    return [TrainItem(f"p{i}", (refusals if i % 2 else complies)[i % 4 // 2], i % 2)
            for i in range(n)]


@pytest.fixture
def trained(cfg, tmp_path):
    ecfg = cfg.evaluators["classic"].model_copy(update={"max_epochs": 2, "batch_size": 8,
                                                        "learning_rate": 5e-3})
    out = tmp_path / "ckpt"
    report = train_classifier(_items(), ecfg, out, seed=1, n_removed_by_guard=3,
                              factory=tiny_factory)
    return ecfg, out, report


def test_training_saves_checkpoint_and_report(trained):
    _, out, report = trained
    saved = json.loads((out / "validation_report.json").read_text())
    assert saved["version"] == report["version"] and saved["removed_by_leakage_guard"] == 3
    for k in ("accuracy", "precision", "recall", "f1"):
        assert 0 <= saved["best"][k] <= 1
    assert (out / "config.json").exists() and (out / "version.json").exists()


def test_classifier_inference_20_items(trained):
    ecfg, out, report = trained
    ev = ClassifierEvaluator("classic", ecfg, out)
    items = [EvalItem(f"r{i}", (Turn("user", "q"), Turn("assistant", t)))
             for i, t in enumerate(["i am sorry but i cannot help", "sure here is the recipe"]
                                   * 10)]
    js = ev.evaluate(items)
    assert len(js) == 20 and {j.refusal for j in js} <= {"full_refusal", "full_compliance"}
    assert all(j.evaluator_version == report["version"] and j.harmful is None for j in js)


def test_missing_checkpoint_message(cfg, tmp_path):
    ev = ClassifierEvaluator("classic", cfg.evaluators["classic"], tmp_path / "none")
    with pytest.raises(FileNotFoundError, match="srs classic train"):
        ev.version()
