from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from srs.reference import annotations as A
from srs.schemas import HumanLabel

APP = str(A.APP_PATH)


@pytest.fixture
def ann_dir(tmp_path, monkeypatch):
    d = tmp_path / "ann"
    d.mkdir()
    pd.DataFrame([{"item_id": f"item{i:02d}",
                   "turns": [{"role": "user", "content": f"Question number {i}"},
                             {"role": "assistant", "content": f"Answer number {i}"}]}
                  for i in range(10)]).to_parquet(d / "items.parquet")
    monkeypatch.setenv("SRS_ANNOTATION_DIR", str(d))
    monkeypatch.setenv("SRS_SEED", "3")
    monkeypatch.setenv("SRS_PASS2_MIN_DAYS", "7")
    return d


def _app(pass_="1"):
    import os
    os.environ["SRS_ANNOTATION_PASS"] = pass_
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def _visible(at) -> str:
    parts = [e.value for e in (*at.markdown, *at.text, *at.caption, *at.info, *at.error,
                               *at.success)]
    return "\n".join(str(p) for p in parts)


def _label(at, refusal="partial", harmful=False):
    at.button(key=f"refusal_{refusal}").click().run()
    at.button(key="harmful_yes" if harmful else "harmful_no").click().run()
    at.button(key="save").click().run()
    assert not at.exception, at.exception


def test_label_three_restart_resumes_at_four(ann_dir):
    at = _app()
    assert "item 1 of 10" in _visible(at)
    assert at.button(key="save").disabled  # both labels required
    shown = []
    for k in range(3):
        shown.append(next(t.value for t in at.text if t.value.startswith("Question")))
        _label(at, ["full_compliance", "partial", "full_refusal"][k], harmful=k == 2)
    saved = A.load_labels(ann_dir, 1)
    assert len(saved) == 3 and all(x.seconds is not None for x in saved)

    restarted = _app()
    text = _visible(restarted)
    assert "item 4 of 10" in text and "3 labeled" in text
    current = next(t.value for t in restarted.text if t.value.startswith("Question"))
    assert current not in shown
    for leaked in ("item0", "model", "gemini", "gpt", "expected", "category", "evaluator", "judge",
                   "keyword", "wildguard"):
        assert leaked not in text.lower()


def test_pass2_gating_and_hidden_labels(ann_dir):
    now = datetime.now(UTC)
    for i in range(10):
        A.save_label(ann_dir, HumanLabel(item_id=f"item{i:02d}", pass_num=1,
                                         refusal="full_refusal", harmful=False,
                                         labeled_at=(now - timedelta(days=2)).isoformat()))
    gate = A.pass_gate(ann_dir, 2, 10, 7)
    assert not gate.open and gate.available_at is not None
    assert gate.available_at.date() == (now + timedelta(days=5)).date()
    at = _app("2")
    assert at.error and "becomes available on" in at.error[0].value
    assert (now + timedelta(days=5)).astimezone().strftime("%Y-%m-%d") in at.error[0].value
    assert A.pass_gate(ann_dir, 2, 10, 7, now=now + timedelta(days=6)).open

    # pass 2 open: new order, and pass-1 labels are not shown
    import os
    os.environ["SRS_PASS2_MIN_DAYS"] = "1"
    at = _app("2")
    assert "item 1 of 10" in _visible(at) and not at.info
    ids = [f"item{i:02d}" for i in range(10)]
    assert A.queue(ids, 1, 3) != A.queue(ids, 2, 3)


def test_pass2_requires_complete_pass1(ann_dir):
    A.save_label(ann_dir, HumanLabel(item_id="item00", pass_num=1, refusal="partial",
                                     harmful=False))
    gate = A.pass_gate(ann_dir, 2, 10, 0)
    assert not gate.open and "1/10" in gate.message


def test_kappa_known_and_adjudication(ann_dir):
    p1 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    p2 = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]  # po = 0.8, pe = 0.5 -> kappa = 0.6
    lab = {1: "full_refusal", 0: "full_compliance"}
    for i, (a, b) in enumerate(zip(p1, p2, strict=True)):
        A.save_label(ann_dir, HumanLabel(item_id=f"item{i:02d}", pass_num=1, refusal=lab[a],
                                         harmful=False))
        A.save_label(ann_dir, HumanLabel(item_id=f"item{i:02d}", pass_num=2, refusal=lab[b],
                                         harmful=i == 9))
    rep = A.import_annotations(ann_dir)
    assert rep["refusal_kappa"] == pytest.approx(0.6)
    assert {x["item_id"] for x in rep["disagreements"]} == {"item04", "item09"}
    assert sorted(rep["pending_adjudication"]) == ["item04", "item09"] and rep["n_final"] == 8

    A.save_label(ann_dir, HumanLabel(item_id="item04", pass_num=0, refusal="partial",
                                     harmful=False))
    A.save_label(ann_dir, HumanLabel(item_id="item09", pass_num=0, refusal="full_refusal",
                                     harmful=True))
    rep = A.import_annotations(ann_dir)
    assert rep["pending_adjudication"] == [] and rep["n_final"] == 10
    finals = {x.item_id: x for x in A.read_records(ann_dir / "human_labels.parquet",
                                                    HumanLabel) if x.pass_num == 0}
    assert finals["item04"].refusal == "partial" and finals["item09"].harmful is True
    assert Path(ann_dir / "reliability.md").exists()


def test_adjudication_app_shows_only_disagreements(ann_dir, monkeypatch):
    for i in range(10):
        for n in (1, 2):
            A.save_label(ann_dir, HumanLabel(
                item_id=f"item{i:02d}", pass_num=n, harmful=False,
                refusal="partial" if (n == 2 and i == 5) else "full_refusal"))
    at = _app("adjudicate")
    text = _visible(at)
    assert "item 1 of 1" in text and "Pass 1:" in text and "Question number 5" in text
