from collections import Counter

import pandas as pd

from srs.data.sources._common import make_prompt
from srs.reference.sampling import allocate, draw_sample, eligible_units, write_sample
from srs.schemas import Judgment, Message, Response

MODELS = ["gemini-a", "glm-a", "kimi-a"]


def _world(n_prompts=40):
    prompts, responses, judgments = [], [], []
    for i in range(n_prompts):
        p = make_prompt("x", str(i), f"q{i}", "refuse" if i % 2 else "comply", None).model_copy(
            update={"category": "privacy", "split": "main"})
        prompts.append(p)
        for m in MODELS:
            rid = f"{m}-{i}"
            status = "api_blocked" if i % 10 == 0 else "ok"
            transcript = [Message(role="user", content=f"q{i}")]
            if status == "ok":
                transcript.append(Message(role="assistant", content="answer"))
            responses.append(Response(response_id=rid, prompt_id=p.prompt_id, model_id=m,
                                      gen_params={}, transcript=transcript,
                                      final_text="answer" if status == "ok" else None,
                                      status=status, run_id="g"))
            disagree = i % 4 == 1
            for e, lab in [("keyword", "full_refusal"),
                           ("judge_a", "full_compliance" if disagree else "full_refusal")]:
                judgments.append(Judgment(item_id=rid, evaluator_id=e, evaluator_version="v",
                                          refusal=lab, status="ok"))
    return prompts, responses, judgments


def test_allocate_caps_and_weights():
    alloc = allocate({"a": 100, "b": 100, "c": 5}, {"a": 1.0, "b": 3.0, "c": 3.0}, 40)
    assert sum(alloc.values()) == 40 and alloc["b"] > alloc["a"] and alloc["c"] <= 5


def test_sample_probabilities_and_exclusions(tmp_path):
    prompts, responses, judgments = _world()
    units = eligible_units(responses, prompts, judgments, ("main",))
    assert len(units) == 3 * 36  # blocked responses are not eligible
    sample = draw_sample(units, 30, seed=5, oversample=3.0)
    assert len(sample) == 30
    assert all(0 < p <= 1 for _, p in sample)
    blocked = {r.response_id for r in responses if r.status != "ok"}
    assert not blocked & {u.response.response_id for u, _ in sample}
    dis = [p for u, p in sample if u.cell[3]]
    agree = [p for u, p in sample if not u.cell[3]]
    assert min(dis) > max(agree)  # disagreements oversampled
    # weighting back recovers the population size per cell
    est = Counter()
    for u, p in sample:
        est[u.cell] += 1 / p
    for cell, total in est.items():
        assert round(total) == sum(1 for x in units if x.cell == cell)

    write_sample(tmp_path, sample, 5, len(units), "main")
    items = pd.read_parquet(tmp_path / "items.parquet")
    assert list(items.columns) == ["item_id", "turns"]  # blinded content only


def test_same_seed_same_sample():
    prompts, responses, judgments = _world()
    units = eligible_units(responses, prompts, judgments, ("main",))
    a = [u.response.response_id for u, _ in draw_sample(units, 30, 5, 3.0)]
    b = [u.response.response_id for u, _ in draw_sample(list(reversed(units)), 30, 5, 3.0)]
    c = [u.response.response_id for u, _ in draw_sample(units, 30, 6, 3.0)]
    assert sorted(a) == sorted(b) and sorted(a) != sorted(c)
