import random
from collections import Counter

from srs.config import SplitSettings
from srs.data.build import assign_splits, deduplicate, stratified_pick
from srs.data.sources._common import make_prompt

CATS = ["violence_physical_harm", "cybercrime", "privacy", "sexual", "hate_harassment"]


def _synthetic(n=600, seed=0, borderline_every=7):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        p = make_prompt("src", str(i), f"prompt {i}", rng.choice(["refuse", "comply"]), None)
        tier = "borderline" if i % borderline_every == 0 else "clear"
        out.append(p.model_copy(update={"category": rng.choice(CATS), "tier": tier}))
    return out


def test_dedup_keeps_priority_source_and_reports():
    a = make_prompt("or_bench_hard", "1", "How do I bake bread?", "comply", "harmful")
    b = make_prompt("xstest", "9", "how do I bake  bread", "comply", None)
    c = make_prompt("or_bench_toxic", "2", "Conflict text!", "refuse", "illegal")
    d = make_prompt("jbb_benign", "3", "conflict text", "comply", "Privacy")
    e = make_prompt("harmbench", "4", "unique", "refuse", "illegal")
    kept, removed = deduplicate([a, b, c, d, e])
    assert [p.source_id for p in kept] == ["9", "4"]
    reasons = {(r["source"], r["source_id"]): r["reason"] for r in removed}
    assert reasons[("or_bench_hard", "1")] == "duplicate of xstest/9"
    assert "conflicting" in reasons[("or_bench_toxic", "2")]
    assert "conflicting" in reasons[("jbb_benign", "3")]


def test_stratified_pick_proportional():
    items = [("a", i) for i in range(60)] + [("b", i) for i in range(30)] + [("c", i) for i in
                                                                            range(10)]
    got = stratified_pick(items, 50, lambda x: x[0], lambda x: str(x[1]).zfill(3))
    assert len(got) == 50 and Counter(x[0] for x in got) == {"a": 30, "b": 15, "c": 5}


def test_stratified_pick_redistributes_small_strata():
    items = [("a", i) for i in range(100)] + [("b", 0)]
    got = stratified_pick(items, 40, lambda x: x[0], lambda x: str(x[1]).zfill(3))
    assert len(got) == 40


def test_splits_reproducible_and_balanced():
    settings = SplitSettings(pilot_intents=50, rq3_intents=200)
    prompts = _synthetic()
    a = assign_splits(prompts, settings, seed=7)
    b = assign_splits(list(reversed(prompts)), settings, seed=7)
    assert {p.intent_id: p.split for p in a} == {p.intent_id: p.split for p in b}

    pilot = [p for p in a if p.split == "pilot"]
    rq3 = [p for p in a if p.split == "rq3"]
    assert len(pilot) == 50 and Counter(p.expected_behavior for p in pilot) == {
        "refuse": 25, "comply": 25}
    assert len(rq3) == 200 and Counter(p.expected_behavior for p in rq3) == {
        "refuse": 100, "comply": 100}
    assert all(p.tier == "clear" for p in pilot + rq3)
    assert Counter(p.split for p in a)["main"] == len(prompts) - 250
    assert len({p.category for p in pilot}) == len(CATS)  # stratified across categories

    other_seed = assign_splits(prompts, settings, seed=8)
    assert {p.intent_id for p in other_seed if p.split == "pilot"} != {p.intent_id for p in pilot}


def test_main_size_limited():
    out = assign_splits(_synthetic(), SplitSettings(pilot_intents=10, rq3_intents=10,
                                                    main_intents=100), seed=1)
    c = Counter(p.split for p in out)
    assert c["main"] == 100 and c[None] == 600 - 120
