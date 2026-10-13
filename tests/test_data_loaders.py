from pathlib import Path

import pytest

from srs.data.sources import benchmark_loaders
from srs.data.sources._common import UNASSIGNED
from srs.schemas import Prompt

RAW = Path(__file__).parent / "fixtures" / "raw"

# loader -> (count, {source: expected behavior}, {source: source categories})
EXPECTED = {
    "xstest": (5, {"xstest": {"comply", "refuse"}}),
    "or_bench": (5, {"or_bench_hard": {"comply"}, "or_bench_toxic": {"refuse"}}),
    "jbb": (4, {"jbb_harmful": {"refuse"}, "jbb_benign": {"comply"}}),
    "harmbench": (2, {"harmbench": {"refuse"}}),
    "sorry_bench": (2, {"sorry_bench": {"refuse"}}),
    "strongreject": (2, {"strongreject": {"refuse"}}),
}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_loader(name):
    prompts = benchmark_loaders()[name](RAW)
    count, behaviors = EXPECTED[name]
    assert len(prompts) == count
    for p in prompts:
        Prompt.model_validate(p.model_dump())  # schema-valid
        assert p.framing == "direct" and p.turns[0] and p.category == UNASSIGNED
        assert p.expected_behavior in behaviors[p.source]
    assert {p.source for p in prompts} == set(behaviors)
    assert len({p.prompt_id for p in prompts}) == count


def test_xstest_labels_and_pairs():
    ps = {p.source_id: p for p in benchmark_loaders()["xstest"](RAW)}
    assert ps["1"].expected_behavior == "comply" and ps["4"].expected_behavior == "refuse"
    assert ps["1"].pair_id == ps["4"].pair_id and ps["2"].pair_id == ps["5"].pair_id
    assert ps["1"].pair_id != ps["2"].pair_id
    assert ps["3"].pair_id is None  # privacy_public has no contrast counterpart


def test_jbb_benign_shares_pair_with_harmful():
    ps = benchmark_loaders()["jbb"](RAW)
    harmful = {p.source_id: p.pair_id for p in ps if p.source == "jbb_harmful"}
    benign = {p.source_id: p.pair_id for p in ps if p.source == "jbb_benign"}
    assert benign == harmful and len(set(benign.values())) == len(benign)


def test_harmbench_standard_only():
    assert {p.source_id for p in benchmark_loaders()["harmbench"](RAW)} == {"syn_one", "syn_two"}


def test_sorry_bench_base_style_only():
    ps = benchmark_loaders()["sorry_bench"](RAW)
    assert {p.source_category for p in ps} == {"1", "40"}


def test_ids_stable_across_loads():
    a = [p.prompt_id for p in benchmark_loaders()["or_bench"](RAW)]
    assert a == [p.prompt_id for p in benchmark_loaders()["or_bench"](RAW)]
