import shutil
from pathlib import Path

import pandas as pd
import pytest

from srs.data.sources import FetchError
from srs.reference.external import (
    load_harmbench_val,
    load_jbb_judge,
    load_reference_items,
    load_wildguardtest,
    load_xstest_completions,
    overlap_key,
)

RAW = Path(__file__).parent / "fixtures" / "raw"


@pytest.fixture
def raw_with_wildguard(tmp_path):
    raw = tmp_path / "raw"
    shutil.copytree(RAW, raw)
    (raw / "wildguardmix").mkdir()
    pd.DataFrame([
        {"prompt": "p1", "adversarial": False, "response": "I can't help.",
         "prompt_harm_label": "harmful", "response_refusal_label": "refusal",
         "response_harm_label": "unharmful", "subcategory": "x"},
        {"prompt": "p2", "adversarial": True, "response": "[synthetic]",
         "prompt_harm_label": "harmful", "response_refusal_label": "compliance",
         "response_harm_label": "harmful", "subcategory": "x"},
        {"prompt": "p3", "adversarial": False, "response": "Sure.",
         "prompt_harm_label": "unharmful", "response_refusal_label": None,
         "response_harm_label": None, "subcategory": "x"},
        {"prompt": "p4", "adversarial": False, "response": None,
         "prompt_harm_label": "unharmful", "response_refusal_label": None,
         "response_harm_label": None, "subcategory": "x"},
    ]).to_parquet(raw / "wildguardmix" / "wildguard_test.parquet")
    return raw


def test_wildguardtest(raw_with_wildguard):
    items = load_wildguardtest(raw_with_wildguard)
    assert [(i.refusal, i.harmful) for i in items] == [
        ("full_refusal", False), ("full_compliance", True), (None, None)]
    assert items[0].prompt_harmful is True and items[2].prompt_harmful is False


def test_jbb_judge():
    items = load_jbb_judge(RAW)
    assert [(i.refusal, i.harmful) for i in items] == [(None, False), (None, True)]
    assert items[0].prompt == "[jailbreak prompt one]"


def test_harmbench_val_majority():
    items = load_harmbench_val(RAW)
    assert [i.harmful for i in items] == [False, True] and all(i.refusal is None for i in items)


def test_xstest_completions():
    items = load_xstest_completions(RAW)
    assert len(items) == 15
    assert {i.refusal for i in items} == {"full_compliance", "full_refusal", "partial"}
    assert all(i.harmful is None for i in items)
    assert len({i.item_id for i in items}) == 15


def test_load_all_and_missing(raw_with_wildguard):
    items, missing = load_reference_items(raw_with_wildguard)
    assert missing == {} and {i.source for i in items} == {
        "wildguardtest", "jbb_judge", "harmbench_val", "xstest_completions"}
    with pytest.raises(FetchError, match="wildguardmix.*Access step"):
        load_reference_items(RAW)
    _, missing = load_reference_items(RAW, require_all=False)
    assert list(missing) == ["wildguardtest"]


def test_overlap_key_normalizes():
    assert overlap_key("Hello  World", "Ok") == overlap_key("hello world", "ok ")
    assert overlap_key("a", "b") != overlap_key("a", "c")
