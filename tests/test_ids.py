import pytest

from srs import ids


def test_canonical_json_key_order():
    assert ids.canonical_json({"b": 1, "a": [1, 2]}) == ids.canonical_json({"a": [1, 2], "b": 1})


def test_ids_are_stable():
    assert ids.intent_id("xstest", "1") == ids.intent_id("xstest", "1")
    assert ids.prompt_id("x", "1", "direct", ["hi"]) == ids.prompt_id("x", "1", "direct", ["hi"])
    h = ids.gen_params_hash({"temperature": 0.0, "max_tokens": 1024})
    assert h == ids.gen_params_hash({"max_tokens": 1024, "temperature": 0.0})
    assert ids.response_id("p", "m", 0, h) == ids.response_id("p", "m", 0, h)
    assert len(ids.intent_id("a", "b")) == ids.ID_LEN


@pytest.mark.parametrize("change", [
    {"source": "y"}, {"source_id": "2"}, {"framing": "academic"}, {"turns": ["hi", "there"]},
])
def test_prompt_id_sensitive_to_each_input(change):
    base = {"source": "x", "source_id": "1", "framing": "direct", "turns": ["hi"]}
    assert ids.prompt_id(**base) != ids.prompt_id(**{**base, **change})


@pytest.mark.parametrize("change", [{"source": "y"}, {"source_id": "2"}])
def test_intent_id_sensitive(change):
    base = {"source": "x", "source_id": "1"}
    assert ids.intent_id(**base) != ids.intent_id(**{**base, **change})


def test_intent_id_ignores_framing():
    # all framings of an intent share its intent id because intent_id has no framing input
    assert ids.intent_id("x", "1") != ids.prompt_id("x", "1", "direct", ["hi"])


@pytest.mark.parametrize("change", [
    {"prompt_id": "q"}, {"model_id": "n"}, {"sample_idx": 1}, {"gen_params_hash": "h2"},
])
def test_response_id_sensitive_to_each_input(change):
    base = {"prompt_id": "p", "model_id": "m", "sample_idx": 0, "gen_params_hash": "h"}
    assert ids.response_id(**base) != ids.response_id(**{**base, **change})


def test_gen_params_hash_sensitive():
    assert ids.gen_params_hash({"max_tokens": 1024}) != ids.gen_params_hash({"max_tokens": 2048})
