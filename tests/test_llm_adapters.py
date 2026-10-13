import itertools

import pytest

from srs.config import ModelConfig
from srs.llm.adapters import GenParams, StructuredSpec, build_request

MSGS = [{"role": "user", "content": "hi"}]
SPEC = StructuredSpec("verdict", {"type": "object", "properties": {"a": {"type": "string"}}})


def _model(**kw):
    return ModelConfig(provider="p", model="m-1", family="f", **kw)


@pytest.mark.parametrize("token_param,temp,seed", list(itertools.product(
    ["max_tokens", "max_completion_tokens"], [True, False], [True, False])))
def test_capability_combinations(token_param, temp, seed):
    m = _model(token_param=token_param, supports_temperature=temp, supports_seed=seed)
    req = build_request(m, MSGS, GenParams(max_tokens=1024, temperature=0.0, seed=7))
    other = {"max_tokens", "max_completion_tokens"} - {token_param}
    assert req.body[token_param] == 1024 and not other & set(req.body)
    assert ("temperature" in req.body) == temp
    assert ("seed" in req.body) == seed
    assert ("temperature" in req.dropped) == (not temp)
    assert ("seed" in req.dropped) == (not seed)
    assert req.body["model"] == "m-1" and req.body["messages"] == MSGS


def test_unset_params_not_dropped():
    req = build_request(_model(supports_temperature=False), MSGS, GenParams(max_tokens=1024))
    assert req.dropped == [] and "temperature" not in req.body


def test_structured_json_schema():
    req = build_request(_model(structured_output="json_schema"), MSGS, GenParams(1024), SPEC)
    assert req.structured_mode == "json_schema"
    assert req.body["response_format"]["json_schema"]["schema"] == SPEC.schema


def test_structured_falls_back_to_json_object():
    req = build_request(_model(structured_output="json_object"), MSGS, GenParams(1024), SPEC)
    assert req.structured_mode == "json_object"
    assert req.body["response_format"] == {"type": "json_object"}


def test_structured_unsupported_dropped():
    req = build_request(_model(structured_output="none"), MSGS, GenParams(1024), SPEC)
    assert req.structured_mode is None and "response_format" not in req.body
    assert "response_format" in req.dropped


def test_extra_body_passed_separately():
    req = build_request(_model(extra_body={"thinking": {"type": "enabled"}}), MSGS, GenParams(1024))
    assert req.extra_body == {"thinking": {"type": "enabled"}} and "thinking" not in req.body
