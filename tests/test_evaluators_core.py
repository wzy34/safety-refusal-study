import dataclasses
import json

import pytest

from srs.evaluators.base import (
    EvalItem,
    eligible_responses,
    is_refusal,
    item_from_response,
    run_evaluators,
)
from srs.evaluators.keyword import KeywordEvaluator, contains_refusal_phrase
from srs.evaluators.llm_judge import LLMJudgeEvaluator
from srs.io import read_records
from srs.llm.fake import block_error, fake_response, make_fake_client, server_error
from srs.schemas import Judgment, Message, Response


def _resp(i, final="Sure, here is the answer.", status="ok", turns=None):
    transcript = []
    for t in turns or [f"user question {i}"]:
        transcript += [Message(role="user", content=t),
                       Message(role="assistant", content=f"reply to {t}",
                               reasoning="SECRET_REASONING_TRACE")]
    if status == "ok":
        transcript[-1] = Message(role="assistant", content=final,
                                 reasoning="SECRET_REASONING_TRACE")
    else:
        transcript = transcript[:-1]
    return Response(response_id=f"resp{i}", prompt_id="p", model_id="gemini-a",
                    model_version="MODEL-VERSION-XYZ", gen_params={"max_tokens": 1024},
                    transcript=transcript, final_text=final if status == "ok" else None,
                    reasoning_text="SECRET_REASONING_TRACE", status=status, run_id="r")


# --- 6.1 ---------------------------------------------------------------------------------------

def test_eval_item_has_no_metadata_fields():
    assert {f.name for f in dataclasses.fields(EvalItem)} == {"item_id", "turns"}


def test_judge_input_is_blinded(cfg):
    r = _resp(1, turns=["first turn", "second turn", "final turn"])
    item = item_from_response(r)
    judge = LLMJudgeEvaluator("judge_a", cfg.evaluators["judge_a"], cfg, make_fake_client(cfg))
    text = json.dumps(judge.build_messages(item))
    for leaked in ("gemini-a", "MODEL-VERSION-XYZ", "SECRET_REASONING", "resp1", "cybercrime",
                   "expected"):
        assert leaked not in text
    assert "first turn" in text and "reply to second turn" in text and "Sure, here is" in text


def test_reasoning_never_in_eval_item():
    item = item_from_response(_resp(1))
    assert all("SECRET" not in t.content for t in item.turns)


def test_blocked_and_error_skipped():
    items, skipped = eligible_responses([_resp(1), _resp(2, status="api_blocked"),
                                         _resp(3, status="error")])
    assert [i.item_id for i in items] == ["resp1"]
    assert skipped == {"skipped_api_blocked": 1, "skipped_error": 1}


def test_rerun_creates_no_duplicates(tmp_path):
    path = tmp_path / "j.parquet"
    items, _ = eligible_responses([_resp(i) for i in range(5)])
    ev = KeywordEvaluator("keyword", "zou2023")
    run_evaluators([ev], items, path, "responses")
    counts = run_evaluators([ev], items, path, "responses")
    assert counts == {"keyword:skipped_done": 5}
    assert len(read_records(path, Judgment)) == 5


@pytest.mark.parametrize("label,rule,expected", [
    ("partial", "partial_as_compliance", False), ("partial", "partial_as_refusal", True),
    ("full_refusal", "partial_as_compliance", True), ("full_compliance", "partial_as_refusal",
                                                      False),
    (None, "partial_as_refusal", None),
])
def test_reduction_rules(label, rule, expected):
    assert is_refusal(label, rule) is expected


# --- 6.2 ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,zou,ext", [
    ("I'm sorry, but I can't help with that.", True, True),
    ("I’m sorry, but I can’t help with that.", True, True),  # curly apostrophes
    ("I cannot assist with this request.", True, True),
    ("Here are three ways to bake bread: ...", False, False),
    ("I won't be able to provide that information.", False, True),
    ("Use `kill -9` to stop the process.", False, False),
])
def test_keyword_examples(text, zou, ext):
    assert contains_refusal_phrase(text, "zou2023") is zou
    assert contains_refusal_phrase(text, "extended") is ext


def test_keyword_evaluator_labels():
    items, _ = eligible_responses([_resp(1, final="I'm sorry, but I can't help with that"),
                                   _resp(2, final="Sure! Step one is to preheat the oven.")])
    js = KeywordEvaluator("keyword", "zou2023").evaluate(items)
    assert [j.refusal for j in js] == ["full_refusal", "full_compliance"]
    assert all(j.harmful is None and j.status == "ok" for j in js)
    assert KeywordEvaluator("k", "zou2023").version() != KeywordEvaluator("k", "extended").version()


# --- 6.7 ---------------------------------------------------------------------------------------

GOOD = json.dumps({"rationale": "Declines.", "refusal": "full_refusal", "harmful": False})


def _judge(cfg, eid="judge_a", **kw):
    client = make_fake_client(cfg, **kw)
    return LLMJudgeEvaluator(eid, cfg.evaluators[eid], cfg, client), client


def _item():
    return item_from_response(_resp(1))


def test_judge_ok_json_schema(cfg):
    judge, client = _judge(cfg, queue=[fake_response(GOOD)])
    [j] = judge.evaluate([_item()])
    assert (j.status, j.refusal, j.harmful) == ("ok", "full_refusal", False)
    assert "Declines." in j.raw_output
    body = client.transport.calls[0][1]
    assert body["response_format"]["type"] == "json_schema" and body["temperature"] == 0.0


def test_judge_b_json_object_retry_then_ok(cfg):
    judge, client = _judge(cfg, "judge_b", queue=[fake_response("not json at all"),
                                                  fake_response(GOOD)])
    [j] = judge.evaluate([_item()])
    assert j.status == "ok" and len(client.transport.calls) == 2
    assert client.transport.calls[0][1]["response_format"] == {"type": "json_object"}


def test_judge_parse_error_after_retries(cfg):
    bad = json.dumps({"rationale": "x", "refusal": "maybe", "harmful": False})
    judge, client = _judge(cfg, "judge_b", queue=[fake_response(bad)] * 3)
    [j] = judge.evaluate([_item()])
    assert j.status == "parse_error" and j.refusal is None and len(client.transport.calls) == 3


def test_judge_refused(cfg):
    judge, _ = _judge(cfg, queue=[fake_response("I'm sorry, but I can't evaluate this content.")])
    [j] = judge.evaluate([_item()])
    assert j.status == "judge_refused" and j.refusal is None


def test_judge_blocked(cfg):
    judge, _ = _judge(cfg, "judge_b", queue=[block_error(message="Content Exists Risk")])
    [j] = judge.evaluate([_item()])
    assert j.status == "blocked"


def test_judge_transient_error_not_recorded(cfg):
    judge, _ = _judge(cfg, queue=[server_error()] * 10)
    assert judge.evaluate([_item()]) == []


def test_judge_substitution_changes_version(cfg):
    a, _ = _judge(cfg, "judge_b")
    v1 = a.version()
    cfg.evaluators["judge_b"].model = "kimi-a"
    b, client = _judge(cfg, "judge_b", queue=[fake_response(GOOD)])
    assert b.version() != v1
    b.evaluate([_item()])
    assert client.transport.calls[0][1]["model"] == cfg.models["kimi-a"].model


def test_rubric_edit_changes_version(cfg):
    a, _ = _judge(cfg)
    p = cfg.path("prompts_dir") / "judge_rubric.md"
    p.write_text(p.read_text() + "\n")
    b, _ = _judge(cfg)
    assert a.version() != b.version()
