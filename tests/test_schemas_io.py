import pytest
from pydantic import ValidationError

from srs.io import read_frame, read_records, upsert_records, write_records
from srs.schemas import HumanLabel, Judgment, Message, Prompt, ReferenceItem, Response


def _prompt(i=0, turns=None, **kw):
    return Prompt(prompt_id=f"p{i}", intent_id=f"i{i}", source="xstest", source_id=str(i),
                  category="privacy", expected_behavior="comply",
                  turns=turns or ["hello"], **kw)


def _response(i=0, **kw):
    kw = {"final_text": "a2", "status": "ok", **kw}
    return Response(
        response_id=f"r{i}", prompt_id="p0", model_id="gemini-a",
        gen_params={"temperature": 0.0, "max_tokens": 1024}, run_id="run1",
        transcript=[Message(role="user", content="t1"),
                    Message(role="assistant", content="a1", reasoning="think"),
                    Message(role="user", content="t2"),
                    Message(role="assistant", content="a2")],
        usage={"prompt_tokens": 3, "completion_tokens": 5}, **kw)


def test_prompt_roundtrip_multiturn(tmp_path):
    recs = [_prompt(0), _prompt(1, turns=["a", "b", "c"], framing="multiturn", pair_id="x")]
    write_records(tmp_path / "p.parquet", recs)
    assert read_records(tmp_path / "p.parquet", Prompt) == recs


def test_response_roundtrip_transcript(tmp_path):
    recs = [_response(0), _response(1, status="api_blocked", block_turn=2, final_text=None)]
    write_records(tmp_path / "r.parquet", recs)
    back = read_records(tmp_path / "r.parquet", Response)
    assert back == recs
    assert back[0].transcript[1].reasoning == "think"
    assert back[0].usage["completion_tokens"] == 5
    assert isinstance(read_frame(tmp_path / "r.parquet")["usage"][0], str)


def test_other_tables_roundtrip(tmp_path):
    j = [Judgment(item_id="r0", evaluator_id="keyword", evaluator_version="v1",
                  refusal="full_refusal", status="ok")]
    h = [HumanLabel(item_id="r0", pass_num=1, refusal="partial", harmful=False, seconds=4.2)]
    ref = [ReferenceItem(item_id="x", source="xstest_resp", source_id="1", prompt="p",
                         response="r", refusal="full_compliance")]
    for name, recs, cls in [("j", j, Judgment), ("h", h, HumanLabel), ("ref", ref, ReferenceItem)]:
        write_records(tmp_path / f"{name}.parquet", recs)
        assert read_records(tmp_path / f"{name}.parquet", cls) == recs


def test_upsert_replaces_by_key(tmp_path):
    path = tmp_path / "r.parquet"
    upsert_records(path, Response, [_response(0, status="error")], key=lambda r: r.response_id)
    n = upsert_records(path, Response, [_response(0), _response(1)], key=lambda r: r.response_id)
    assert n == 2
    assert {r.status for r in read_records(path, Response)} == {"ok"}


def test_empty_first_turn_rejected():
    with pytest.raises(ValidationError):
        _prompt(turns=["  "])


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        Judgment(item_id="r", evaluator_id="k", evaluator_version="v", status="ok", model="gemini")


def _upsert_judgments(path, worker, n):
    for i in range(n):
        upsert_records(path, Judgment,
                       [Judgment(item_id=f"r{i}", evaluator_id=f"ev{worker}",
                                 evaluator_version="v1", refusal="full_refusal", status="ok")],
                       key=lambda j: (j.item_id, j.evaluator_id))


def test_concurrent_upserts_lose_nothing(tmp_path):
    """Each upsert opens its own lock file handle, so flock serializes them across threads
    just as it does across processes."""
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "j.parquet"
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda w: _upsert_judgments(path, w, 15), range(4)))
    assert len(read_records(path, Judgment)) == 60
