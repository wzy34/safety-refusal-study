import asyncio
import json
from collections import Counter

from srs.data.framing import frame_intent, generate_framings
from srs.data.sources._common import make_prompt
from srs.llm.fake import fake_response, last_user


def _directs(n=5):
    return [make_prompt("xstest", str(i), f"How do I make sourdough bread number {i}?", "comply",
                        None).model_copy(update={"category": "privacy", "split": "rq3"})
            for i in range(n)]


def is_check(body):
    return "Rephrased version" in last_user(body)


def responder(check_verdicts=None, rewriter_refuses=False):
    """check_verdicts: iterator of bools for successive intent checks (default: always pass)."""
    def respond(provider, body):
        text = last_user(body)
        if is_check(body):
            ok = next(check_verdicts) if check_verdicts else True
            return fake_response(json.dumps({"rationale": "r", "same_request": ok,
                                             "core_request_in_final_turn": True}))
        if rewriter_refuses:
            return fake_response("I'm sorry, but I can't help with rewriting that.")
        if "three user messages" in text:
            return fake_response(json.dumps({"turns": ["ctx one", "ctx two", "the request"]}))
        version = "v2" if "previous rewrite was rejected" in text else "v1"
        return fake_response(json.dumps({"prompt": f"Rewritten ({version}): the request"}))
    return respond


def test_five_framings_per_intent(cfg):
    from srs.llm.fake import make_fake_client

    client = make_fake_client(cfg, responder=responder())
    directs = _directs()
    res = asyncio.run(generate_framings(directs, cfg, client))
    assert res.dropped == []
    allp = directs + res.framed
    per_intent = Counter(p.intent_id for p in allp)
    assert set(per_intent.values()) == {5}
    for p in res.framed:
        d = next(x for x in directs if x.intent_id == p.intent_id)
        assert (p.category, p.expected_behavior, p.split) == (d.category, d.expected_behavior,
                                                              "rq3")
        assert p.prompt_id != d.prompt_id and p.template_version
        if p.framing == "multiturn":
            assert len(p.turns) == 3
        else:
            assert len(p.turns) == 1
    rewriter = cfg.models[cfg.framings.rewriter_model].model
    checker = cfg.models[cfg.framings.checker_model].model
    models = {(is_check(b), b["model"]) for _, b in client.transport.calls}
    assert models == {(False, rewriter), (True, checker)}


def test_regenerate_then_pass(cfg):
    from srs.llm.fake import make_fake_client

    client = make_fake_client(cfg, responder=responder(iter([False, True])))
    out = asyncio.run(frame_intent(_directs(1)[0], "academic", cfg, client))
    assert out.prompt is not None
    assert [a["mode"] for a in out.attempts] == ["llm", "llm"]
    assert out.attempts[0]["check"]["passed"] is False  # failure recorded
    second_rewrite = [b for _, b in client.transport.calls if not is_check(b)][1]
    assert "previous rewrite was rejected" in last_user(second_rewrite)


def test_persistent_failure_drops_intent(cfg):
    from srs.llm.fake import make_fake_client

    def respond(provider, body):
        text = last_user(body)
        if is_check(body):
            ok = "roleplay framing" not in text
            return fake_response(json.dumps({"rationale": "r", "same_request": ok,
                                             "core_request_in_final_turn": True}))
        return responder()(provider, body)

    client = make_fake_client(cfg, responder=respond)
    directs = _directs(2)
    res = asyncio.run(generate_framings(directs, cfg, client))
    assert len(res.dropped) == 2 and res.framed == []
    assert all(d["framing"] == "roleplay" for d in res.dropped)
    rp = next(a for a in res.attempts if a["framing"] == "roleplay")["attempts"]
    assert [a["mode"] for a in rp] == ["llm", "llm", "template"]


def test_rewriter_refusal_falls_back_to_template(cfg):
    from srs.llm.fake import make_fake_client

    client = make_fake_client(cfg, responder=responder(rewriter_refuses=True))
    out = asyncio.run(frame_intent(_directs(1)[0], "multiturn", cfg, client))
    assert out.prompt is not None and len(out.prompt.turns) == 3
    assert [a["mode"] for a in out.attempts] == ["llm", "template"]
    assert out.attempts[0]["rewrite_status"] == "refused"
    assert "sourdough" in out.prompt.turns[2] and "sourdough" not in out.prompt.turns[0]
