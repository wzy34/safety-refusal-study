from srs.evaluators.base import EvalItem, Turn
from srs.evaluators.safety_model import SafetyModelEvaluator, parse_output, render_prompt


def _item(i, answer, context=()):
    return EvalItem(f"r{i}", (*context, Turn("user", f"question {i}"), Turn("assistant", answer)))


def fake_wildguard(prompt: str) -> str:
    answer = prompt.split("AI assistant:\n", 1)[1]
    if "GARBLED" in answer:
        return "Harmful request: maybe"
    refusal = "yes" if "sorry" in answer else "no"
    harmful = "yes" if "[harmful]" in answer else "no"
    return f"Harmful request: yes\nResponse refusal: {refusal}\nHarmful response: {harmful}"


def test_template_matches_official_format():
    text = render_prompt(_item(1, "Sure."))
    assert text.startswith("<s><|user|>\n[INST] You are a classifier")
    assert "Human user:\nquestion 1\n\nAI assistant:\nSure.\n\n---\n\nAnswers: [/INST]\n" in text
    assert text.endswith("<|assistant|>\n")


def test_multiturn_context_rendered():
    text = render_prompt(_item(1, "final", (Turn("user", "u1"), Turn("assistant", "a1"))))
    assert "Human user:\nUser: u1\nAssistant: a1\nUser: question 1\n" in text


def test_parse_strict():
    assert parse_output("Harmful request: yes\nResponse refusal: no\nHarmful response: no") == (
        True, False, False)
    assert parse_output(" Harmful request: no\nResponse refusal: yes\nHarmful response: no\n")
    assert parse_output("Harmful request: yes\nResponse refusal: no") is None
    assert parse_output("Sure! Harmful request: yes\nResponse refusal: no\nHarmful response: no"
                        ) is None


def test_twenty_items_local(cfg, tmp_path):
    answers = ["sorry, I can't", "Sure, [harmful] details", "Here you go", "GARBLED"] * 5
    items = [_item(i, a) for i, a in enumerate(answers)]
    ev = SafetyModelEvaluator("wildguard", cfg.evaluators["wildguard"], tmp_path,
                              generate_fn=fake_wildguard)
    js = ev.evaluate(items)  # the autouse fixture fails any network access
    assert len(js) == 20
    ok = [j for j in js if j.status == "ok"]
    assert len(ok) == 15
    assert {(j.refusal, j.harmful) for j in ok} == {("full_refusal", False),
                                                    ("full_compliance", True),
                                                    ("full_compliance", False)}
    bad = [j for j in js if j.status == "parse_error"]
    assert len(bad) == 5 and all(j.raw_output == "Harmful request: maybe" and j.refusal is None
                                 for j in bad)
