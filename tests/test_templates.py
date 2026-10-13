import pytest

from srs.templates import TemplateError, load_template, render_string


def test_edit_changes_version(tmp_path):
    p = tmp_path / "judge_rubric.md"
    p.write_text("Rubric v1: {{conversation}}")
    v1 = load_template(tmp_path, "judge_rubric.md").version
    assert load_template(tmp_path, "judge_rubric.md").version == v1
    p.write_text("Rubric v1: {{conversation}}.")
    assert load_template(tmp_path, "judge_rubric.md").version != v1


def test_render_and_missing_variable(tmp_path):
    (tmp_path / "t.md").write_text('Return JSON {"a": 1} for {{ request }}')
    t = load_template(tmp_path, "t.md")
    assert t.render(request="X") == 'Return JSON {"a": 1} for X'
    with pytest.raises(TemplateError, match="request"):
        t.render()


def test_render_string():
    assert render_string("a {{x}} b", x="1") == "a 1 b"
