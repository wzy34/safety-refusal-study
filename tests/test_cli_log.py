import logging

from typer.testing import CliRunner

from srs.cli import app
from srs.log import MAX_MESSAGE_CHARS, TruncateFilter, preview


def test_help_lists_every_group():
    out = CliRunner().invoke(app, ["--help"]).output
    for group in ("data", "generate", "classic", "evaluate", "annotate", "analyze", "llm"):
        assert group in out


def test_invalid_config_exits_nonzero(tmp_path):
    res = CliRunner().invoke(app, ["--config-dir", str(tmp_path), "data", "build"])
    assert res.exit_code == 2


def test_preview_truncates():
    text = "word " * 500
    p = preview(text)
    assert len(p) < 80 and "chars" in p
    assert preview(None) == "<empty>"


def test_long_log_message_truncated():
    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "resp %s", ("y" * 5000,), None)
    TruncateFilter().filter(rec)
    assert len(rec.getMessage()) < MAX_MESSAGE_CHARS + 40
