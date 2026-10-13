"""Logging that never prints full prompt or response text.

Code should log identifiers and statuses; `preview()` gives a short, single-line excerpt when
some context is needed. As a backstop, a filter truncates any over-long log message.
"""

from __future__ import annotations

import logging

MAX_MESSAGE_CHARS = 300
PREVIEW_CHARS = 40


def preview(text: str | None, n: int = PREVIEW_CHARS) -> str:
    if not text:
        return "<empty>"
    flat = " ".join(text.split())
    return flat if len(flat) <= n else flat[:n] + f"…(+{len(flat) - n} chars)"


class TruncateFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if len(msg) > MAX_MESSAGE_CHARS:
            record.msg = msg[:MAX_MESSAGE_CHARS] + f"…[truncated {len(msg) - MAX_MESSAGE_CHARS}]"
            record.args = ()
        return True


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s",
                                           "%H:%M:%S"))
    handler.addFilter(TruncateFilter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpx2", "openai", "urllib3", "datasets", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
