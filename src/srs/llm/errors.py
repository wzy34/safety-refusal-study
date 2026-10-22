"""Provider error classification (design D3): `retry`, `blocked` or `fatal`.

Block detection is data-driven from providers.yaml (error codes, message patterns, finish
reasons), so codes discovered in the pilot are added without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from srs.config import ProviderConfig

ErrorKind = Literal["retry", "blocked", "fatal"]


@dataclass(eq=False)
class ProviderError(Exception):
    """A provider failure normalized from SDK exceptions (or raised by the fake transport)."""

    status_code: int | None = None
    code: str | None = None
    type: str | None = None
    message: str = ""
    body: Any = None
    is_timeout: bool = False
    is_connection: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (f"ProviderError(status={self.status_code}, code={self.code}, type={self.type}, "
                f"message={self.message[:200]!r})")

    def to_payload(self) -> dict[str, Any]:
        return {"status_code": self.status_code, "code": self.code, "type": self.type,
                "message": self.message, "body": self.body}


def from_openai_exception(exc: Exception) -> ProviderError:
    import openai

    if isinstance(exc, openai.APITimeoutError):
        return ProviderError(message=str(exc), is_timeout=True)
    if isinstance(exc, openai.APIConnectionError):
        return ProviderError(message=str(exc), is_connection=True)
    if isinstance(exc, openai.APIStatusError):
        body = exc.body
        err = body.get("error", body) if isinstance(body, dict) else {}
        err = err if isinstance(err, dict) else {}
        code = err.get("code", getattr(exc, "code", None))
        return ProviderError(
            status_code=exc.status_code,
            code=None if code is None else str(code),
            type=err.get("type"),
            message=str(err.get("message") or exc.message),
            body=body,
        )
    return ProviderError(message=f"{type(exc).__name__}: {exc}")


def from_anthropic_exception(exc: Exception) -> ProviderError:
    import anthropic

    # APITimeoutError subclasses APIConnectionError, so it is checked first.
    if isinstance(exc, anthropic.APITimeoutError):
        return ProviderError(message=str(exc), is_timeout=True)
    if isinstance(exc, anthropic.APIConnectionError):
        return ProviderError(message=str(exc), is_connection=True)
    if isinstance(exc, anthropic.APIStatusError):
        body = exc.body
        err = body.get("error", body) if isinstance(body, dict) else {}
        err = err if isinstance(err, dict) else {}
        etype = err.get("type")  # e.g. rate_limit_error, overloaded_error, invalid_request_error
        return ProviderError(status_code=exc.status_code, code=etype, type=etype,
                             message=str(err.get("message") or exc.message), body=body)
    return ProviderError(message=f"{type(exc).__name__}: {exc}")


def classify_error(provider: ProviderConfig, err: ProviderError) -> ErrorKind:
    if is_block(provider, err):
        return "blocked"
    if err.is_timeout or err.is_connection:
        return "retry"
    if err.status_code is not None and err.status_code in provider.retry_status_codes:
        return "retry"
    return "fatal"


def is_block(provider: ProviderConfig, err: ProviderError) -> bool:
    if err.status_code is not None and err.status_code in provider.blocked_status_codes:
        return True
    codes = {c for c in (err.code, err.type) if c}
    if codes & set(provider.blocked_error_codes):
        return True
    msg = err.message.lower()
    return any(p.lower() in msg for p in provider.blocked_message_patterns)


def block_code(err: ProviderError) -> str:
    return str(err.code or err.type or err.status_code or "blocked")


def is_blocked_finish(provider: ProviderConfig, finish_reason: str | None) -> bool:
    """Match on the part before any ':' detail, e.g. Gemini's
    "content_filter: PROHIBITED_CONTENT" or "content_filter: RECITATION"."""
    if finish_reason is None:
        return False
    return finish_reason.split(":", 1)[0].strip() in provider.blocked_finish_reasons
