"""One-time migration (2026-10-20) of records written before two fixes:

1. Block detection: Gemini reports `finish_reason` with a detail suffix
   ("content_filter: PROHIBITED_CONTENT", "content_filter: RECITATION"), which the old exact
   match missed, so those responses were stored as `ok` with an empty answer. They become
   `api_blocked`, exactly as `runner.run_job` now records them.
2. `no_answer`: `ok` responses with an empty visible answer (mostly thinking variants that
   ran out of tokens) become `no_answer`. Their old judgments stay on disk; analysis ignores
   judgments of non-`ok` responses.

It also reparses WildGuard `parse_error` judgments from their stored raw output with the
current parser (trailing text after the three labels is now ignored); nothing is regenerated.

Back up `data/processed/` first. Dry run by default:

    uv run python scripts/migrate_response_status.py           # report only
    uv run python scripts/migrate_response_status.py --apply
"""

import sys
from collections import Counter

from srs.config import SafetyModelEvaluatorConfig, load_config
from srs.evaluators.base import judgment_key, judgments_path
from srs.evaluators.safety_model import parse_output
from srs.generation.runner import responses_path
from srs.io import read_records, upsert_records
from srs.llm.errors import is_blocked_finish
from srs.schemas import Judgment, Response, utcnow


def migrate_responses(cfg, responses: list[Response]) -> tuple[list[Response], Counter]:
    changed, counts = [], Counter()
    for r in responses:
        if r.status != "ok":
            continue
        provider = cfg.providers[cfg.models[r.model_id].provider] if r.model_id in cfg.models \
            else None
        if provider is not None and is_blocked_finish(provider, r.finish_reason):
            transcript = list(r.transcript)
            if transcript and transcript[-1].role == "assistant":
                transcript.pop()  # a blocked turn has no assistant reply
            changed.append(r.model_copy(update={
                "status": "api_blocked", "transcript": transcript, "final_text": None,
                "reasoning_text": None, "error": r.finish_reason,
                "block_turn": sum(m.role == "user" for m in transcript)}))
            counts[f"{r.model_id} -> api_blocked"] += 1
        elif not (r.final_text or "").strip():
            changed.append(r.model_copy(update={"status": "no_answer"}))
            counts[f"{r.model_id} -> no_answer"] += 1
    return changed, counts


def reparse_wildguard(cfg, judgments: list[Judgment]) -> list[Judgment]:
    ids = {k for k, e in cfg.evaluators.items() if isinstance(e, SafetyModelEvaluatorConfig)}
    out = []
    for j in judgments:
        if j.evaluator_id not in ids or j.status != "parse_error":
            continue
        parsed = parse_output(j.raw_output or "")
        if parsed is None:
            continue
        _, refusal, harmful = parsed
        out.append(j.model_copy(update={
            "status": "ok", "refusal": "full_refusal" if refusal else "full_compliance",
            "harmful": harmful, "created_at": utcnow()}))
    return out


def main(apply: bool) -> None:
    cfg = load_config()
    rpath, jpath = responses_path(cfg), judgments_path(cfg, "responses")
    changed, counts = migrate_responses(cfg, read_records(rpath, Response))
    reparsed = reparse_wildguard(cfg, read_records(jpath, Judgment))
    for k, v in sorted(counts.items()):
        print(f"{k:28} {v}")
    print(f"responses changed: {len(changed)}; wildguard parse_error -> ok: {len(reparsed)}")
    if not apply:
        print("dry run; pass --apply to write")
        return
    upsert_records(rpath, Response, changed, key=lambda r: r.response_id)
    upsert_records(jpath, Judgment, reparsed, key=judgment_key)
    print("written")


if __name__ == "__main__":
    main("--apply" in sys.argv[1:])
