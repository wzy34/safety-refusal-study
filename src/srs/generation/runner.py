"""Response generation: prompts x models x samples -> `Response` records.

- Primary responses use sample index 0 and the configured deterministic settings.
- The stability subset adds samples 1..n at a non-zero temperature for a fixed set of intents.
- Multi-turn prompts are run live: each scripted user turn is sent after the model's own reply
  to the previous one. A provider block stops the conversation and records the turn.
- A reply with no visible answer (e.g. the thinking budget ran out first) is `no_answer`:
  kept on disk but neither judged nor counted in refusal rates.
- Responses already `ok`, `api_blocked` or `no_answer` are skipped; `error` responses are
  retried.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from srs import ids
from srs.config import StudyConfig
from srs.data.build import balanced_pick, processed_dir
from srs.io import read_records, upsert_records
from srs.llm.adapters import GenParams
from srs.llm.client import ChatClient
from srs.schemas import Message, Prompt, Response

log = logging.getLogger(__name__)

FINAL_STATUSES = ("ok", "api_blocked", "no_answer")
CHUNK = 200  # responses are checkpointed to disk after every CHUNK completions


def responses_path(cfg: StudyConfig) -> Path:
    return processed_dir(cfg) / "responses.parquet"


@dataclass(frozen=True)
class Job:
    prompt: Prompt
    model_id: str
    sample_idx: int
    gen_params: dict[str, Any]  # configured parameters (hashed into the response id)

    @property
    def response_id(self) -> str:
        return ids.response_id(self.prompt.prompt_id, self.model_id, self.sample_idx,
                               ids.gen_params_hash(self.gen_params))

    @property
    def params(self) -> GenParams:
        return GenParams(max_tokens=self.gen_params["max_tokens"],
                         temperature=self.gen_params["temperature"],
                         seed=self.gen_params["seed"])


def primary_params(cfg: StudyConfig, model_id: str | None = None) -> dict[str, Any]:
    g = cfg.experiment.generation
    max_tokens = g.max_tokens
    if model_id is not None and cfg.models[model_id].is_reasoning and g.reasoning_max_tokens:
        max_tokens = g.reasoning_max_tokens
    return {"temperature": g.temperature, "max_tokens": max_tokens, "seed": g.seed,
            "system_prompt": g.system_prompt}


def stability_params(cfg: StudyConfig, model_id: str | None = None) -> dict[str, Any]:
    return {**primary_params(cfg, model_id), "temperature": cfg.experiment.stability.temperature}


def select_split(prompts: list[Prompt], split: str) -> list[Prompt]:
    """Prompts of one split. RQ3 includes all framings; other splits are direct only."""
    return [p for p in prompts if p.split == split and (split == "rq3" or p.framing == "direct")]


def stability_prompts(cfg: StudyConfig, prompts: list[Prompt]) -> list[Prompt]:
    s = cfg.experiment.stability
    pool = [p for p in select_split(prompts, s.split) if p.framing == "direct"
            and p.tier == "clear"]
    return balanced_pick(pool, s.n_intents, cfg.experiment.seed, "stability")


def plan_jobs(cfg: StudyConfig, prompts: list[Prompt], model_ids: list[str], *,
              stability: bool = False, limit: int | None = None) -> list[Job]:
    prompts = prompts[:limit] if limit is not None else prompts
    if not stability:
        return [Job(p, m, 0, primary_params(cfg, m)) for m in model_ids for p in prompts]
    n = cfg.experiment.stability.n_samples
    return [Job(p, m, k, stability_params(cfg, m)) for m in model_ids for p in prompts
            for k in range(1, n + 1)]


def _system(job: Job) -> list[dict[str, str]]:
    sp = job.gen_params.get("system_prompt")
    return [{"role": "system", "content": sp}] if sp else []


async def run_job(client: ChatClient, job: Job, run_id: str) -> Response:
    messages = _system(job)
    transcript = [Message(role="system", content=m["content"]) for m in messages]
    usage: dict[str, int] = defaultdict(int)
    latency = 0.0
    salt = str(job.sample_idx) if job.sample_idx > 0 else None
    comp = None
    status, block_turn, error, dropped = "ok", None, None, []
    for turn_idx, user_turn in enumerate(job.prompt.turns, start=1):
        messages = [*messages, {"role": "user", "content": user_turn}]
        transcript.append(Message(role="user", content=user_turn))
        comp = await client.complete(job.model_id, messages, job.params, cache_salt=salt)
        dropped = comp.dropped_params
        latency += comp.latency_s or 0.0
        for k, v in (comp.usage or {}).items():
            if isinstance(v, int):
                usage[k] += v
        if comp.status != "ok":
            status, error = comp.status, comp.error or comp.block_code
            block_turn = turn_idx if comp.status == "api_blocked" else None
            break
        # only the visible answer goes back into the conversation, never the reasoning
        messages = [*messages, {"role": "assistant", "content": comp.text or ""}]
        transcript.append(Message(role="assistant", content=comp.text or "",
                                  reasoning=comp.reasoning))
    assert comp is not None
    ok = status == "ok"
    if ok and not (comp.text or "").strip():
        status = "no_answer"
    return Response(
        response_id=job.response_id,
        prompt_id=job.prompt.prompt_id,
        model_id=job.model_id,
        model_version=comp.model_version,
        gen_params=job.gen_params,
        sample_idx=job.sample_idx,
        transcript=transcript,
        final_text=comp.text if ok else None,
        reasoning_text=comp.reasoning if ok else None,
        finish_reason=comp.finish_reason,
        status=status,  # type: ignore[arg-type]
        block_turn=block_turn,
        error=error,
        dropped_params=dropped,
        usage=dict(usage),
        latency_s=round(latency, 3),
        run_id=run_id,
    )


def pending_jobs(jobs: list[Job], existing: list[Response]) -> list[Job]:
    done = {r.response_id for r in existing if r.status in FINAL_STATUSES}
    return [j for j in jobs if j.response_id not in done]


async def run_generation(cfg: StudyConfig, client: ChatClient, jobs: list[Job], run_id: str,
                         out_path: Path | None = None) -> dict[str, int]:
    """Run pending jobs, checkpointing to `out_path`. Returns status counts for this run.

    Each provider drains its own queue with `max_concurrency` workers, so providers run in
    parallel and a slow call holds up only its own worker, not a whole chunk.
    """
    out_path = out_path or responses_path(cfg)
    todo = pending_jobs(jobs, read_records(out_path, Response))
    counts: dict[str, int] = defaultdict(int)
    counts["skipped_done"] = len(jobs) - len(todo)
    queues: dict[str, deque[Job]] = defaultdict(deque)
    for j in todo:  # job order (model order) is kept within each provider
        queues[cfg.models[j.model_id].provider].append(j)
    buffer: list[Response] = []
    written = 0

    def flush() -> None:
        nonlocal written
        if buffer:
            upsert_records(out_path, Response, buffer, key=lambda r: r.response_id)
            written += len(buffer)
            buffer.clear()
            left = {p: len(q) for p, q in queues.items() if q}
            log.info("generation %d/%d done (%s) queued=%s", written, len(todo), dict(counts),
                     left)

    async def worker(queue: deque[Job]) -> None:
        while queue:
            r = await run_job(client, queue.popleft(), run_id)
            counts[r.status] += 1
            buffer.append(r)
            if len(buffer) >= CHUNK:
                flush()

    workers = [worker(q) for p, q in queues.items()
               for _ in range(min(cfg.providers[p].max_concurrency, len(q)))]
    try:
        await asyncio.gather(*workers)
    finally:
        flush()
    return dict(counts)


# --- dry run ---------------------------------------------------------------------------------


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def dry_run_estimate(cfg: StudyConfig, jobs: list[Job], existing: list[Response],
                     expected_output_tokens: int = 400) -> list[dict[str, Any]]:
    """Planned calls and estimated tokens per model for pending jobs (no requests made).
    Input tokens ~ chars/4, including earlier turns of multi-turn prompts; output tokens are
    given as an expected value and as the max_tokens upper bound."""
    rows: dict[str, dict[str, Any]] = {}
    for job in pending_jobs(jobs, existing):
        r = rows.setdefault(job.model_id, {"model_id": job.model_id, "responses": 0, "calls": 0,
                                           "est_input_tokens": 0, "est_output_tokens": 0,
                                           "max_output_tokens": 0})
        r["responses"] += 1
        history = _est_tokens(job.gen_params.get("system_prompt") or "")
        for turn in job.prompt.turns:
            history += _est_tokens(turn)
            r["calls"] += 1
            r["est_input_tokens"] += history
            r["est_output_tokens"] += expected_output_tokens
            r["max_output_tokens"] += job.gen_params["max_tokens"]
            history += expected_output_tokens
    for r in rows.values():
        m = cfg.models[r["model_id"]]
        if m.price_in_per_mtok is not None and m.price_out_per_mtok is not None:
            r["est_cost_usd"] = round((r["est_input_tokens"] * m.price_in_per_mtok
                                       + r["est_output_tokens"] * m.price_out_per_mtok) / 1e6, 2)
            r["max_cost_usd"] = round((r["est_input_tokens"] * m.price_in_per_mtok
                                       + r["max_output_tokens"] * m.price_out_per_mtok) / 1e6, 2)
    return list(rows.values())
