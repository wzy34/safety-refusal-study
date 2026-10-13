import asyncio
import json
from collections import Counter

from typer.testing import CliRunner

from srs import cli
from srs.data.build import prompts_path
from srs.data.sources._common import make_prompt
from srs.generation import runner
from srs.ids import prompt_id
from srs.io import read_records, write_records
from srs.llm.fake import (
    block_error,
    echo_responder,
    fake_response,
    last_user,
    make_fake_client,
    server_error,
)
from srs.schemas import Response

MODELS = ["gemini-a", "deepseek-a", "kimi-a", "glm-a"]


def _prompts(n=6, split="pilot"):
    out = []
    for i in range(n):
        p = make_prompt("xstest", str(i), f"Question {i}?", "comply" if i % 2 else "refuse", None)
        out.append(p.model_copy(update={"category": "privacy", "split": split}))
    return out


def _multiturn(split="rq3"):
    d = _prompts(1, split)[0]
    turns = ["context one", "context two", "the request"]
    return d.model_copy(update={"framing": "multiturn", "turns": turns,
                                "prompt_id": prompt_id(d.source, d.source_id, "multiturn", turns)})


def _run(cfg, client, jobs, path):
    return asyncio.run(runner.run_generation(cfg, client, jobs, "run-1", path))


def test_coverage_and_idempotent_rerun(cfg, tmp_path):
    path = tmp_path / "responses.parquet"
    jobs = runner.plan_jobs(cfg, _prompts(), MODELS)
    client = make_fake_client(cfg)
    counts = _run(cfg, client, jobs, path)
    rs = read_records(path, Response)
    assert counts["ok"] == 24 and len(rs) == 24
    assert Counter((r.prompt_id, r.model_id) for r in rs).most_common(1)[0][1] == 1
    assert all(r.run_id == "run-1" and r.final_text and r.gen_params["max_tokens"] == 1024
               for r in rs)
    n_calls = len(client.transport.calls)
    again = _run(cfg, client, runner.plan_jobs(cfg, _prompts(), MODELS), path)
    assert again["skipped_done"] == 24 and len(client.transport.calls) == n_calls
    assert {r.response_id for r in read_records(path, Response)} == {r.response_id for r in rs}


def test_error_responses_retried_next_run(cfg, tmp_path):
    path = tmp_path / "r.parquet"
    jobs = runner.plan_jobs(cfg, _prompts(1), ["gemini-a"])
    n = cfg.providers["google"].max_retries + 1
    first = _run(cfg, make_fake_client(cfg, queue=[server_error()] * n), jobs, path)
    assert first == {"skipped_done": 0, "error": 1}
    second = _run(cfg, make_fake_client(cfg), jobs, path)
    assert second == {"skipped_done": 0, "ok": 1}
    assert [r.status for r in read_records(path, Response)] == ["ok"]


def test_reasoning_kept_separately(cfg, tmp_path):
    client = make_fake_client(cfg, queue=[fake_response("Final.", reasoning="thinking...")])
    r = asyncio.run(runner.run_job(client, runner.plan_jobs(cfg, _prompts(1), ["deepseek-b"])[0],
                                   "run"))
    assert r.final_text == "Final." and r.reasoning_text == "thinking..."
    assert "thinking" not in r.final_text


def test_multiturn_live_transcript(cfg):
    client = make_fake_client(cfg)
    job = runner.plan_jobs(cfg, [_multiturn()], ["kimi-a"])[0]
    r = asyncio.run(runner.run_job(client, job, "run"))
    assert [m.role for m in r.transcript] == ["user", "assistant"] * 3
    assert r.status == "ok" and r.final_text == "Fake reply to: the request"
    third_request = client.transport.calls[2][1]["messages"]
    assert [m["role"] for m in third_request] == ["user", "assistant", "user", "assistant", "user"]
    assert third_request[1]["content"] == "Fake reply to: context one"  # model's own reply
    assert r.usage["completion_tokens"] == 60


def test_multiturn_block_at_turn_two(cfg):
    def respond(provider, body):
        return block_error() if last_user(body) == "context two" else echo_responder(provider,
                                                                                     body)
    client = make_fake_client(cfg, responder=respond)
    job = runner.plan_jobs(cfg, [_multiturn()], ["glm-a"])[0]
    r = asyncio.run(runner.run_job(client, job, "run"))
    assert r.status == "api_blocked" and r.block_turn == 2 and r.final_text is None
    assert [m.content for m in r.transcript] == ["context one", "Fake reply to: context one",
                                                 "context two"]
    assert len(client.transport.calls) == 2


def test_stability_subset(cfg, tmp_path):
    cfg.experiment.stability.n_intents = 4
    cfg.experiment.stability.split = "main"
    prompts = _prompts(10, split="main")
    subset = runner.stability_prompts(cfg, prompts)
    assert len(subset) == 4
    jobs = runner.plan_jobs(cfg, subset, ["gemini-a", "kimi-a"], stability=True)
    client = make_fake_client(cfg)
    path = tmp_path / "r.parquet"
    _run(cfg, client, jobs, path)
    rs = read_records(path, Response)
    assert len(rs) == 4 * 2 * 5 and {r.sample_idx for r in rs} == {1, 2, 3, 4, 5}
    assert all(r.gen_params["temperature"] == 0.7 for r in rs)
    assert len(client.transport.calls) == 40  # distinct calls despite identical requests
    primary = runner.plan_jobs(cfg, subset, ["gemini-a"])
    assert {j.response_id for j in primary}.isdisjoint({r.response_id for r in rs})


def test_dry_run_and_limit(cfg):
    prompts = _prompts(20) + [_multiturn("pilot")]
    jobs = runner.plan_jobs(cfg, prompts, MODELS)
    rows = runner.dry_run_estimate(cfg, jobs, [])
    assert {r["model_id"] for r in rows} == set(MODELS)
    assert all(r["responses"] == 21 and r["calls"] == 23 for r in rows)
    limited = runner.plan_jobs(cfg, prompts, MODELS, limit=10)
    assert Counter(j.model_id for j in limited) == {m: 10 for m in MODELS}


def test_cli_generate_end_to_end(cfg, monkeypatch):
    write_records(prompts_path(cfg), _prompts(5) + [_multiturn("pilot")])
    client = make_fake_client(cfg)
    monkeypatch.setattr(cli, "_real_client", lambda c, ids: client)
    monkeypatch.setitem(cli._state, "cfg", cfg)
    res = CliRunner().invoke(cli.app, ["generate", "--set", "pilot", "--dry-run"])
    assert res.exit_code == 0 and "dry run" in res.output and client.transport.calls == []

    res = CliRunner().invoke(cli.app, ["generate", "--set", "pilot", "--models", "gemini-a",
                                       "--models", "glm-b", "--limit", "4"])
    assert res.exit_code == 0, res.output
    rs = read_records(runner.responses_path(cfg), Response)
    assert len(rs) == 8
    manifests = list((cfg.path("results_dir") / "manifests").glob("*.json"))
    assert len(manifests) == 1
    m = json.loads(manifests[0].read_text())
    assert {r.run_id for r in rs} == {m["run_id"]} and m["counts"]["ok"] == 8
    assert m["args"]["models"] == ["gemini-a", "glm-b"]
