"""End-to-end dataset build and framing on fixture raw data with the fake client."""

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from srs.data.build import build_dataset, load_prompts, processed_dir
from srs.data.framing import frame_rq3
from srs.data.sources import FetchError
from srs.llm.fake import fake_response, last_user, make_fake_client

RAW = Path(__file__).parent / "fixtures" / "raw"


def respond(provider, body):
    text = last_user(body)
    if "classify the topic" in text:
        return fake_response(json.dumps({"rationale": "r", "category": "privacy"}))
    if "clear or contested" in text:
        return fake_response(json.dumps({"rationale": "r", "borderline": "phishing" in text}))
    if "Rephrased version" in text:
        return fake_response(json.dumps({"rationale": "r", "same_request": True,
                                         "core_request_in_final_turn": True}))
    if "three user messages" in text:
        return fake_response(json.dumps({"turns": ["a", "b", "c"]}))
    return fake_response(json.dumps({"prompt": "rewritten"}))


@pytest.fixture
def small_cfg(cfg):
    cfg.experiment.splits.pilot_intents = 4
    cfg.experiment.splits.rq3_intents = 4
    return cfg


def test_build_then_frame(small_cfg, tmp_path):
    cfg = small_cfg
    client = make_fake_client(cfg, responder=respond)
    res = asyncio.run(build_dataset(cfg, client, RAW))
    report = json.loads((processed_dir(cfg) / "build_report.json").read_text())
    for key in ("by_source", "by_split", "by_expected_behavior", "by_category", "by_tier",
                "by_framing", "dedup_removed", "dropped_intents"):
        assert key in report
    assert report["by_tier"]["borderline"] >= 2  # sorry_bench "40" rule + phishing screen
    assert report["by_framing"] == {"direct": len(res.prompts)}
    assert sum(report["by_split"].values()) == len(res.prompts)
    assert report["llm_assigned_categories"] == 5  # the xstest fixture prompts
    assert (processed_dir(cfg) / "build_report.md").exists()

    again = asyncio.run(build_dataset(cfg, client, RAW))
    assert [(p.prompt_id, p.split) for p in again.prompts] == [
        (p.prompt_id, p.split) for p in res.prompts]

    fr = asyncio.run(frame_rq3(cfg, client))
    prompts = load_prompts(cfg)
    rq3_intents = {p.intent_id for p in prompts if p.split == "rq3" and p.framing == "direct"}
    assert len(rq3_intents) == 4 and len(fr.framed) == 16
    report = json.loads((processed_dir(cfg) / "build_report.json").read_text())
    assert report["by_framing"]["multiturn"] == 4


def test_missing_gated_source_names_access_step(cfg, tmp_path):
    raw = tmp_path / "raw"
    shutil.copytree(RAW, raw)
    shutil.rmtree(raw / "sorry_bench")
    with pytest.raises(FetchError, match="sorry_bench.*huggingface.co/datasets/sorry-bench"):
        asyncio.run(build_dataset(cfg, make_fake_client(cfg, responder=respond), raw))
