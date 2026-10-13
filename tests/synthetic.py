"""A small synthetic study written to a config's data paths, for analysis tests."""

import numpy as np
import pandas as pd

from srs.data.build import processed_dir, prompts_path
from srs.data.sources._common import make_prompt
from srs.evaluators.base import judgments_path
from srs.generation.runner import responses_path
from srs.ids import prompt_id
from srs.io import write_records
from srs.reference.sampling import annotation_dir
from srs.schemas import HumanLabel, Judgment, Message, ReferenceItem, Response

CATS = ["violence_physical_harm", "cybercrime", "privacy", "sexual"]
FRAMES = ["direct", "academic", "roleplay", "hypothetical", "multiturn"]
EVALS = ["keyword", "classic", "wildguard", "judge_a", "judge_b"]
TS = "2026-10-01T00:00:00+00:00"


def build(cfg, n_main=80, n_rq3=24, seed=0):
    rng = np.random.default_rng(seed)
    models = cfg.evaluated
    prompts = []
    for i in range(n_main + n_rq3):
        split = "main" if i < n_main else "rq3"
        exp = "refuse" if i % 2 else "comply"
        base = make_prompt("syn", str(i), f"synthetic request {i}", exp, None).model_copy(
            update={"category": CATS[(i // 2) % 4], "split": split,
                    "tier": "borderline" if i % 13 == 0 and split == "main" else "clear"})
        prompts.append(base)
        if split == "rq3":
            for fr in FRAMES[1:]:
                turns = [f"{fr} t{k}" for k in range(3)] if fr == "multiturn" else [f"{fr} {i}"]
                prompts.append(base.model_copy(update={
                    "framing": fr, "turns": turns,
                    "prompt_id": prompt_id("syn", str(i), fr, turns)}))
    write_records(prompts_path(cfg), prompts)

    responses, judgments, truth = [], [], {}
    for mi, m in enumerate(models):
        for p in prompts:
            rid = f"{m}:{p.prompt_id}"
            blocked = rng.random() < 0.03
            # model mi: FAR grows and ORR shrinks with mi; framings lower refusal a bit
            p_refuse = (0.95 - 0.08 * mi) if p.expected_behavior == "refuse" else 0.05 + 0.03 * mi
            if p.framing != "direct":
                p_refuse -= 0.15
            refused = bool(rng.random() < p_refuse)
            truth[rid] = refused
            transcript = [Message(role="user", content=p.turns[0])]
            if not blocked:
                transcript.append(Message(role="assistant", content="synthetic answer"))
            responses.append(Response(
                response_id=rid, prompt_id=p.prompt_id, model_id=m, gen_params={},
                transcript=transcript, final_text=None if blocked else "synthetic answer",
                status="api_blocked" if blocked else "ok", run_id="syn"))
            if blocked:
                continue
            for ei, ev in enumerate(EVALS):
                noisy = refused if rng.random() > 0.05 + 0.03 * ei else not refused
                status = "ok"
                if ev == "judge_b" and rng.random() < 0.04:
                    status = "blocked"
                judgments.append(Judgment(
                    item_id=rid, evaluator_id=ev, evaluator_version="v1",
                    refusal=("full_refusal" if noisy else "full_compliance")
                    if status == "ok" else None,
                    harmful=None if ev in ("keyword", "classic") or status != "ok"
                    else (not noisy and p.expected_behavior == "refuse"),
                    status=status, created_at=TS))
    write_records(responses_path(cfg), responses)
    write_records(judgments_path(cfg, "responses"), judgments)

    refs, ext = [], []
    for i in range(60):
        src = "xstest_completions" if i % 2 else "wildguardtest"
        ref_ref = bool(rng.random() < 0.5)
        refs.append(ReferenceItem(item_id=f"ref{i}", source=src, source_id=str(i), prompt="p",
                                  response="r",
                                  refusal="full_refusal" if ref_ref else "full_compliance",
                                  harmful=None if src == "xstest_completions" else not ref_ref))
        for ev in EVALS:
            lab = ref_ref if rng.random() > 0.1 else not ref_ref
            ext.append(Judgment(item_id=f"ref{i}", target="external", evaluator_id=ev,
                                evaluator_version="v1",
                                refusal="full_refusal" if lab else "full_compliance",
                                harmful=None if ev in ("keyword", "classic") else not lab,
                                status="ok", created_at=TS))
    write_records(processed_dir(cfg) / "reference_items.parquet", refs)
    write_records(judgments_path(cfg, "external"), ext)

    ok = [r for r in responses if r.status == "ok"
          and next(p for p in prompts if p.prompt_id == r.prompt_id).framing == "direct"]
    idx = rng.choice(len(ok), 200, replace=False)
    d = annotation_dir(cfg, "main")
    d.mkdir(parents=True, exist_ok=True)
    labels, meta = [], []
    for k in sorted(idx):
        r = ok[k]
        labels.append(HumanLabel(item_id=r.response_id, pass_num=0, harmful=False,
                                 refusal="full_refusal" if truth[r.response_id]
                                 else "full_compliance", labeled_at=TS))
        prob = 0.5 if k % 2 else 0.2
        meta.append({"item_id": r.response_id, "inclusion_prob": prob, "weight": 1 / prob})
    write_records(d / "human_labels.parquet", labels)
    pd.DataFrame(meta).to_parquet(d / "sample.parquet")
    return prompts, responses
