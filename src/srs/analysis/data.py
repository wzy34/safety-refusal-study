"""Load every analysis input once: responses (with prompt metadata), judgments on study
responses and on external reference items, reference items, and self-annotation labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from srs.analysis.metrics import judgments_frame, responses_frame
from srs.config import StudyConfig
from srs.data.build import load_prompts, processed_dir
from srs.evaluators.base import judgments_path
from srs.generation.runner import responses_path
from srs.io import read_records
from srs.reference.sampling import annotation_dir
from srs.schemas import ADJUDICATED_PASS, HumanLabel, Judgment, ReferenceItem, Response


@dataclass
class AnalysisData:
    resp: pd.DataFrame
    judg: pd.DataFrame
    judg_ext: pd.DataFrame
    refs: pd.DataFrame
    human: pd.DataFrame  # final self-annotation labels joined with sampling weights
    inputs: list[str] = field(default_factory=list)


def load_analysis_data(cfg: StudyConfig, annotation: str = "main") -> AnalysisData:
    paths = {"prompts": processed_dir(cfg) / "prompts.parquet",
             "responses": responses_path(cfg),
             "judgments": judgments_path(cfg, "responses"),
             "judgments_external": judgments_path(cfg, "external"),
             "reference_items": processed_dir(cfg) / "reference_items.parquet"}
    ann = annotation_dir(cfg, annotation)
    resp = responses_frame(read_records(paths["responses"], Response), load_prompts(cfg), cfg)
    # Responses from models dropped from the study (e.g. Kimi, replaced by Qwen) stay on disk
    # but are left out of every per-model analysis.
    if not resp.empty:
        resp = resp[resp["model_id"].isin(cfg.evaluated)].reset_index(drop=True)
    judg = judgments_frame(read_records(paths["judgments"], Judgment))
    # Only judgments of `ok` responses count; a response relabeled later (e.g. to no_answer)
    # keeps its old judgments on disk, but they must not enter the agreement analyses.
    if not judg.empty and not resp.empty:
        judg = judg[judg["item_id"].isin(resp.loc[resp["status"] == "ok", "response_id"])
                    ].reset_index(drop=True)
    judg_ext = judgments_frame(read_records(paths["judgments_external"], Judgment))
    refs = pd.DataFrame([r.model_dump() for r in read_records(paths["reference_items"],
                                                              ReferenceItem)])
    human = _human(ann)
    inputs = [str(p) for p in [*paths.values(), ann / "human_labels.parquet",
                               ann / "sample.parquet"] if Path(p).exists()]
    return AnalysisData(resp, judg, judg_ext, refs, human, inputs)


def _human(ann: Path) -> pd.DataFrame:
    labels = read_records(ann / "human_labels.parquet", HumanLabel)
    finals = pd.DataFrame([x.model_dump() for x in labels if x.pass_num == ADJUDICATED_PASS])
    sample = ann / "sample.parquet"
    if finals.empty or not sample.exists():
        return pd.DataFrame()
    meta = pd.read_parquet(sample)[["item_id", "inclusion_prob", "weight"]]
    return finals.merge(meta, on="item_id")
