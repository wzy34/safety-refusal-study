"""Evaluator construction from configuration."""

from __future__ import annotations

from srs.config import (
    ClassifierEvaluatorConfig,
    KeywordEvaluatorConfig,
    LLMJudgeEvaluatorConfig,
    SafetyModelEvaluatorConfig,
    StudyConfig,
)


def judge_models(cfg: StudyConfig, evaluator_ids: list[str]) -> list[str]:
    return sorted({e.model for i in evaluator_ids
                   if isinstance(e := cfg.evaluators[i], LLMJudgeEvaluatorConfig)})


def make_evaluator(cfg: StudyConfig, evaluator_id: str, client=None):
    ecfg = cfg.evaluators[evaluator_id]
    if isinstance(ecfg, KeywordEvaluatorConfig):
        from srs.evaluators.keyword import KeywordEvaluator

        return KeywordEvaluator(evaluator_id, ecfg.phrase_list)
    if isinstance(ecfg, ClassifierEvaluatorConfig):
        from srs.evaluators.classic import ClassifierEvaluator

        return ClassifierEvaluator(evaluator_id, ecfg, cfg.root / ecfg.checkpoint_dir)
    if isinstance(ecfg, SafetyModelEvaluatorConfig):
        from srs.evaluators.safety_model import SafetyModelEvaluator

        return SafetyModelEvaluator(evaluator_id, ecfg, cfg.root)
    if isinstance(ecfg, LLMJudgeEvaluatorConfig):
        from srs.evaluators.llm_judge import LLMJudgeEvaluator

        if client is None:
            raise ValueError(f"evaluator {evaluator_id} needs a chat client")
        return LLMJudgeEvaluator(evaluator_id, ecfg, cfg, client)
    raise TypeError(f"unknown evaluator kind for {evaluator_id}")
