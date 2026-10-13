"""LLM rubric judges (Judge A: Gemini, Judge B: DeepSeek by default). Both use the same versioned
rubric, greedy decoding where supported, and JSON output with the rationale first.

Status mapping: valid JSON -> ok; declined to judge -> judge_refused; provider block ->
blocked; still invalid after retries -> parse_error. Transient API errors produce no
judgment, so the item is retried on the next run.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import BaseModel

from srs.config import LLMJudgeEvaluatorConfig, StudyConfig
from srs.evaluators.base import EvalItem
from srs.ids import short_hash
from srs.llm.adapters import GenParams
from srs.llm.client import ChatClient
from srs.llm.structured import structured_call
from srs.schemas import Judgment
from srs.templates import load_template

STATUS = {"ok": "ok", "refused": "judge_refused", "blocked": "blocked",
          "parse_error": "parse_error"}


class JudgeVerdict(BaseModel):
    rationale: str
    refusal: Literal["full_compliance", "partial", "full_refusal"]
    harmful: bool


class LLMJudgeEvaluator:
    def __init__(self, evaluator_id: str, ecfg: LLMJudgeEvaluatorConfig, cfg: StudyConfig,
                 client: ChatClient):
        self.id = evaluator_id
        self.ecfg = ecfg
        self.cfg = cfg
        self.client = client
        self.rubric = load_template(cfg.path("prompts_dir"), ecfg.rubric)

    def version(self) -> str:
        m = self.cfg.models[self.ecfg.model]
        return "judge-" + short_hash({
            "rubric": self.rubric.version, "provider": m.provider, "model": m.model,
            "structured_output": m.structured_output, "retries": self.ecfg.max_parse_retries,
            "max_tokens": self.ecfg.max_tokens, "seed": self.cfg.experiment.seed,
        })[:12]

    def build_messages(self, item: EvalItem) -> list[dict[str, str]]:
        return [{"role": "user", "content": self.rubric.render(conversation=item.render())}]

    async def _judge(self, item: EvalItem, version: str) -> Judgment | None:
        params = GenParams(max_tokens=self.ecfg.max_tokens, temperature=0.0,
                           seed=self.cfg.experiment.seed)
        res = await structured_call(self.client, self.ecfg.model, self.build_messages(item),
                                    JudgeVerdict, params, self.ecfg.max_parse_retries)
        if res.status == "error":
            return None
        v = res.value
        return Judgment(item_id=item.item_id, evaluator_id=self.id, evaluator_version=version,
                        refusal=v.refusal if v else None, harmful=v.harmful if v else None,
                        status=STATUS[res.status],  # type: ignore[arg-type]
                        raw_output=res.raw_text)

    def evaluate(self, items: list[EvalItem]) -> list[Judgment]:
        version = self.version()

        async def run():
            return await asyncio.gather(*(self._judge(it, version) for it in items))

        return [j for j in asyncio.run(run()) if j is not None]
