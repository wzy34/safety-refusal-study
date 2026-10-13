## Context

This is a greenfield project. The repository holds only `research_ideas.md` and the OpenSpec scaffolding, and is not yet a git repository. See proposal.md for motivation, and the specs for the requirements.

Constraints that shape the approach:
- **APIs:** Gemini (Google), GLM (Zhipu), Kimi (Moonshot) and DeepSeek only. All four accept OpenAI-style Chat Completions requests (Gemini through its OpenAI-compatible endpoint). Gemini replaced GPT on 2026-10-11.
- **Hardware:** Apple M4 Pro with 24 GB RAM. No CUDA GPU.
- **People:** no external annotators. The researcher labels a sample themselves.
- **Language:** English only.
- **Content:** responses may contain harmful material and must stay out of version control.

## Goals / Non-Goals

**Goals:**
- Every result can be traced to a config snapshot, prompt-template hashes and a code version.
- A crash or rerun never pays twice for the same API call.
- Every stage can be tested offline.
- The four providers differ only in configuration, not in code paths.
- Each RQ analysis reduces to a group-by over long-format tables plus a statistical test.

**Non-Goals:**
- Adaptive jailbreak attacks (GCG, PAIR, Crescendo). The multi-turn framing is scripted and identical across models.
- Non-English prompts.
- A web service or multi-user deployment. The annotation app is local and single-user.
- Releasing raw harmful completions.
- Batch-API submission. It is deferred; it would plug in behind the same client interface if cost requires it.

## Decisions

### D1. Official `openai` SDK with per-provider `base_url`, not LangChain or LiteLLM
**Choice:** one `AsyncOpenAI` instance per provider, wrapped in a thin `ChatClient`.

**Why:** the study's data *is* the raw response. That means exact request control, raw `finish_reason`, `reasoning_content`, `usage` and provider error bodies (needed to detect blocks), and cache keys computed over the exact request.
- LangChain normalizes or hides these fields, adds hidden defaults and retries, and changes often between versions.
- LiteLLM adds its own parameter-mapping layer, which isn't needed when every provider already accepts OpenAI-style requests.
- Native SDKs would mean four separate code paths.

Gemini is reached through its OpenAI-compatible endpoint (`generativelanguage.googleapis.com/v1beta/openai/`), not the `google-genai` SDK, so all providers share one path. Gemini-specific fields (`reasoning_effort`, `extra_body.google`) go in the model's `extra_body`.

Gemini caveats to settle in the pilot: thinking can be fully disabled only on 2.5 Flash-series models (`reasoning_effort: none`), so the non-reasoning variant must be one of them or use `minimal` with that limitation reported; prompt-level safety blocks and seed support on the compat endpoint are undocumented and must be observed; the API's default safety settings are kept, so blocks count as `api_blocked`, not as model refusals.

### D2. Per-model capability adapters declared in config
`models.yaml` declares, for each model:
- `token_param` (`max_tokens` or `max_completion_tokens`);
- `supports_temperature` and `supports_seed`;
- `structured_output` (`json_schema`, `json_object` or `none`);
- `is_reasoning`;
- `family` (provider family, used for the self-preference analysis).

`build_request()` is a pure function of `(ModelConfig, messages, gen_params)`, which makes it easy to unit-test. Any configured parameter that gets dropped is recorded in the completion record (see the spec on temperature not applied).

### D3. Provider error classification as a per-provider hook
`errors.py` maps a provider name and an exception or finish reason to `retry`, `blocked` or `fatal`.
- Initial rules: HTTP 429, 5xx and timeouts → `retry`. `finish_reason == "content_filter"` → `blocked`. Known sensitive-content error codes for each provider → `blocked`. Anything else → `fatal`.
- Exact codes for GLM, Kimi and DeepSeek are confirmed during the pilot (task 6.x) and added as data in `providers.yaml`, not hard-coded.
- Retries use tenacity with exponential backoff and jitter. Concurrency is capped by an `asyncio.Semaphore`, and request rate by a simple token bucket, both per provider.

### D4. SQLite request cache plus Parquet tables
- **Cache:** SQLite stores `request_hash → request JSON, raw response JSON, status, timestamp`. The hash is SHA-256 over canonical JSON of `{provider, model, messages, params}`. SQLite was chosen over JSONL files because it handles safe concurrent writes from one async process, does fast key lookups, and survives crashes. Only `ok` and `api_blocked` results are cached.
- **Data tables:** Parquet files under `data/processed/` and `results/`, in long format: `prompts`, `responses`, `judgments`, `human_labels` and `reference_items`. Parquet was chosen over CSV for typed columns and nested fields (turns, transcripts).
- **IDs:**
  - `prompt_id` = hash(source, source id, framing, turns).
  - `intent_id` = hash(source, source id).
  - `response_id` = hash(prompt_id, model_id, sample_idx, gen_params_hash).

  Deterministic ids are what make reruns idempotent.

### D5. Package layout and CLI

```
src/srs/
  config.py schemas.py io.py manifest.py cli.py
  llm/        client.py adapters.py errors.py cache.py ratelimit.py fake.py
  data/       sources/<benchmark>.py taxonomy.py tiers.py framing.py intent_check.py build.py
  generation/ runner.py
  evaluators/ base.py keyword.py classic.py safety_model.py llm_judge.py
  reference/  external.py sampling.py annotate_app.py annotations.py
  analysis/   metrics.py stats.py agreement.py plots.py rq1.py … rq5.py
configs/      providers.yaml models.yaml evaluators.yaml taxonomy.yaml framings.yaml experiment.yaml
prompts/      judge_rubric.md rewrite_<framing>.md intent_check.md category_assign.md tier_screen.md
```

The CLI uses typer subcommand groups: `data`, `generate`, `classic`, `evaluate`, `annotate`, `analyze`. Configs are parsed into pydantic models. `.env` is loaded with python-dotenv.

**Environment:** uv (`pyproject.toml` + `uv.lock`), Python 3.12. Choosing 3.12 over the system's 3.13 is deliberate: the support status of MLX, torch and statsmodels wheels on 3.13 is uncertain, while 3.12 is safe.

### D6. Evaluator interface
```python
class Evaluator(Protocol):
    id: str
    def version(self) -> str: ...                 # template/model hash
    def evaluate(self, items: list[EvalItem]) -> list[Judgment]: ...
```

`EvalItem` holds only blinded content: user turns, earlier assistant turns for multi-turn prompts, and the final answer. A response id is attached for bookkeeping but never rendered into any evaluator input.

The evaluate stage skips `(response_id, evaluator_id, version)` triples that are already judged.

- **Keyword:** case-insensitive substring match against the Zou et al. prefix list, plus an optional extended list. Two evaluator ids: `keyword` and `keyword_ext`.
- **Classic:** fine-tuned `roberta-base` for binary refusal on the WildGuardMix-train response-refusal labels.
  - Train on the Mac GPU (MPS), with a 10% validation split and early stopping.
  - Max sequence length 512, truncating from the middle so both the start (where refusals usually appear) and the end of the response are kept.
  - Leakage guard: drop training items whose normalized prompt+response hash appears in any reference set.
- **Safety model:** WildGuard converted to MLX 8-bit once (`mlx_lm.convert -q --q-bits 8`) and cached in `models/`. It uses WildGuard's official input template, greedy decoding and a maximum of 32 new tokens. Its three output lines are parsed with a strict regex.
  - The MLX 8-bit version differs from the published fp16 weights, so its accuracy on WildGuardTest is reported as a fidelity check (see risks).
- **LLM judges:** one rubric (`prompts/judge_rubric.md`) with definitions and worked examples of full compliance, partial and full refusal, plus a harmfulness definition.
  - The judge outputs JSON in this order: `{rationale, refusal, harmful}`. Rationale first, so the model reasons before it labels.
  - Judge A (Gemini) uses `json_schema` output. Judge B (DeepSeek) uses `json_object` output plus pydantic validation and up to 2 retries.
  - A judge refusal is detected when no JSON comes back and the text matches refusal patterns → `judge_refused`.

### D7. Framing generation
- **Template-first, LLM-paraphrase second.** `framings.yaml` holds wrappers for each framing (academic, role-play, hypothetical) and three-turn scaffolds for `multiturn`. A rewrite model (configured; default Gemini) adapts each template to the intent so it reads naturally.
- **Intent check.** Another call (default DeepSeek, so the checker is from a different family than the rewriter) answers whether the rewrite asks for the same information. Up to 3 attempts, then the intent is dropped.
- **Unsafe intents.** The rewriter may refuse to rewrite them. That is handled the same way as an intent-check failure: retry with the template only (no paraphrase), so most intents still get all 5 framings.
- **Multi-turn generation** runs live (D1 client): user turn → model reply → next turn. Prefilling assistant turns is out of scope because it is a different kind of attack.

### D8. Analysis statistics
- **Confidence intervals:** Wilson intervals for proportions, and bootstrap (2,000 resamples, seeded) for derived quantities such as J, κ and Kendall's τ.
- **Paired tests:** exact McNemar for paired model and framing comparisons, with Holm correction.
- **Regression:** a logistic GLM with standard errors clustered by `intent_id` (statsmodels), instead of a mixed model with crossed random effects. Mixed-model options in Python are either approximate (`BinomialBayesMixedGLM`) or require R (pymer4). Clustered errors are robust and easy to explain to readers.
- **Agreement:** Cohen's κ (scikit-learn), Krippendorff's α (`krippendorff`), and weighted precision/recall/F1 using inverse inclusion probabilities.
- **Plots:** matplotlib, one consistent palette, saved as PNG and PDF.

### D9. Annotation app
A Streamlit app that reads the sample parquet and writes labels incrementally to `results/annotations/pass{n}.parquet` after every label, so a crash loses at most one label.
- Keyboard shortcuts: 1/2/3 for the refusal label, h/n for harmful.
- Order: shuffled with a seed derived from the pass number.
- Pass gating: the pass-2 interval is enforced from the timestamp of the last pass-1 label.
- Blinding: the app only ever loads blinded fields; metadata joins happen at import time.

### D10. Pilot checkpoint before the full run
50 intents × 8 models (direct framing) plus a small RQ3 slice go through the whole pipeline before the main run. It produces:
- block-code discovery,
- judge refusal rates,
- token and cost totals,
- a 100-item self-annotated calibration sample that selects the primary evaluator.

Judge prompts are then frozen (version hash recorded). Any change after that requires re-judging everything.

## Risks / Trade-offs

- **Every LLM judge belongs to an evaluated model family (self-preference).** → Report agreement on own-family vs. other-family responses (study-analysis spec). WildGuard, run locally from outside these families, serves as the independent evaluator.
- **DeepSeek (Judge B) may block or refuse judging harmful content at high rates.** → Measure in the pilot. If the block-plus-refusal rate exceeds 5%, switch Judge B to GLM or Kimi through config.
- **Provider content filters can make a model look safer than it is.** → `api_blocked` is a separate outcome, excluded from FAR/ORR denominators and reported alongside them.
- **Single annotator who is also the researcher.** → Blinding, two passes at least 7 days apart, test-retest κ, and external human-labeled datasets as the main accuracy reference. Stated as a limitation.
- **MLX 8-bit WildGuard may differ from the published model.** → Report its F1 on WildGuardTest next to the published figure. If the gap is more than 2 points, fall back to fp16 on MPS (`transformers`); the cost is runtime, not code structure.
- **Benchmark contamination**, since models may have been trained on XSTest or AdvBench. → Noted as a limitation. Framing variants partly mitigate it because they are new text.
- **Rewrites drifting from the original intent.** → The intent check, plus a manual spot check of 50 rewrites before RQ3 generation.
- **Model versions changing silently behind provider aliases.** → Use dated or pinned model IDs where offered, and record the `model` string the API returns.
- **Cost overruns.** → Dry-run estimates, `--limit`, caching, and the pilot cost total, which extrapolates the full-run budget.
- **Clustered-SE GLM vs. a mixed model.** → Simpler and more robust, but it does not partition variance by intent. Acceptable for the RQs, which compare fixed effects.

## Migration Plan

Nothing to migrate (greenfield). Bring-up order:
1. `git init`, the uv project, configs and `.env`.
2. Run the pilot end to end.
3. Freeze the prompts.
4. Main and RQ3 generation.
5. Evaluation.
6. Self-annotation passes 1 and 2.
7. Analysis.

## Open Questions

- Exact model IDs: two per provider, ideally a reasoning/non-reasoning pair. Configuration only.
- API endpoint region for Zhipu and Moonshot (`.cn` vs. international). Configuration only.
- Exact content-filter error codes for each provider. Discovered in the pilot and recorded as data.
- Final unified category list (about 8–10). Edited in `taxonomy.yaml` after reviewing source categories in task 2.x. No code impact.
