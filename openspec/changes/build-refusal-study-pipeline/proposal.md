## Why

The study in `research_ideas.md` asks whether LLMs refuse appropriately: under-refusal (accepting unsafe requests) versus over-refusal (rejecting safe ones). It also asks whether the automated evaluators used to measure this can be trusted (RQ1–RQ5). The project currently contains no code. Answering the RQs needs one reproducible pipeline that:
- builds a labeled prompt set,
- collects responses from 8 models across 4 API providers,
- labels every response with 5 automated evaluators plus human reference labels,
- produces the statistics and figures for each RQ.

## What Changes

- New Python project (`srs` package, managed with uv) driven by a typer CLI and YAML configs. API keys come from `.env`. Each run records a manifest of its config and code version.
- **Async API client.** Uses the official `openai` SDK with a separate `base_url` for each provider: Gemini (via its OpenAI-compatible endpoint), DeepSeek, Qwen/Alibaba Cloud Bailian and GLM/Zhipu; the Claude judge goes through the official `anthropic` SDK.
  - Each model has an adapter that adjusts request parameters to what the model supports.
  - Rate limits apply per provider. Retries use exponential backoff.
  - Provider content-filter blocks are recorded as `api_blocked`, never as model refusals.
  - Every call goes through an SQLite cache keyed by a hash of the request.
- **Prompt dataset builder.**
  - Converts XSTest, OR-Bench, JailbreakBench, HarmBench, SORRY-Bench and StrongREJECT into one prompt schema.
  - Maps every prompt to one unified risk taxonomy and tags it clear or borderline.
  - Builds safe/unsafe matched pairs.
  - Renders 5 framings (direct, academic, role-play, hypothetical, scripted 3-turn), with an LLM check that each rewrite keeps the original intent.
  - Splits the prompts into pilot, main and RQ3 sets.
- **Response generation runner.** Covers 8 evaluated models (2 per provider). Runs single-turn prompts and scripted multi-turn conversations. Stores reasoning traces separately from the final answer. Supports `--dry-run` cost estimates and `--limit`.
- **Five evaluators that share one interface:**
  1. Keyword matching.
  2. Fine-tuned RoBERTa-base classifier.
  3. WildGuard 7B run locally with MLX at 8-bit.
  4. Claude Haiku 5.5 rubric judge (Judge A; Gemini until 2026-10-15).
  5. DeepSeek rubric judge (Judge B), using the same rubric.

  All evaluators write to one judgment schema (3-class refusal + harmfulness). When a judge refuses to judge, or a parse fails, that is recorded as a separate status.
- **Reference labels.**
  - Loads human-labeled public datasets: WildGuardTest, the JailbreakBench judge-comparison set, the HarmBench classifier validation set and the XSTest response annotations.
  - Draws a stratified annotation sample with inverse-probability weights.
  - Provides a blind Streamlit app for two-pass self-annotation.
- **Analysis.**
  - FAR (by compliance and by harmfulness), ORR, Youden's J and confidence intervals.
  - McNemar tests with Holm correction, and logistic regression with standard errors clustered by intent.
  - Category profiles and framing flip rates.
  - The FAR × ORR plane with its Pareto frontier.
  - Evaluator agreement (κ, α), accuracy against the reference, signed bias, ranking stability and self-preference.
  - Tables and figures for RQ1–RQ5.
- A pilot checkpoint (50 intents × 8 models) before the full run, to fix provider block codes, judge refusal rates, cost and the frozen judge prompts.

## Capabilities

### New Capabilities
- `study-config`: Typed YAML configuration (providers, models, evaluators, taxonomy, framings, experiment), secrets from `.env`, versioned prompt templates, run manifests and CLI entry points.
- `llm-client`: The OpenAI-compatible async client: per-model request adapters, provider rate limits, retries, error classification (including `api_blocked`) and the SQLite request cache.
- `prompt-dataset`: Benchmark loaders, unified taxonomy, clear/borderline tiers, matched pairs, framing generation, intent-preservation check and dataset splits.
- `response-generation`: Running prompts × models × samples (single- and multi-turn) to produce `Response` records with status, reasoning text, usage and provenance.
- `response-evaluators`: The five evaluators behind one interface, classifier training, the shared judgment schema, and the rules for reducing labels to yes/no.
- `reference-labels`: Loading external human-labeled datasets, stratified annotation sampling with weights, and the blind two-pass Streamlit annotation app.
- `study-analysis`: Metrics, statistical tests, agreement analysis and per-RQ tables and figures.

### Modified Capabilities
<!-- None: the project has no existing specs. -->

## Impact

- **New code:** `pyproject.toml`, `src/srs/**`, `configs/*.yaml`, `prompts/*`, `tests/**`, `.env.example` and `.gitignore`. The folder should be initialized as a git repository so run manifests can record the code version.
- **Dependencies:** `openai`, `tenacity`, `pydantic`, `typer`, `pyyaml`, `pandas`, `pyarrow`, `datasets`, `mlx-lm`, `transformers`, `torch`, `scikit-learn`, `statsmodels`, `scipy`, `krippendorff`, `streamlit`, `matplotlib` and `pytest`.
- **External systems:**
  - The Gemini, DeepSeek, Alibaba Cloud Bailian and Zhipu APIs, which cost money.
  - Hugging Face Hub, which needs `HF_TOKEN` and accepted licenses for the gated WildGuard and WildGuardMix.
  - About 8 GB of local RAM for WildGuard under MLX.
- **Data handling:** responses can contain harmful content. `data/`, `results/` and `models/` are git-ignored, and logs never print full response text.
- **Constraints:** English only. No external annotators; the reference is public human-labeled datasets plus the researcher's own blind labels.
