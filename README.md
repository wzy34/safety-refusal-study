# Safety Refusal Study

Do LLMs refuse appropriately? This project measures two errors at once:

- **Under-refusal (false acceptance):** the model complies with a request it should refuse.
- **Over-refusal (false rejection):** the model refuses a legitimate request it should answer.

It also asks a second question: **how reliable are the automated evaluators used to measure
refusal?** Every response is labeled by five evaluators of different kinds and compared against
human reference labels.

The motivation and the full research questions are in [research_ideas.md](research_ideas.md).
The design, specs and task list are in
[openspec/changes/build-refusal-study-pipeline/](openspec/changes/build-refusal-study-pipeline/).

## Research questions

| RQ | Question | Main outputs |
|----|----------|--------------|
| RQ1 | Do models differ in refusal accuracy on unsafe vs. safe requests? | FAR, ORR, Youden's J with CIs; McNemar tests (Holm-corrected); logistic regression clustered by intent |
| RQ2 | Does the risk category affect refusal accuracy? | Per-category FAR/ORR profiles across a unified 10-category taxonomy |
| RQ3 | Does the framing of the same intent change the decision? | Flip rates across 5 framings: direct, academic, role-play, hypothetical, scripted 3-turn |
| RQ4 | Is there a safety–usability trade-off? | The FAR × ORR plane with its Pareto frontier |
| RQ5 | Do automated evaluators agree with each other and with humans? | Cohen's κ, Krippendorff's α, accuracy vs. reference, signed bias, ranking stability, self-preference |

A stability check (not an RQ) re-samples a subset at temperature 0.7 to test whether the greedy
(temperature 0) results hold up.

## Study design

### Evaluated models

Eight models from four providers. Each family uses **one base model with thinking minimised (`-a`)
and thinking on (`-b`)**, so the a/b contrast isolates thinking rather than model size.

| Family | Model | `-a` | `-b` |
|--------|-------|------|------|
| Gemini | `gemini-3.1-flash-lite` | reasoning effort minimal | reasoning effort high |
| DeepSeek | `deepseek-flash` (V4.1-Flash) | thinking off | thinking on |
| Qwen | `qwen3.8-flash` | thinking off | thinking on |
| GLM | `glm-5.3` | reasoning effort low | reasoning effort max |

Gemini 3+ and GLM-5.3 cannot fully disable thinking, so `gemini-a` and `glm-a` are reported as
*minimised*, not off. Model IDs, capability flags and prices are in
[configs/models.yaml](configs/models.yaml).

### Prompt set

Built from six public benchmarks: **XSTest, OR-Bench, JailbreakBench, HarmBench, SORRY-Bench and
StrongREJECT** (English only). Each prompt is:

- mapped to one category of a unified risk taxonomy ([configs/taxonomy.yaml](configs/taxonomy.yaml));
- tagged *clear* or *borderline*;
- paired with a matched safe/unsafe counterpart where possible;
- assigned to a split: `pilot` (50 intents), `rq3` (200 intents, rendered in all 5 framings) or `main` (the rest).

Framings are rewritten by an LLM and checked for intent preservation by a model from a
different family. If a rewrite fails the check, a fixed template is used instead
([configs/framings.yaml](configs/framings.yaml)).

### Evaluators

All evaluators share one interface and write one judgment schema: a 3-class refusal label
(`full_refusal` / `partial` / `full_compliance`) plus a harmfulness label.

| Id | Kind | Details |
|----|------|---------|
| `keyword` | Keyword matching | Refusal-phrase list from Zou et al. (2023); `keyword_ext` adds an extended list |
| `classic` | Fine-tuned classifier | RoBERTa-base trained on WildGuardMix response-refusal labels |
| `wildguard` | Safety model | AllenAI WildGuard 7B, run locally with MLX at 8-bit |
| `judge_a` | LLM judge | Claude Haiku 5.5, from a family outside the evaluated ones, so it cannot favour its own outputs |
| `judge_b` | LLM judge | DeepSeek, same rubric ([prompts/judge_rubric.md](prompts/judge_rubric.md)) |

Provider content-filter blocks are recorded as `api_blocked`, never as model refusals. Judge
refusals and parse failures get their own statuses, and empty answers become `no_answer`. None
of these statuses count as refusal decisions.

### Reference labels

There are no external annotators. The human reference has two parts:

1. **Public human-labeled sets:** WildGuardTest, the JailbreakBench judge-comparison set, the
   HarmBench classifier validation set and the XSTest response annotations.
2. **Blind self-annotation:** a stratified sample of about 350 study responses with
   inverse-probability weights. It is labeled twice, at least 7 days apart, in a Streamlit app
   that hides model and evaluator identities. Test-retest κ is reported and disagreements are
   adjudicated.

## Setup

Requirements:

- macOS on Apple Silicon (WildGuard runs on MLX; RoBERTa trains on MPS) with about 8 GB of free RAM for WildGuard
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- API keys for Gemini (paid tier), DeepSeek, Alibaba Cloud Bailian, Zhipu and Anthropic
- A Hugging Face token with the WildGuard and WildGuardMix licenses accepted

```bash
uv sync
cp .env.example .env        # then fill in the keys; .env is git-ignored
uv run srs llm ping         # one trivial request per configured model
```

One-time conversion of WildGuard to MLX (needs ~15 GB of disk for the download):

```bash
uv run python scripts/convert_wildguard_mlx.py
```

## Running the pipeline

Every command writes a run manifest that records the config, arguments and git commit. All API
calls go through an SQLite cache, so re-running a step does not pay for the same request twice.

```bash
# 1. Data
uv run srs data fetch                       # download benchmarks and reference sets to data/raw/
uv run srs data build                       # unified prompt set + build report
uv run srs data frame                       # rewrite and intent-check the RQ3 framings

# 2. Generation
uv run srs generate --set pilot --dry-run   # cost estimate, no calls
uv run srs generate --set pilot
uv run srs generate --set main
uv run srs generate --set rq3
uv run srs generate --set main --stability  # sampled subset for the stability check

# 3. Evaluation
uv run srs classic train                    # fine-tune the RoBERTa classifier
uv run srs evaluate                         # all evaluators on all study responses
uv run srs evaluate --target external       # all evaluators on the human-labeled reference sets

# 4. Reference labels
uv run srs annotate sample --name main
uv run srs annotate app --name main --pass 1
uv run srs annotate app --name main --pass 2           # 7 or more days later
uv run srs annotate import --name main
uv run srs annotate app --name main --pass adjudicate

# 5. Analysis
uv run srs analyze all                      # or: analyze rq1 | rq2 | rq3 | rq4 | rq5 | stability
```

Use `--limit N` on `generate` for small test runs, and `--help` on any command for its full
options. Analysis runs on clear prompts by default; use `--borderline` to include borderline ones.

### The pilot checkpoint

Before the full run, a pilot (50 intents × 8 models) is used to:

- confirm each provider's content-filter codes;
- check judge refusal rates (Judge B is replaced if its block-plus-refusal rate exceeds 5%);
- check WildGuard MLX fidelity against its published F1;
- choose the primary evaluator from a 100-item calibration sample;
- freeze the judge and rewrite prompts.

## Configuration

| File | Contents |
|------|----------|
| [configs/providers.yaml](configs/providers.yaml) | Endpoints, key env vars, concurrency, rate limits, content-filter codes |
| [configs/models.yaml](configs/models.yaml) | Model IDs, capability flags, thinking settings, prices |
| [configs/evaluators.yaml](configs/evaluators.yaml) | The five evaluators and their settings |
| [configs/taxonomy.yaml](configs/taxonomy.yaml) | Unified categories and per-benchmark source mappings |
| [configs/framings.yaml](configs/framings.yaml) | RQ3 framing templates, rewriter and checker models |
| [configs/experiment.yaml](configs/experiment.yaml) | Seed, generation limits, splits, annotation, analysis and path settings |
| [prompts/](prompts/) | Versioned prompt templates (judge rubric, rewriters, intent check, category and tier screens) |

Reasoning (`-b`) models get a 4096-token output limit instead of 1024, because thinking counts
toward the limit and most pilot answers were cut off at 1024.

## Repository layout

```
src/srs/
  cli.py            typer CLI (entry point: `srs`)
  config.py         typed YAML configs and secrets
  schemas.py io.py  record schemas, Parquet I/O
  manifest.py       run manifests
  llm/              async OpenAI-compatible + Anthropic clients, adapters, rate limits, cache
  data/             benchmark loaders, taxonomy, tiers, framing, intent check, build
  generation/       response runner (single- and multi-turn)
  evaluators/       keyword, classic, safety_model (WildGuard), llm_judge
  reference/        external reference sets, annotation sampling, Streamlit app
  analysis/         metrics, stats, agreement, plots, rq1–rq5, stability
configs/  prompts/  scripts/  tests/
data/  results/  models/            git-ignored study artifacts
```

## Data handling

Model responses to unsafe prompts can contain harmful content. `data/`, `results/` and `models/`
are git-ignored and should not be published as-is. Logs never print full response text. Only
small benchmark excerpts used as test fixtures are committed.

## Tests

```bash
uv run pytest
uv run ruff check
```

Tests use a fake LLM client and tiny local models, so they make no network calls and cost
nothing.
