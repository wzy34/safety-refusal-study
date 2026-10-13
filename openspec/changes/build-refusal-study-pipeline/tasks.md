## 1. Project setup

- [x] 1.1 Run `git init` and add a `.gitignore` covering `data/`, `results/`, `models/`, `.env`, the cache DB and `.venv`; verify with `git check-ignore data/x results/x models/x .env`
- [x] 1.2 Install uv, create `pyproject.toml` (Python 3.12, `srs` package under `src/`, dependencies from the proposal, pytest and ruff as dev dependencies) and `uv.lock`; verify `uv sync` succeeds and `uv run python -c "import srs"` works
- [x] 1.3 Create the package skeleton from design D5 (empty modules plus `__init__.py`) and `tests/`; verify `uv run pytest` runs (0 tests, no errors)
- [x] 1.4 Add `.env.example` listing `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `MOONSHOT_API_KEY`, `ZHIPU_API_KEY` and `HF_TOKEN`; verify the file exists and contains no real values

## 2. Configuration, schemas and manifests

- [x] 2.1 Implement pydantic config models and loader for `providers.yaml`, `models.yaml`, `evaluators.yaml`, `taxonomy.yaml`, `framings.yaml` and `experiment.yaml`, including the cross-check that each model's provider exists; verify with tests for valid load, unknown field, wrong type and unknown provider
- [x] 2.2 Write initial config files: 4 providers, 8 model placeholders with capability flags (D2), the 5 evaluators, an experiment section with pilot/main/rq3 sizes, seed, temperature, max_tokens=1024 and stability-subset settings; verify they load through 2.1
- [x] 2.3 Implement secret loading from env/`.env`, with a clear error for a missing key; verify with a test that unsets a key and checks the error names the variable
- [x] 2.4 Implement pydantic record schemas (`Prompt`, `Response`, `Judgment`, `HumanLabel`, `ReferenceItem`) and Parquet read/write helpers; verify with round-trip tests, including nested turns and transcripts
- [x] 2.5 Implement deterministic ID helpers (`prompt_id`, `intent_id`, `response_id`, canonical-JSON hashing); verify with tests showing stability and sensitivity to each input
- [x] 2.6 Implement versioned prompt templates (load from `prompts/`, content-hash version); verify with a test that editing a template changes its hash
- [x] 2.7 Implement run manifests (run id, command, config snapshot without secrets, git commit plus uncommitted-changes flag, timestamps, counts); verify with a test that a manifest contains no key values and records the uncommitted-changes flag
- [x] 2.8 Implement the typer CLI with the `data`, `generate`, `classic`, `evaluate`, `annotate` and `analyze` groups (stubs), and logging that never prints full prompt or response text; verify `uv run srs --help` lists every group

## 3. LLM client

- [x] 3.1 Implement `build_request()` per-model adapters (token parameter name, temperature/seed support, structured-output mode, recording of dropped parameters); verify with unit tests for each capability combination
- [x] 3.2 Implement the SQLite request cache (hash → request, raw response, status, timestamp; only `ok` and `api_blocked` stored); verify with tests for a hit, a miss after a parameter change, and persistence across reopen
- [x] 3.3 Implement per-provider concurrency semaphores and a token-bucket rate limiter; verify with an async test that in-flight requests never exceed the limit
- [x] 3.4 Implement error classification (`retry`, `blocked` or `fatal`) driven by `providers.yaml` code lists, plus `finish_reason == "content_filter"`; verify with table-driven tests
- [x] 3.5 Implement `ChatClient.complete()` combining the adapter, cache, limiter and retries with backoff, and parsing `reasoning_content`, usage and latency into a `Completion`; verify with tests against the fake client covering success, a retry that then succeeds, a block and retries exhausted
- [x] 3.6 Implement the fake client (scripted completions, blocks and errors, no network); verify the full llm test suite passes with networking disabled
- [ ] 3.7 Add an `srs llm ping` smoke command that sends one trivial request to each configured model; verify manually after keys are set that each of the 8 models returns status `ok`

## 4. Prompt dataset

- [ ] 4.1 Implement `srs data fetch` to download all six benchmarks into `data/raw/` with a clear error for gated or missing access; verify the files exist and a missing-token run prints the access step
- [x] 4.2 Implement source loaders for XSTest, OR-Bench (Hard-1K and Toxic), JailbreakBench (harmful and benign), HarmBench, SORRY-Bench and StrongREJECT, producing `Prompt` records with expected behavior; verify with a test per loader on a small fixture (count, schema, expected-behavior mapping)
- [ ] 4.3 Review the source categories, finalize 8–10 unified categories in `taxonomy.yaml`, and implement the mapping with an unmapped-category failure; verify that the build lists no unmapped categories and that a test with an injected unknown category fails
- [x] 4.4 Implement LLM category assignment for prompts without risk categories, with the LLM-assigned flag; verify on a fixture of 10 safe prompts that every record gets a valid category and the flag
- [x] 4.5 Implement clarity tiers (configured source rules plus an LLM screen); verify with a fixture test and by checking that the build report shows tier counts
- [x] 4.6 Implement matched pair ids for XSTest contrast prompts and JailbreakBench harmful/benign behaviors; verify with a test that each JBB benign behavior shares a pair id with its harmful counterpart
- [x] 4.7 Implement deduplication and seeded stratified splits (pilot = 50 intents, main, rq3 ≈ 200 intents); verify with tests that the splits are identical across rebuilds and the pilot is balanced
- [x] 4.8 Write framing templates and rewrite prompts for academic, role-play, hypothetical and 3-turn multi-turn framings; implement framing generation with template-only fallback; verify on 5 fixture intents that each gets 5 framings and every multi-turn prompt has exactly 3 turns
- [x] 4.9 Implement the intent-preservation check (different-family checker, 3 attempts, drop and report); verify with fake-client tests for pass, a regenerate-then-pass and a persistent failure
- [ ] 4.10 Implement `srs data build` and `srs data frame` writing `prompts.parquet` and a build report (counts by source, split, expected behavior, category, tier and framing; dedup removals; dropped intents); verify by running on the real data and reviewing the report
- [ ] 4.11 Manually spot-check 50 framing rewrites and 30 LLM-assigned categories, then fix templates or the taxonomy if needed; verify with the notes saved in `results/qa/dataset_review.md`

## 5. Response generation

- [x] 5.1 Implement the single-turn generation runner (prompt × model × sample, deterministic response ids, skips responses already completed, retries `error`); verify with fake-client tests for coverage and idempotent reruns
- [x] 5.2 Implement the scripted multi-turn loop with live assistant turns and recording of the turn where a block occurred; verify with fake-client tests: a complete 3-turn transcript, and a block at turn 2
- [x] 5.3 Implement the stability subset (n samples at the configured temperature, sample indices 1..n); verify with a test that the expected number of extra responses is produced
- [x] 5.4 Implement `--dry-run` (planned calls and estimated tokens per model, no requests) and `--limit`; verify with tests that the fake client receives zero calls in dry-run and that the limit is respected
- [x] 5.5 Wire `srs generate --set ... --models ...` with run manifests; verify an end-to-end run on the fake client writes `responses.parquet` and a manifest referencing the run id

## 6. Evaluators

- [x] 6.1 Implement the `Evaluator` protocol, blinded `EvalItem` construction, the evaluate stage (skipping already-judged triples and non-`ok` responses), and the two yes/no reduction rules; verify with tests that judge input contains no model name, expected behavior, category or reasoning text, and that reruns create no duplicates
- [x] 6.2 Implement the keyword evaluators (`keyword` with the Zou et al. list, `keyword_ext`); verify with unit tests on refusal and compliance examples
- [x] 6.3 Implement classifier training data assembly from WildGuardMix-train with the reference-overlap leakage guard; verify with a test that injected overlapping items are removed
- [ ] 6.4 Implement RoBERTa-base fine-tuning on MPS (validation split, early stopping, middle truncation) via `srs classic train`; verify that the checkpoint and a validation report (accuracy/P/R/F1) are saved in `models/`
- [x] 6.5 Implement classifier inference as an evaluator; verify on 20 fixture responses that labels are produced and the version matches the checkpoint
- [ ] 6.6 Convert WildGuard to MLX 8-bit (one-time script) and implement the safety-model evaluator with the official template, greedy decoding and strict output parsing (`parse_error` on mismatch); verify on 20 fixture responses that labels are produced with no network calls
- [x] 6.7 Write the judge rubric (definitions, worked examples, JSON with rationale first) and implement the LLM judge evaluator (json_schema for Gemini, json_object plus validation and retries for DeepSeek, detection of `judge_refused` and `blocked`); verify with fake-client tests for each judge status
- [x] 6.8 Implement `srs evaluate --evaluators ... --target responses|external`; verify an end-to-end fake run writes `judgments.parquet` with all 5 evaluator ids

## 7. Reference labels and annotation

- [ ] 7.1 Implement loaders for WildGuardTest, the JailbreakBench judge-comparison set, the HarmBench classifier validation set and the XSTest annotated completions, mapped to the shared label vocabulary; verify with a test per loader on fixtures, and check the item counts against the published sizes
- [x] 7.2 Implement the stratified annotation sampler (oversampling disagreements, inclusion probabilities, seed recorded, blocked responses excluded); verify with tests that probabilities are in (0, 1] and that a rebuild with the same seed gives the same sample
- [x] 7.3 Implement the Streamlit annotation app (blinded fields only, seeded shuffle per pass, keyboard shortcuts, save after every label, resume); verify by running `srs annotate app` on a 10-item fixture: label 3 items, restart, and confirm it resumes at item 4 with no metadata visible
- [x] 7.4 Implement pass-2 gating (minimum interval, first-pass labels hidden, new shuffle); verify with a test that pass 2 is refused before the interval and that the available date is shown
- [x] 7.5 Implement `srs annotate import`, the test-retest κ report, the list of disagreements and adjudication storage; verify with a test on synthetic two-pass labels with known κ

## 8. Analysis

- [x] 8.1 Implement core metrics (FAR_comply, FAR_harm, ORR, J, block rate, Wilson CIs, both yes/no rules, clear-tier default); verify with the hand-computed scenario from the study-analysis spec (FAR 0.3, ORR 0.2, J 0.5)
- [x] 8.2 Implement stats helpers (exact McNemar, Holm, seeded bootstrap, Kendall's W, Spearman, logistic GLM clustered by intent); verify with tests against known values or scipy/statsmodels reference outputs
- [x] 8.3 Implement agreement tools (pairwise κ, Krippendorff's α, weighted P/R/F1, signed bias, Kendall's τ with bootstrap CI, self-preference split by family, judge status rates); verify with synthetic-data tests where agreement is known
- [x] 8.4 Implement RQ1 (pairwise table, FAR–ORR correlation, sensitivity tables for each evaluator); verify on synthetic data that 8 models give 28 comparisons per metric
- [x] 8.5 Implement RQ2 (category heatmaps, Kendall's W, interaction GLM table); verify that the heatmaps and table are written for synthetic data
- [x] 8.6 Implement RQ3 (framing refusal rates, flip rates against direct for each expected behavior, intent-consistency score); verify with a synthetic test with known flip rates
- [x] 8.7 Implement RQ4 (FAR × ORR plot with CI error bars, Pareto marking, distance to origin); verify with a test that a dominated model is not marked Pareto-optimal
- [x] 8.8 Implement RQ5 (all reliability outputs against external and weighted self-annotation references); verify that the output tables are complete on synthetic data
- [x] 8.9 Wire `srs analyze rq1..rq5|all` with manifests, CSV/PNG/PDF outputs and a fixed seed; verify that two reruns produce identical CSVs

## 9. Pilot checkpoint

- [ ] 9.1 Fill in real model IDs and endpoints, run `srs llm ping`, then `srs generate --set pilot --dry-run`; verify that all 8 models respond and record the cost estimate
- [ ] 9.2 Run pilot generation (50 intents × 8 models, plus a small RQ3 slice); verify coverage in the manifest counts and record the observed block codes for each provider in `providers.yaml`
- [ ] 9.3 Run all 5 evaluators on the pilot and on the external reference sets; verify that judge refusal/block/parse rates are reported, and switch Judge B to GLM or Kimi if DeepSeek's combined rate exceeds 5%
- [ ] 9.4 Check WildGuard MLX fidelity on WildGuardTest against the published F1; verify the gap is 2 points or less, otherwise switch to fp16 MPS per the design
- [ ] 9.5 Self-annotate a 100-item pilot calibration sample and select the primary evaluator based on agreement with the reference; record the choice and its rationale in `experiment.yaml`
- [ ] 9.6 Review pilot outputs end to end (analysis runs on pilot data, cost extrapolation for the main run), then freeze judge and rewrite prompts; verify that the frozen template hashes are recorded in `results/qa/pilot_review.md`

## 10. Full study run

- [ ] 10.1 Generate the main and RQ3 splits plus the stability subset for all 8 models; verify that manifest counts match the planned coverage and investigate any `error` status
- [ ] 10.2 Run all 5 evaluators on all responses; verify that each `ok` response has a judgment from every evaluator (or a recorded non-ok judge status)
- [ ] 10.3 Draw the ~350-item annotation sample and complete pass 1; verify that pass1.parquet has every item labeled
- [ ] 10.4 After 7 or more days, complete pass 2, import and adjudicate; verify that the test-retest κ report exists and every disagreement is adjudicated
- [ ] 10.5 Run `srs analyze all` and review RQ1–RQ5 tables and figures; verify that every output file is present and the manifests name the primary evaluator and input files
