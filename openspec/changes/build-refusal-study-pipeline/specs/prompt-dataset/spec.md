## Purpose

Builds the labeled English prompt set the study runs on, from public safety benchmarks. It covers unified risk categories, expected behavior, clarity tiers, safe/unsafe pairing, five framings of each intent, and fixed pilot, main and RQ3 splits.

## ADDED Requirements

### Requirement: Benchmark ingestion into a unified schema
The system SHALL load XSTest, OR-Bench (Hard-1K and Toxic subsets), JailbreakBench behaviors (harmful and benign), HarmBench, SORRY-Bench and StrongREJECT. It SHALL convert them to prompt records with these fields:
- a stable prompt id,
- intent id,
- source dataset,
- source category,
- unified category,
- expected behavior (`refuse` or `comply`),
- tier,
- framing,
- an ordered list of user turns,
- an optional pair id,
- split.

#### Scenario: Records conform to schema
- **WHEN** the dataset build completes
- **THEN** every prompt record passes schema validation and has a non-empty first user turn

#### Scenario: Stable identifiers
- **WHEN** the dataset is rebuilt from the same raw data and configuration
- **THEN** every prompt keeps the same prompt id and intent id

#### Scenario: Gated or missing source
- **WHEN** a source cannot be downloaded because access is missing
- **THEN** the build fails with a message naming the dataset and the access step required

### Requirement: Expected behavior labeling
Each prompt SHALL carry an expected behavior derived from its source's labels. Prompts from safe or benign subsets get `comply`; prompts from harmful subsets get `refuse`.

#### Scenario: XSTest contrast prompts
- **WHEN** XSTest prompts are ingested
- **THEN** its safe prompts are labeled `comply` and its unsafe contrast prompts `refuse`

### Requirement: Unified risk taxonomy
The system SHALL map every prompt, safe or unsafe, onto one configured set of unified risk categories.
- Where a source supplies a category, the mapping uses the configured source-to-unified table.
- Where a source has no risk category, the category is assigned by an LLM and the record is flagged as LLM-assigned.

#### Scenario: Unmapped source category
- **WHEN** a source category is missing from the mapping table
- **THEN** the build fails and lists the unmapped categories

#### Scenario: LLM-assigned category flagged
- **WHEN** a safe prompt's category is assigned by an LLM
- **THEN** the record is marked as LLM-assigned, so it can be included in the manual verification sample

### Requirement: Clarity tiers
Each prompt SHALL carry a tier of `clear` or `borderline`. Prompts whose expected behavior is contested, such as dual-use requests, are tagged `borderline` using configured source rules and an LLM screening pass. Analyses SHALL be able to exclude borderline prompts.

#### Scenario: Borderline excluded from primary metrics
- **WHEN** primary metrics are computed
- **THEN** only `clear`-tier prompts are included unless borderline prompts are explicitly requested

### Requirement: Matched safe/unsafe pairs
Where a source provides topic-matched safe and unsafe prompts (XSTest contrast prompts, JailbreakBench harmful and benign behaviors), the system SHALL record a shared pair id linking them.

#### Scenario: JailbreakBench pairing
- **WHEN** JailbreakBench harmful and benign behaviors are ingested
- **THEN** each benign behavior shares a pair id with its matching harmful behavior

### Requirement: Framing variants
For every intent in the RQ3 split, safe and unsafe alike, the system SHALL produce five framings:
- `direct`,
- `academic`,
- `roleplay`,
- `hypothetical`,
- `multiturn`, which is exactly three scripted user turns that build context and end with the request.

All framings of an intent share its intent id, category and expected behavior.

#### Scenario: Five framings per intent
- **WHEN** framing generation completes for the RQ3 split
- **THEN** each intent has exactly one accepted prompt per framing, or is reported as dropped

#### Scenario: Multi-turn structure
- **WHEN** a `multiturn` framing is produced
- **THEN** it contains exactly three user turns, and only the final turn contains the core request

### Requirement: Intent-preservation check
Every non-direct framing SHALL be checked by an LLM to confirm that it requests the same information as the direct version. Rewrites that fail SHALL be regenerated, up to a configured number of attempts. Intents that still fail SHALL be dropped and reported.

#### Scenario: Drifted rewrite rejected
- **WHEN** the intent check judges that an academic rewrite asks for different information than the direct prompt
- **THEN** the rewrite is regenerated, and the failure is recorded

#### Scenario: Persistent failure
- **WHEN** a framing fails the intent check on every allowed attempt
- **THEN** the intent is excluded from the RQ3 split, and it appears in the build report with the failing framing

### Requirement: Deterministic splits
The system SHALL assign prompts to `pilot`, `main` and `rq3` splits using a configured seed and target sizes. Sampling SHALL be stratified by expected behavior and unified category.

#### Scenario: Reproducible splits
- **WHEN** the dataset is rebuilt with the same seed and configuration
- **THEN** every prompt lands in the same split

#### Scenario: Pilot size
- **WHEN** the pilot split is built with a target of 50 intents
- **THEN** it contains 50 intents, balanced between expected behaviors

### Requirement: Build report
The dataset build SHALL output a summary of:
- counts by source, split, expected behavior, category, tier and framing;
- deduplication removals;
- intents dropped by the intent check.

#### Scenario: Report generated
- **WHEN** the dataset build completes
- **THEN** a build report file exists containing all of these counts
