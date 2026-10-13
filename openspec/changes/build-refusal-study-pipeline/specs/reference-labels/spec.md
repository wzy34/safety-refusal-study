## Purpose

Supplies the human reference labels that RQ5 measures evaluators against. These come from public human-labeled datasets, plus the researcher's own blind two-pass labels on a weighted sample of the study's responses.

## ADDED Requirements

### Requirement: External human-labeled reference sets
The system SHALL load WildGuardTest, the JailbreakBench judge-comparison set, the HarmBench classifier validation set and the XSTest annotated completions. Each item SHALL be normalized to a reference record containing:
- prompt,
- response,
- refusal label,
- harmful label (where available),
- source dataset.

#### Scenario: Labels mapped to shared vocabulary
- **WHEN** a reference set uses its own label names
- **THEN** they are mapped to the shared refusal and harmful vocabulary, and labels the set lacks are null

### Requirement: Stratified annotation sample
The system SHALL draw a configurable number of study responses for self-annotation (default 350). The sample is stratified by model, expected behavior and framing, with responses where the automated evaluators disagree oversampled. Each sampled item SHALL carry its inclusion probability so analyses can weight it back.

#### Scenario: Weights recorded
- **WHEN** the annotation sample is drawn
- **THEN** every sampled item has an inclusion probability between 0 and 1, and the sampling seed is recorded

#### Scenario: Blocked responses excluded
- **WHEN** the sample is drawn
- **THEN** responses with status `api_blocked` or `error` are not eligible

### Requirement: Blind annotation interface
The system SHALL provide a local annotation interface. It shows each sampled conversation and collects:
- a 3-class refusal label,
- a harmful label (for prompts expected to be refused),
- optional notes.

It MUST hide the model identity, evaluator outputs, expected behavior and category, and it SHALL present items in random order.

#### Scenario: Hidden metadata
- **WHEN** an item is shown to the annotator
- **THEN** no model name, evaluator label, expected behavior or category is visible

#### Scenario: Resume annotation
- **WHEN** the annotator closes the interface and reopens it
- **THEN** completed labels are kept and annotation resumes at the next unlabeled item

### Requirement: Two-pass annotation
The interface SHALL support a second annotation pass over the same items. Items appear in a newly shuffled order, and first-pass labels stay hidden. The second pass SHALL NOT open until a configured minimum interval (default 7 days) has passed since the first pass was completed.

#### Scenario: Second pass hides first-pass labels
- **WHEN** the annotator labels an item in pass 2
- **THEN** the pass-1 label for that item is not shown

#### Scenario: Interval enforced
- **WHEN** the annotator attempts to start pass 2 before the minimum interval has elapsed
- **THEN** the interface refuses and shows the date when pass 2 becomes available

### Requirement: Annotation export and reliability
The system SHALL import completed annotations as human label records containing:
- response id,
- pass number,
- labels,
- notes,
- seconds spent.

It SHALL report test-retest agreement (Cohen's κ) between passes. Items that disagree between passes SHALL be flagged for a final adjudicated label.

#### Scenario: Test-retest reported
- **WHEN** both passes are imported
- **THEN** a report gives κ for the refusal label and the harmful label, and lists items that disagree between passes

#### Scenario: Adjudication
- **WHEN** an item's pass-1 and pass-2 labels differ
- **THEN** the item gets an adjudicated label, which is used as the reference label
