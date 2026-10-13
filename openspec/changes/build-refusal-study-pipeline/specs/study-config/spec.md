## Purpose

Defines how the study is configured, how secrets are supplied, how prompt templates are versioned, and how each pipeline run records its provenance. Every result must be traceable to the exact settings that produced it.

## ADDED Requirements

### Requirement: Validated configuration files
The system SHALL load study settings from YAML files covering providers, evaluated models, evaluators, taxonomy, framings and experiment parameters. It SHALL validate them against a typed schema before any pipeline stage runs.

#### Scenario: Valid configuration loads
- **WHEN** all configuration files are present and well-formed
- **THEN** every pipeline command starts with the parsed settings available

#### Scenario: Invalid configuration is rejected
- **WHEN** a configuration file has a missing required field, an unknown field or a value of the wrong type
- **THEN** the command exits with a non-zero status before making any API call
- **AND** the error message names the file and the offending field

#### Scenario: Model references an unknown provider
- **WHEN** a model entry names a provider that is not defined in the provider configuration
- **THEN** configuration loading fails with an error naming the model and the missing provider

### Requirement: Secrets from environment
The system SHALL read API keys and the Hugging Face token only from environment variables, optionally loaded from a local `.env` file. It MUST NOT write secrets to logs, caches, manifests or result files.

#### Scenario: Missing key for a used provider
- **WHEN** a command needs a provider whose API key variable is unset
- **THEN** the command fails before sending requests, with a message naming the missing variable

#### Scenario: Secrets excluded from manifests
- **WHEN** a run manifest is written
- **THEN** it contains no API key or token values

### Requirement: Versioned prompt templates
The system SHALL store judge rubrics, rewrite prompts and intent-check prompts as text templates. It SHALL identify each template version by a content hash, and record that hash with every output produced from it.

#### Scenario: Template edit changes version
- **WHEN** a judge rubric template's text changes
- **THEN** judgments produced afterwards carry a different evaluator version than those produced before

### Requirement: Run manifests
Every pipeline command that produces data SHALL write a run manifest. The manifest records:
- a unique run id,
- the command and its arguments,
- a snapshot of the resolved configuration,
- the code version (commit id, and whether there were uncommitted changes),
- start and end timestamps,
- counts of records produced.

#### Scenario: Manifest written for a generation run
- **WHEN** a generation command completes
- **THEN** a manifest exists for that run id, and every response it produced references the run id

#### Scenario: Uncommitted code flagged
- **WHEN** a command runs while the working tree has uncommitted changes
- **THEN** the manifest records that the code state had uncommitted changes

### Requirement: Command-line interface
The system SHALL expose each stage as a command-line subcommand:
- data fetch, build and frame;
- generate;
- classifier train;
- evaluate;
- annotate sample, app and import;
- analyze.

#### Scenario: Help lists stages
- **WHEN** the user runs the CLI with `--help`
- **THEN** all stage subcommands are listed with short descriptions

### Requirement: Sensitive data containment
The system SHALL keep raw benchmark data, model responses, judgments and trained model weights out of version control. Log output MUST NOT include full prompt or response text.

#### Scenario: Ignored data directories
- **WHEN** the project is initialized
- **THEN** the data, results and model directories are listed in the version-control ignore file

#### Scenario: Logs truncate content
- **WHEN** a stage logs information about a response
- **THEN** the log shows identifiers and status only, never the full response text
