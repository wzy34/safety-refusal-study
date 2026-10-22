## Purpose

Provides one reliable way to send chat requests to the Gemini, DeepSeek, Qwen and GLM APIs. It reports provider blocks, errors and model output faithfully, and never pays twice for an identical request.

## ADDED Requirements

### Requirement: Unified chat completion across providers
The client SHALL accept a model configuration and a list of chat messages, and return a completion record containing:
- final text,
- reasoning text (if any),
- finish reason,
- token usage,
- latency,
- a status of `ok`, `api_blocked` or `error`,
- the raw provider response.

#### Scenario: Successful completion
- **WHEN** a provider returns a normal response
- **THEN** the record has status `ok`, the response text, the finish reason, token usage and the raw response

#### Scenario: Reasoning content separated
- **WHEN** a provider returns reasoning content alongside the answer
- **THEN** the reasoning appears in the reasoning text field and is excluded from the final text

### Requirement: Per-model request adaptation
The client SHALL build each request according to the model's declared capabilities:
- which parameter name sets the output token limit,
- whether temperature and seed are supported,
- which structured-output mode is available.

It MUST NOT send parameters a model declares unsupported.

#### Scenario: Temperature unsupported
- **WHEN** a model's configuration declares that temperature is unsupported
- **THEN** the request omits temperature
- **AND** the completion record notes that the configured temperature was not applied

#### Scenario: Structured output fallback
- **WHEN** a judge requests structured output from a model that supports only a generic JSON mode
- **THEN** the request uses that JSON mode and the response is validated against the expected schema

### Requirement: Provider rate limiting
The client SHALL enforce a configured maximum number of simultaneous requests, and a configured request rate, for each provider separately.

#### Scenario: Concurrency capped
- **WHEN** more requests are submitted than a provider's concurrency limit allows
- **THEN** no more than that number are in flight to that provider at once

### Requirement: Error classification and retries
The client SHALL classify every provider error as one of:
- retryable (rate limit, timeout, server error),
- content-filter block,
- fatal (for example, an invalid request or authentication failure).

It SHALL retry retryable errors with exponential backoff, up to a configured limit.

#### Scenario: Rate limit retried
- **WHEN** a provider responds with a rate-limit error and a later attempt succeeds
- **THEN** the returned record has status `ok`

#### Scenario: Content filter recorded as block
- **WHEN** a provider rejects a request, or truncates a response, because of its content filter
- **THEN** the record has status `api_blocked` with the provider's error code or finish reason
- **AND** the request is not retried

#### Scenario: Finish reason with a detail suffix
- **WHEN** a provider reports a finish reason such as `content_filter: PROHIBITED_CONTENT` (Gemini)
- **THEN** it is matched against the provider's blocked finish reasons by the part before the `:`, and the full string is kept as the block code

#### Scenario: Retries exhausted
- **WHEN** a retryable error persists beyond the retry limit
- **THEN** the record has status `error` with the last error details, and it is not cached as a final result

### Requirement: Request cache
The client SHALL cache final results (`ok` and `api_blocked`) under a hash of the full request: provider, model, messages and parameters. When an identical request is repeated, it SHALL return the cached result without contacting the provider.

#### Scenario: Cache hit
- **WHEN** an identical request is issued a second time
- **THEN** the cached record is returned and no network call is made

#### Scenario: Parameter change misses cache
- **WHEN** a request differs from a cached one only in max output tokens
- **THEN** the cache is not used and a new call is made

#### Scenario: Interrupted run resumes
- **WHEN** a run is interrupted and restarted
- **THEN** requests completed before the interruption are served from the cache

### Requirement: Offline test double
The system SHALL provide a fake client with the same interface. It returns scripted completions, blocks and errors without network access, for use in automated tests.

#### Scenario: Tests run offline
- **WHEN** the test suite runs without network access or API keys
- **THEN** all pipeline tests that use the fake client pass
