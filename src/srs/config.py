"""Typed study configuration loaded from `configs/*.yaml`.

Each file is validated against its own pydantic model (unknown fields are errors), then
cross-checked: every model names a defined provider, every evaluator/framing role names a
defined model, and every taxonomy mapping targets a defined category.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"

FRAMINGS = ("direct", "academic", "roleplay", "hypothetical", "multiturn")
Framing = Literal["direct", "academic", "roleplay", "hypothetical", "multiturn"]
PartialRule = Literal["partial_as_refusal", "partial_as_compliance"]


class ConfigError(Exception):
    """Configuration is missing, malformed or inconsistent."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- providers.yaml -------------------------------------------------------------------------


class ProviderConfig(_Strict):
    # Wire protocol: OpenAI-compatible Chat Completions, or Anthropic's native Messages API.
    api: Literal["openai", "anthropic"] = "openai"
    base_url: str
    api_key_env: str
    max_concurrency: int = Field(8, ge=1)
    requests_per_minute: float = Field(300, gt=0)
    timeout_s: float = Field(120, gt=0)
    max_retries: int = Field(6, ge=0)
    retry_status_codes: list[int] = [408, 409, 429, 500, 502, 503, 504]
    # Content-filter detection. Codes/patterns are confirmed during the pilot and kept here as
    # data, not in code.
    blocked_status_codes: list[int] = []
    blocked_error_codes: list[str] = []
    blocked_message_patterns: list[str] = []
    blocked_finish_reasons: list[str] = ["content_filter"]


class ProvidersFile(_Strict):
    providers: dict[str, ProviderConfig]


# --- models.yaml ----------------------------------------------------------------------------


class ModelConfig(_Strict):
    provider: str
    model: str
    family: str
    token_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    supports_temperature: bool = True
    supports_seed: bool = False
    structured_output: Literal["json_schema", "json_object", "none"] = "none"
    is_reasoning: bool = False
    extra_body: dict = {}
    price_in_per_mtok: float | None = None
    price_out_per_mtok: float | None = None

    @property
    def is_placeholder(self) -> bool:
        return self.model.startswith("SET_ME")


class ModelsFile(_Strict):
    models: dict[str, ModelConfig]
    evaluated: list[str]


# --- evaluators.yaml ------------------------------------------------------------------------


class KeywordEvaluatorConfig(_Strict):
    kind: Literal["keyword"]
    phrase_list: Literal["zou2023", "extended"] = "zou2023"


class ClassifierEvaluatorConfig(_Strict):
    kind: Literal["classifier"]
    base_model: str = "roberta-base"
    checkpoint_dir: str = "models/roberta-refusal"
    train_dataset: str = "allenai/wildguardmix"
    train_config: str = "wildguardtrain"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_epochs: int = 3
    patience: int = 1
    val_fraction: float = Field(0.1, gt=0, lt=1)


class SafetyModelEvaluatorConfig(_Strict):
    kind: Literal["safety_model"]
    hf_repo: str = "allenai/wildguard"
    mlx_path: str = "models/wildguard-mlx-8bit"
    q_bits: int = 8
    max_new_tokens: int = 32


class LLMJudgeEvaluatorConfig(_Strict):
    kind: Literal["llm_judge"]
    model: str
    rubric: str = "judge_rubric.md"
    max_parse_retries: int = Field(2, ge=0)
    max_tokens: int = 1024


EvaluatorConfig = Annotated[
    KeywordEvaluatorConfig
    | ClassifierEvaluatorConfig
    | SafetyModelEvaluatorConfig
    | LLMJudgeEvaluatorConfig,
    Field(discriminator="kind"),
]


class EvaluatorsFile(_Strict):
    evaluators: dict[str, EvaluatorConfig]


# --- taxonomy.yaml --------------------------------------------------------------------------


class TierRule(_Strict):
    """Marks prompts `borderline` when they come from `source` and (optionally) one of the
    listed source categories or expected behaviors."""

    source: str
    source_categories: list[str] | None = None
    expected_behavior: Literal["refuse", "comply"] | None = None
    reason: str = ""


class TaxonomyFile(_Strict):
    categories: dict[str, str]
    # source -> source category -> unified category
    source_maps: dict[str, dict[str, str]] = {}
    # sources whose prompts carry no usable risk category; categories come from an LLM
    llm_assigned_sources: list[str] = []
    category_model: str | None = None
    # manual fixes for prompts the LLM cannot categorize: "source/source_id" -> category
    category_overrides: dict[str, str] = {}
    tier_rules: list[TierRule] = []
    tier_screen_model: str | None = None
    tier_screen_sources: list[str] = []


# --- framings.yaml --------------------------------------------------------------------------


class FramingTemplate(_Strict):
    wrapper: str | None = None
    turns: list[str] | None = None
    rewrite_prompt: str | None = None

    @model_validator(mode="after")
    def _one_form(self) -> FramingTemplate:
        if (self.wrapper is None) == (self.turns is None):
            raise ValueError("exactly one of 'wrapper' or 'turns' must be set")
        return self


class FramingsFile(_Strict):
    rewriter_model: str
    checker_model: str
    max_attempts: int = Field(3, ge=1)
    intent_check_prompt: str = "intent_check.md"
    templates: dict[Literal["academic", "roleplay", "hypothetical", "multiturn"], FramingTemplate]

    @model_validator(mode="after")
    def _multiturn_three(self) -> FramingsFile:
        mt = self.templates.get("multiturn")
        if mt is not None and (mt.turns is None or len(mt.turns) != 3):
            raise ValueError("templates.multiturn must define exactly 3 turns")
        missing = {"academic", "roleplay", "hypothetical", "multiturn"} - set(self.templates)
        if missing:
            raise ValueError(f"templates missing framings: {sorted(missing)}")
        return self


# --- experiment.yaml ------------------------------------------------------------------------


class GenerationSettings(_Strict):
    temperature: float = 0.0
    max_tokens: int = Field(1024, ge=1024)
    # Thinking tokens count toward max_tokens on these APIs, so models with is_reasoning get
    # their own (larger) limit; None means they use max_tokens too.
    reasoning_max_tokens: int | None = Field(None, ge=1024)
    seed: int | None = None
    system_prompt: str | None = None


class StabilitySettings(_Strict):
    n_samples: int = Field(5, ge=1)
    temperature: float = Field(0.7, gt=0)
    n_intents: int = Field(40, ge=0)
    split: str = "main"


class SplitSettings(_Strict):
    pilot_intents: int = Field(50, ge=0)
    rq3_intents: int = Field(200, ge=0)
    main_intents: int | None = None  # None: every remaining intent


class AnnotationSettings(_Strict):
    sample_size: int = Field(350, ge=1)
    pilot_calibration_size: int = Field(100, ge=1)
    pass2_min_days: float = Field(7, ge=0)
    disagreement_oversample: float = Field(3.0, ge=1)


class AnalysisSettings(_Strict):
    primary_evaluator: str
    primary_evaluator_rationale: str | None = None
    primary_rule: PartialRule = "partial_as_refusal"
    include_borderline: bool = False
    # Splits pooled for the primary (direct-framing) analyses; the pilot stays out of the study
    # results unless asked for (`srs analyze --splits pilot`).
    primary_splits: list[Literal["pilot", "main", "rq3"]] = ["main", "rq3"]
    bootstrap_resamples: int = Field(2000, ge=100)
    ci_level: float = Field(0.95, gt=0, lt=1)


class PathSettings(_Strict):
    data_dir: str = "data"
    results_dir: str = "results"
    models_dir: str = "models"
    prompts_dir: str = "prompts"
    cache_path: str = "data/cache/requests.sqlite"


class ExperimentFile(_Strict):
    seed: int
    generation: GenerationSettings = GenerationSettings()
    stability: StabilitySettings = StabilitySettings()
    splits: SplitSettings = SplitSettings()
    annotation: AnnotationSettings = AnnotationSettings()
    analysis: AnalysisSettings
    paths: PathSettings = PathSettings()


# --- aggregate ------------------------------------------------------------------------------


class StudyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: Path
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    evaluated: list[str]
    evaluators: dict[str, EvaluatorConfig]
    taxonomy: TaxonomyFile
    framings: FramingsFile
    experiment: ExperimentFile

    def path(self, name: Literal["data_dir", "results_dir", "models_dir", "prompts_dir",
                                 "cache_path"]) -> Path:
        return self.root / getattr(self.experiment.paths, name)

    def provider_for(self, model_id: str) -> ProviderConfig:
        return self.providers[self.models[model_id].provider]

    def snapshot(self) -> dict:
        """JSON-able config for manifests. Holds env-var *names*, never secret values."""
        return self.model_dump(mode="json", exclude={"root"})


_FILES: dict[str, type[BaseModel]] = {
    "providers.yaml": ProvidersFile,
    "models.yaml": ModelsFile,
    "evaluators.yaml": EvaluatorsFile,
    "taxonomy.yaml": TaxonomyFile,
    "framings.yaml": FramingsFile,
    "experiment.yaml": ExperimentFile,
}


def _format_validation(fname: str, err: ValidationError) -> str:
    lines = [f"{fname}: invalid configuration"]
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "<root>"
        lines.append(f"  {fname}: field '{loc}': {e['msg']}")
    return "\n".join(lines)


def _load_file(config_dir: Path, fname: str) -> BaseModel:
    path = config_dir / fname
    if not path.exists():
        raise ConfigError(f"{fname}: file not found at {path}")
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{fname}: YAML parse error: {e}") from e
    try:
        return _FILES[fname].model_validate(raw)
    except ValidationError as e:
        raise ConfigError(_format_validation(fname, e)) from e


def _cross_check(cfg: StudyConfig) -> None:
    errors: list[str] = []
    for mid, m in cfg.models.items():
        if m.provider not in cfg.providers:
            errors.append(f"models.yaml: model '{mid}' names unknown provider '{m.provider}'")
        elif cfg.providers[m.provider].api == "anthropic":
            if m.structured_output == "json_object":
                errors.append(f"models.yaml: model '{mid}' is on the Anthropic API, which has "
                              "no json_object mode; use json_schema or none")
            if m.token_param != "max_tokens" or m.supports_seed:
                errors.append(f"models.yaml: model '{mid}' is on the Anthropic API, which takes "
                              "max_tokens and no seed")
    for mid in cfg.evaluated:
        if mid not in cfg.models:
            errors.append(f"models.yaml: evaluated model '{mid}' is not defined under models")
    for eid, ev in cfg.evaluators.items():
        if isinstance(ev, LLMJudgeEvaluatorConfig) and ev.model not in cfg.models:
            errors.append(f"evaluators.yaml: evaluator '{eid}' names unknown model '{ev.model}'")
    fr = cfg.framings
    for role in ("rewriter_model", "checker_model"):
        if getattr(fr, role) not in cfg.models:
            errors.append(f"framings.yaml: {role} '{getattr(fr, role)}' is not a defined model")
    if fr.rewriter_model in cfg.models and fr.checker_model in cfg.models:
        if cfg.models[fr.rewriter_model].family == cfg.models[fr.checker_model].family:
            errors.append("framings.yaml: checker_model must come from a different family "
                          "than rewriter_model")
    tx = cfg.taxonomy
    for role in ("category_model", "tier_screen_model"):
        mid = getattr(tx, role)
        if mid is not None and mid not in cfg.models:
            errors.append(f"taxonomy.yaml: {role} '{mid}' is not a defined model")
    for src, table in tx.source_maps.items():
        for sc, uc in table.items():
            if uc not in tx.categories:
                errors.append(f"taxonomy.yaml: source_maps.{src}.{sc} -> unknown category '{uc}'")
    for key, uc in tx.category_overrides.items():
        if uc not in tx.categories:
            errors.append(f"taxonomy.yaml: category_overrides.{key} -> unknown category '{uc}'")
    pe = cfg.experiment.analysis.primary_evaluator
    if pe not in cfg.evaluators:
        errors.append(f"experiment.yaml: analysis.primary_evaluator '{pe}' is not defined")
    if errors:
        raise ConfigError("\n".join(errors))


def load_config(config_dir: Path | str | None = None) -> StudyConfig:
    config_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    parsed = {fname: _load_file(config_dir, fname) for fname in _FILES}
    models_file: ModelsFile = parsed["models.yaml"]  # type: ignore[assignment]
    cfg = StudyConfig(
        root=config_dir.resolve().parent,
        providers=parsed["providers.yaml"].providers,  # type: ignore[attr-defined]
        models=models_file.models,
        evaluated=models_file.evaluated,
        evaluators=parsed["evaluators.yaml"].evaluators,  # type: ignore[attr-defined]
        taxonomy=parsed["taxonomy.yaml"],  # type: ignore[arg-type]
        framings=parsed["framings.yaml"],  # type: ignore[arg-type]
        experiment=parsed["experiment.yaml"],  # type: ignore[arg-type]
    )
    _cross_check(cfg)
    return cfg


# --- secrets --------------------------------------------------------------------------------

SECRET_ENV_VARS = ("GEMINI_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "ZHIPU_API_KEY",
                   "HF_TOKEN")


class MissingSecretError(ConfigError):
    """A required API key or token is not set in the environment or `.env`."""


_dotenv_loaded = False


def load_dotenv_once(path: Path | None = None) -> None:
    """Load `.env` from the project root into os.environ, never overriding set variables."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    from dotenv import load_dotenv

    load_dotenv(path or PROJECT_ROOT / ".env", override=False)
    _dotenv_loaded = True


def get_secret(var: str) -> str:
    import os

    load_dotenv_once()
    value = os.environ.get(var, "").strip()
    if not value:
        raise MissingSecretError(
            f"Missing secret: environment variable {var} is not set. "
            f"Add it to .env (see .env.example) or export it."
        )
    return value


def require_provider_keys(cfg: StudyConfig, model_ids: list[str]) -> None:
    """Fail before any request if a provider needed by `model_ids` has no API key."""
    needed = sorted({cfg.provider_for(m).api_key_env for m in model_ids})
    for var in needed:
        get_secret(var)


def secret_values() -> list[str]:
    """Currently set secret values, used to scrub anything written to disk."""
    import os

    load_dotenv_once()
    return [v for k in SECRET_ENV_VARS if len(v := os.environ.get(k, "").strip()) >= 4]
