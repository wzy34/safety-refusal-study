from pathlib import Path

import pytest
import yaml

from srs.config import ConfigError, load_config


def _edit(path: Path, fn) -> None:
    data = yaml.safe_load(path.read_text())
    fn(data)
    path.write_text(yaml.safe_dump(data))


def test_real_configs_load(cfg, config_dir):
    assert set(cfg.providers) == {"google", "deepseek", "alibaba", "zhipu", "anthropic"}
    assert cfg.providers["anthropic"].api == "anthropic" and cfg.providers["google"].api == "openai"
    assert cfg.models[cfg.evaluators["judge_a"].model].family not in {
        cfg.models[m].family for m in cfg.evaluated}
    assert len(cfg.evaluated) == 8
    assert {cfg.models[m].family for m in cfg.evaluated} == {"gemini", "deepseek", "qwen", "glm"}
    assert {"keyword", "classic", "wildguard", "judge_a", "judge_b"} <= set(cfg.evaluators)
    assert 8 <= len(cfg.taxonomy.categories) <= 10
    assert cfg.experiment.generation.max_tokens == 1024
    assert cfg.experiment.splits.pilot_intents == 50
    assert cfg.root == config_dir.parent


def test_unknown_field_rejected(config_dir):
    _edit(config_dir / "experiment.yaml", lambda d: d["generation"].update(temprature=0.5))
    with pytest.raises(ConfigError) as e:
        load_config(config_dir)
    assert "experiment.yaml" in str(e.value) and "generation.temprature" in str(e.value)


def test_wrong_type_rejected(config_dir):
    _edit(config_dir / "providers.yaml",
          lambda d: d["providers"]["google"].update(max_concurrency="lots"))
    with pytest.raises(ConfigError) as e:
        load_config(config_dir)
    assert "providers.yaml" in str(e.value) and "providers.google.max_concurrency" in str(e.value)


def test_missing_required_field(config_dir):
    _edit(config_dir / "models.yaml", lambda d: d["models"]["gemini-a"].pop("provider"))
    with pytest.raises(ConfigError) as e:
        load_config(config_dir)
    assert "models.yaml" in str(e.value) and "models.gemini-a.provider" in str(e.value)


def test_unknown_provider(config_dir):
    _edit(config_dir / "models.yaml", lambda d: d["models"]["qwen-a"].update(provider="nope"))
    with pytest.raises(ConfigError) as e:
        load_config(config_dir)
    assert "'qwen-a'" in str(e.value) and "'nope'" in str(e.value)


def test_max_tokens_floor(config_dir):
    _edit(config_dir / "experiment.yaml", lambda d: d["generation"].update(max_tokens=256))
    with pytest.raises(ConfigError):
        load_config(config_dir)


def test_same_family_checker_rejected(config_dir):
    _edit(config_dir / "framings.yaml", lambda d: d.update(checker_model="gemini-a"))
    with pytest.raises(ConfigError, match="different family"):
        load_config(config_dir)


def test_taxonomy_target_must_exist(config_dir):
    _edit(config_dir / "taxonomy.yaml",
          lambda d: d["source_maps"]["harmbench"].update(illegal="not_a_category"))
    with pytest.raises(ConfigError, match="not_a_category"):
        load_config(config_dir)


def test_anthropic_model_rejects_json_object(config_dir):
    _edit(config_dir / "models.yaml",
          lambda d: d["models"]["judge-claude"].update(structured_output="json_object"))
    with pytest.raises(ConfigError) as e:
        load_config(config_dir)
    assert "judge-claude" in str(e.value) and "json_object" in str(e.value)
