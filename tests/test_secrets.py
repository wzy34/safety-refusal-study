import pytest

from srs import config
from srs.config import MissingSecretError, get_secret, require_provider_keys


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    monkeypatch.setattr(config, "_dotenv_loaded", True)


def test_missing_key_names_variable(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(MissingSecretError, match="DEEPSEEK_API_KEY"):
        get_secret("DEEPSEEK_API_KEY")


def test_present_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test-123")
    assert get_secret("GEMINI_API_KEY") == "sk-test-123"


def test_require_provider_keys(cfg, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test")
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    require_provider_keys(cfg, ["gemini-a", "gemini-b"])
    with pytest.raises(MissingSecretError, match="ZHIPU_API_KEY"):
        require_provider_keys(cfg, ["gemini-a", "glm-a"])


def test_dotenv_does_not_override(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_dotenv_loaded", False)
    env = tmp_path / ".env"
    env.write_text("MOONSHOT_API_KEY=from-file\nHF_TOKEN=hf-from-file\n")
    monkeypatch.setenv("MOONSHOT_API_KEY", "from-env")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    config.load_dotenv_once(env)
    assert get_secret("MOONSHOT_API_KEY") == "from-env"
    assert get_secret("HF_TOKEN") == "hf-from-file"
