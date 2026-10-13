import json
import subprocess

from srs import config
from srs.manifest import RunManifest, git_state


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_manifest_has_no_secret_values(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_dotenv_loaded", True)
    monkeypatch.setenv("GEMINI_API_KEY", "sk-SUPERSECRET-123")
    monkeypatch.setenv("HF_TOKEN", "hf_SECRETTOKEN")
    with RunManifest("generate", cfg, args={"note": "key sk-SUPERSECRET-123"},
                     out_dir=tmp_path) as m:
        m.counts["responses"] = 3
    text = m.path.read_text()
    assert "SUPERSECRET" not in text and "SECRETTOKEN" not in text
    data = json.loads(text)
    assert data["run_id"] == m.run_id and data["counts"] == {"responses": 3}
    assert data["status"] == "ok" and data["started_at"] and data["ended_at"]
    assert data["config"]["providers"]["google"]["api_key_env"] == "GEMINI_API_KEY"
    assert {"commit", "uncommitted_changes"} <= set(data["code"])


def test_failed_run_still_writes(cfg, tmp_path):
    try:
        with RunManifest("evaluate", cfg, out_dir=tmp_path) as m:
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert json.loads(m.path.read_text())["status"] == "failed: RuntimeError"


def test_uncommitted_changes_flag(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
         "--allow-empty", "-m", "init")
    clean = git_state(tmp_path)
    assert clean["commit"] and clean["uncommitted_changes"] is False
    (tmp_path / "x.py").write_text("print(1)")
    assert git_state(tmp_path)["uncommitted_changes"] is True
