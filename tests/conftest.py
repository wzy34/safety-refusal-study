import shutil
from pathlib import Path

import pytest

from srs.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A writable copy of the real configs and prompts inside tmp_path."""
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    if (ROOT / "prompts").exists():
        shutil.copytree(ROOT / "prompts", tmp_path / "prompts")
    return tmp_path / "configs"


@pytest.fixture
def cfg(config_dir: Path):
    c = load_config(config_dir)
    for pc in c.providers.values():  # keep fake-client tests fast; limiter has its own tests
        pc.requests_per_minute = 600_000
    return c


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Every test runs offline: any outbound socket connection fails loudly."""
    import socket

    def _blocked(*args, **kwargs):
        raise RuntimeError("network access attempted during tests")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
