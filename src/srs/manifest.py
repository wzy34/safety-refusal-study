"""Run manifests: what produced a set of outputs (config snapshot, code version, counts)."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from srs.config import PROJECT_ROOT, StudyConfig, secret_values
from srs.schemas import utcnow


def new_run_id(command: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{command.replace(' ', '-')}-{uuid.uuid4().hex[:6]}"


def git_state(repo: Path = PROJECT_ROOT) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                                 timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "uncommitted_changes": bool(status) if status is not None else None}


def scrub(obj: Any, secrets: list[str]) -> Any:
    """Replace any secret value appearing in strings with '***'."""
    if isinstance(obj, dict):
        return {k: scrub(v, secrets) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [scrub(v, secrets) for v in obj]
    if isinstance(obj, str):
        for s in secrets:
            obj = obj.replace(s, "***")
    return obj


@dataclass
class RunManifest:
    """Use as a context manager; the manifest is written on exit (status ok or failed)."""

    command: str
    cfg: StudyConfig
    args: dict[str, Any] = field(default_factory=dict)
    out_dir: Path | None = None
    run_id: str = ""
    counts: dict[str, int] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    path: Path | None = None

    def __post_init__(self) -> None:
        self.run_id = self.run_id or new_run_id(self.command)
        self.out_dir = self.out_dir or self.cfg.path("results_dir") / "manifests"

    def __enter__(self) -> RunManifest:
        self.started_at = utcnow()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        status = "ok" if exc_type is None else f"failed: {exc_type.__name__}"
        self.write(status)

    def to_dict(self, status: str) -> dict[str, Any]:
        data = {
            "run_id": self.run_id,
            "command": self.command,
            "argv": sys.argv,
            "args": self.args,
            "status": status,
            "started_at": self.started_at,
            "ended_at": utcnow(),
            "code": git_state(self.cfg.root),
            "counts": self.counts,
            "extra": self.extra,
            "config": self.cfg.snapshot(),
        }
        return scrub(json.loads(json.dumps(data, default=str)), secret_values())

    def write(self, status: str = "ok") -> Path:
        assert self.out_dir is not None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / f"{self.run_id}.json"
        self.path.write_text(json.dumps(self.to_dict(status), indent=2, sort_keys=True))
        return self.path
