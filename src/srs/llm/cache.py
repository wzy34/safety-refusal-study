"""SQLite request cache (design D4): request hash -> request, final payload, status, time.

Only final results (`ok`, `api_blocked`) are stored, so errors are retried on the next run.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from srs.ids import canonical_json, sha256_hex
from srs.schemas import utcnow

CACHEABLE = ("ok", "api_blocked")


def request_hash(provider: str, body: dict[str, Any], extra_body: dict[str, Any],
                 salt: str | None = None) -> str:
    """SHA-256 over the full request: provider, model, messages and every parameter. `salt`
    separates intentionally repeated identical requests (e.g. stability samples)."""
    key: dict[str, Any] = {"provider": provider, "body": body, "extra_body": extra_body}
    if salt is not None:
        key["salt"] = salt
    return sha256_hex(key)


class RequestCache:
    def __init__(self, path: Path | str):
        self.path = Path(path) if str(path) != ":memory:" else path
        if isinstance(self.path, Path):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS requests ("
            " hash TEXT PRIMARY KEY, request TEXT NOT NULL, payload TEXT NOT NULL,"
            " status TEXT NOT NULL, created_at TEXT NOT NULL)"
        )

    def get(self, key: str) -> dict[str, Any] | None:
        row = self._db.execute("SELECT payload, status FROM requests WHERE hash = ?",
                               (key,)).fetchone()
        if row is None:
            return None
        return {"payload": json.loads(row[0]), "status": row[1]}

    def put(self, key: str, request: dict[str, Any], payload: dict[str, Any], status: str) -> None:
        if status not in CACHEABLE:
            raise ValueError(f"only {CACHEABLE} results are cached, got {status!r}")
        self._db.execute(
            "INSERT OR REPLACE INTO requests (hash, request, payload, status, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (key, canonical_json(request), canonical_json(payload), status, utcnow()),
        )

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]

    def close(self) -> None:
        self._db.close()
