"""Parquet read/write helpers for the record tables in `srs.schemas`."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from srs.schemas import Record


def write_records(path: Path, records: Iterable[Record]) -> int:
    """Atomically write records to Parquet (replacing the file). Returns the row count."""
    rows = [r.to_row() for r in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), tmp)
    os.replace(tmp, path)
    return len(rows)


def read_records[R: Record](path: Path, cls: type[R]) -> list[R]:
    if not path.exists():
        return []
    return [cls.model_validate(row) for row in pq.read_table(path).to_pylist()]


def read_frame(path: Path) -> pd.DataFrame:
    """Read a table as a DataFrame (JSON fields stay as strings)."""
    if not path.exists():
        return pd.DataFrame()
    return pq.read_table(path).to_pandas()


def upsert_records[R: Record](path: Path, cls: type[R], new: Iterable[R],
                               key: Callable[[R], object]) -> int:
    """Merge `new` into the table at `path`, replacing rows with the same key. Returns the
    total row count."""
    merged = {key(r): r for r in read_records(path, cls)}
    for r in new:
        merged[key(r)] = r
    return write_records(path, merged.values())
