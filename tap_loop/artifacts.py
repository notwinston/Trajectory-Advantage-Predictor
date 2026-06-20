"""Atomic TAP fragment writing and optional Parquet compaction."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
from typing import Any, Iterable


VECTOR_DIMS = {
    "policy_fingerprint": 16,
    "candidate_embedding": 256,
    "trajectory_embedding": 256,
    "gradient_sketch": 64,
    "historical_candidate_embedding": 256,
    "historical_gradient_sketch": 64,
}


class ParquetUnavailableError(RuntimeError):
    pass


def assert_finite(value: Any, *, path: str = "value") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} is not finite: {value}")
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite(item, path=f"{path}[{index}]")
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, path=f"{path}.{key}")


def validate_vector_dims(row: dict[str, Any]) -> None:
    for key, dim in VECTOR_DIMS.items():
        if key in row and row[key] is not None and len(row[key]) != dim:
            raise ValueError(f"{key} must have dimension {dim}, got {len(row[key])}")


def validate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    for row in output:
        validate_vector_dims(row)
        assert_finite(row)
    return output


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class TapArtifactWriter:
    """Write append-friendly fragments and compact them into final tables."""

    def __init__(self, run_root: Path, *, require_parquet: bool = False):
        self.run_root = run_root
        self.fragment_root = run_root / "fragments"
        self.parquet_root = run_root / "parquet"
        self.journal_root = run_root / "journals"
        self.require_parquet = require_parquet

    def write_fragment(
        self,
        table: str,
        rows: Iterable[dict[str, Any]],
        *,
        fragment_id: str,
    ) -> Path:
        validated = validate_rows(rows)
        path = self.fragment_root / table / f"{fragment_id}.jsonl"
        text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in validated)
        atomic_write_text(path, text)
        append_jsonl(
            self.journal_root / f"{table}.jsonl",
            {"table": table, "fragment_id": fragment_id, "path": str(path), "rows": len(validated)},
        )
        return path

    def read_table_fragments(self, table: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        table_root = self.fragment_root / table
        if not table_root.exists():
            return rows
        for path in sorted(table_root.glob("*.jsonl")):
            rows.extend(read_jsonl(path))
        return rows

    def compact_table(self, table: str) -> Path:
        rows = validate_rows(self.read_table_fragments(table))
        self.parquet_root.mkdir(parents=True, exist_ok=True)
        parquet_path = self.parquet_root / f"{table}.parquet"
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as exc:
            if self.require_parquet:
                raise ParquetUnavailableError("pyarrow is required to write TAP Parquet outputs") from exc
            fallback = self.parquet_root / f"{table}.jsonl"
            atomic_write_text(
                fallback,
                "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            )
            return fallback

        table_obj = pa.Table.from_pylist(rows)
        tmp_path = parquet_path.with_suffix(".parquet.tmp")
        pq.write_table(table_obj, tmp_path)
        tmp_path.replace(parquet_path)
        return parquet_path

    def compact_all(self, tables: Iterable[str] = ("states", "trajectories", "candidates", "history")) -> dict[str, Path]:
        return {table: self.compact_table(table) for table in tables}

