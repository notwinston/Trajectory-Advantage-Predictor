"""Run layout helpers for TAP v1 Prime jobs."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
from pathlib import Path


SUBDIRS = (
    "data",
    "probes",
    "checkpoints/chains",
    "checkpoints/branches",
    "fragments",
    "parquet",
    "models",
    "reports",
    "logs",
    "hf_cache",
    "journals",
)


def default_run_id(prefix: str = "tap_v1") -> str:
    now = dt.datetime.now(dt.timezone.utc)
    return f"{prefix}_{now:%Y%m%d_%H%M%S}"


@dataclass(frozen=True)
class TapRunLayout:
    run_id: str
    remote_root: Path
    local_root: Path

    def ensure_local(self) -> None:
        for subdir in SUBDIRS:
            (self.local_root / subdir).mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.local_root / "manifest.json"


def build_layout(
    *,
    run_id: str,
    local_root: Path,
    remote_base: Path = Path("/mnt/prime_tap/tap_runs"),
) -> TapRunLayout:
    return TapRunLayout(run_id=run_id, remote_root=remote_base / run_id, local_root=local_root)

