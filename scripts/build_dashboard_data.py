#!/usr/bin/env python3
"""Build compact data for the local TAP dashboard."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tap.features import FEATURE_KEYS, feature_vector
from tap.predictor import CONTEXT_KEYS, load_labels


OUT = ROOT / "dashboard" / "public" / "tap_dashboard_data.json"

RUNS = [
    {
        "id": "doseresp_run1",
        "name": "Dose response run 1",
        "path": ROOT / "results" / "doseresp_run1" / "labels.jsonl",
        "domain": "math",
        "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    },
    {
        "id": "math_doseresp_n8",
        "name": "Archive: math dose response n=8",
        "path": ROOT / "results" / "labels_archive" / "math_doseresp_n8.jsonl",
        "domain": "math",
        "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    },
    {
        "id": "math_qwen25_n115",
        "name": "Archive: math Qwen2.5 n=115",
        "path": ROOT / "results" / "labels_archive" / "math_qwen25_n115.jsonl",
        "domain": "math",
        "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    },
    {
        "id": "science_qwen3_n56",
        "name": "Archive: science Qwen3 n=56",
        "path": ROOT / "results" / "labels_archive" / "science_qwen3_n56.jsonl",
        "domain": "science",
        "model": "Qwen/Qwen3-1.7B",
    },
]


PROMPT_SEARCH_ROOTS = (
    ROOT / "data",
    ROOT / "outputs",
    ROOT / "results",
)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def clean(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    return value


def row_target(row: dict[str, Any]) -> float | None:
    value = row.get("lift_nll")
    return float(value) if finite(value) else None


def candidate_id(row: dict[str, Any]) -> str:
    cohort = row.get("cohort") or {}
    return str(row.get("candidate_id") or cohort.get("name") or "unknown")


def load_prompt_index() -> dict[str, dict[str, Any]]:
    """Best-effort local prompt hydration from JSONL splits/caches."""

    index: dict[str, dict[str, Any]] = {}
    paths: list[Path] = []
    for root in PROMPT_SEARCH_ROOTS:
        if root.exists():
            paths.extend(root.rglob("*.jsonl"))
    for path in sorted(paths):
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rid = row.get("id")
                    prompt = row.get("question") or row.get("problem") or row.get("text")
                    if not rid or not prompt:
                        continue
                    index.setdefault(
                        str(rid),
                        {
                            "id": str(rid),
                            "prompt": str(prompt),
                            "answer": row.get("answer"),
                            "solution": row.get("solution"),
                            "source": row.get("source"),
                            "split": row.get("split"),
                            "path": str(path.relative_to(ROOT)),
                        },
                    )
        except OSError:
            continue
    return index


def compact_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    summary = row.get("reward_summary") or {}
    fields = (
        "reward_mean",
        "reward_std",
        "pass_rate",
        "group_passrate_mean",
        "group_passrate_std",
        "frac_nondegenerate",
        "frac_all_correct",
        "frac_all_wrong",
        "mean_logprob",
        "surprisal_mean",
        "redundancy_mean",
        "len_mean",
        "len_std",
        "n_groups",
        "n_rollouts",
    )
    return {
        "index": index,
        "chainId": row.get("chain_id"),
        "anchorIndex": row.get("anchor_index", row.get("state_index")),
        "seed": row.get("seed"),
        "liftAcc": row.get("lift_acc"),
        "liftNll": row.get("lift_nll"),
        "utility": row.get("utility"),
        "klDrift": row.get("kl_drift"),
        "klTrain": row.get("kl_train"),
        "meanReward": row.get("mean_reward"),
        "nContrib": row.get("n_contrib"),
        "rolloutCount": row.get("rollout_count"),
        "wallClockS": row.get("wall_clock_s"),
        "targetSimilarity": row.get("target_similarity"),
        "fingerprintNll": row.get("fingerprint_nll"),
        "fingerprintEntropy": row.get("fingerprint_entropy"),
        "rewardSummary": {k: summary.get(k) for k in fields if k in summary},
        "raw": row,
    }


def vectorize(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names = list(FEATURE_KEYS) + list(CONTEXT_KEYS)
    xs: list[list[float]] = []
    ys: list[float] = []
    for row in rows:
        y = row_target(row)
        if y is None:
            continue
        summary = row.get("reward_summary") or {}
        extra = {"target_similarity": row.get("target_similarity")}
        vec = list(feature_vector(summary, keys=FEATURE_KEYS, extra=extra))
        vec += [row.get(k) if row.get(k) is not None else float("nan") for k in CONTEXT_KEYS]
        xs.append(vec)
        ys.append(y)
    return np.asarray(xs, float), np.asarray(ys, float), names


def impute(X: np.ndarray, means: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if means is None:
        with np.errstate(invalid="ignore"):
            means = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
    return np.where(np.isfinite(X), X, means), means


def fit_ridge_snapshot(X: np.ndarray, y: np.ndarray, t: int, ridge_lambda: float = 1.0) -> dict[str, Any]:
    Xtr_raw = X[:t]
    ytr = y[:t]
    Xtr, imp = impute(Xtr_raw)
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd == 0] = 1.0
    Xs = (Xtr - mu) / sd
    base = float(ytr.mean())
    eye = np.eye(Xs.shape[1])
    w = np.linalg.solve(Xs.T @ Xs + ridge_lambda * eye, Xs.T @ (ytr - base))
    pred = Xs @ w + base
    rmse = float(np.sqrt(np.mean((pred - ytr) ** 2))) if len(ytr) else 0.0
    return {
        "t": t,
        "base": base,
        "imputeMeans": imp.tolist(),
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "weights": w.tolist(),
        "trainRmse": rmse,
    }


def build_run(spec: dict[str, Any]) -> dict[str, Any] | None:
    path = spec["path"]
    if not path.exists():
        return None
    raw_rows = load_labels(path)
    rows = [r for r in raw_rows if row_target(r) is not None]
    if not rows:
        return None

    X, y, names = vectorize(rows)
    prompt_index = load_prompt_index()
    by_candidate: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_candidate[candidate_id(row)].append(i)

    candidates = []
    for cid, idxs in sorted(by_candidate.items()):
        first = rows[idxs[0]]
        cohort = first.get("cohort") or {}
        meta = cohort.get("meta") or {}
        with np.errstate(invalid="ignore"):
            x_mean = np.nanmean(np.where(np.isfinite(X[idxs]), X[idxs], np.nan), axis=0)
        x_mean = np.where(np.isfinite(x_mean), x_mean, np.nan)
        y_vals = y[idxs]
        candidates.append(
            {
                "id": cid,
                "kind": cohort.get("kind") or "unknown",
                "noise": meta.get("label_noise_frac"),
                "size": cohort.get("size"),
                "meta": meta,
                "promptIds": cohort.get("prompt_ids") or [],
                "prompts": [
                    prompt_index.get(str(pid), {"id": str(pid), "prompt": None})
                    for pid in (cohort.get("prompt_ids") or [])
                ],
                "seeds": [rows[i].get("seed") for i in idxs],
                "rowIndices": idxs,
                "rows": [compact_row(rows[i], i) for i in idxs],
                "hasRawTrajectories": False,
                "trueMean": float(np.mean(y_vals)),
                "trueMin": float(np.min(y_vals)),
                "trueMax": float(np.max(y_vals)),
                "xMean": x_mean.tolist(),
                "nContribMean": float(np.mean([rows[i].get("n_contrib", 0.0) or 0.0 for i in idxs])),
                "klMean": float(np.mean([rows[i].get("kl_drift", 0.0) or 0.0 for i in idxs])),
                "targetSimilarity": first.get("target_similarity"),
            }
        )

    snapshots = [fit_ridge_snapshot(X, y, t) for t in range(1, len(y) + 1)]
    kind_counts: dict[str, int] = defaultdict(int)
    for c in candidates:
        kind_counts[c["kind"]] += 1

    return {
        "id": spec["id"],
        "name": spec["name"],
        "path": str(path.relative_to(ROOT)),
        "domain": spec["domain"],
        "model": spec["model"],
        "target": "lift_nll",
        "backend": "ridge",
        "features": names,
        "rowCount": len(rows),
        "rawRowCount": len(raw_rows),
        "candidateCount": len(candidates),
        "kindCounts": dict(sorted(kind_counts.items())),
        "candidates": candidates,
        "snapshots": snapshots,
    }


def main() -> None:
    runs = [r for r in (build_run(spec) for spec in RUNS) if r is not None]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(clean({"runs": runs}), indent=2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} with {len(runs)} runs")


if __name__ == "__main__":
    main()
