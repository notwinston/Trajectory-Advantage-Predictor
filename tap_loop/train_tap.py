"""Train/evaluate TAP v1 baselines and the small TAP torch model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tap_loop.artifacts import read_jsonl
from tap_loop.training import NUMERIC_FEATURES, evaluate_baselines


def load_table_rows(run_root: Path, table: str) -> list[dict[str, Any]]:
    parquet_path = run_root / "parquet" / f"{table}.parquet"
    jsonl_path = run_root / "parquet" / f"{table}.jsonl"
    if parquet_path.exists():
        try:
            import pyarrow.parquet as pq

            return [dict(row) for row in pq.read_table(parquet_path).to_pylist()]
        except Exception:
            if not jsonl_path.exists():
                raise
    return read_jsonl(jsonl_path)


def load_candidate_rows(run_root: Path) -> list[dict[str, Any]]:
    return load_table_rows(run_root, "candidates")


def _history_tensor_for_state(history_rows: list[dict[str, Any]]) -> list[list[float]]:
    records = sorted(history_rows, key=lambda row: int(row.get("history_position", 0)))[:4]
    output: list[list[float]] = []
    for row in records:
        numeric = [
            float(row.get("relative_age", 0.0) or 0.0),
            float(row.get("historical_reward_mean", 0.0) or 0.0),
            float(row.get("historical_advantage_mean", 0.0) or 0.0),
            float(row.get("historical_mean_log_probability", 0.0) or 0.0),
            float(row.get("historical_mean_entropy", 0.0) or 0.0),
            float(row.get("historical_update_norm", 0.0) or 0.0),
            float(row.get("historical_training_loss_change", 0.0) or 0.0),
            float(row.get("historical_candidate_log_probability_change", 0.0) or 0.0),
        ]
        output.append(
            list(row.get("historical_candidate_embedding", [0.0] * 256))
            + list(row.get("historical_gradient_sketch", [0.0] * 64))
            + numeric
        )
    while len(output) < 4:
        output.append([0.0] * (256 + 64 + 8))
    return output


def _state_groups(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(str(row["state_id"]), []).append(index)
    return groups


def train_small_tap_if_available(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    states: list[dict[str, Any]] | None = None,
    history_rows: list[dict[str, Any]] | None = None,
    epochs: int = 80,
    learning_rate: float = 2e-3,
    seed: int = 1729,
) -> dict[str, float | str]:
    try:
        import torch
        from torch import nn

        from tap_loop.training import build_tap_model
    except Exception as exc:
        return {"status": "skipped", "reason": f"torch unavailable: {exc}"}

    if len(rows) < 4:
        return {"status": "skipped", "reason": "not enough candidate rows"}

    state_by_id = {str(row["state_id"]): row for row in states or []}
    history_by_state: dict[str, list[dict[str, Any]]] = {}
    for row in history_rows or []:
        history_by_state.setdefault(str(row["state_id"]), []).append(row)

    torch.manual_seed(seed)
    numeric = torch.tensor(
        [[float(row.get(feature, 0.0) or 0.0) for feature in NUMERIC_FEATURES] for row in rows],
        dtype=torch.float32,
    )
    mean = numeric.mean(dim=0, keepdim=True)
    std = numeric.std(dim=0, keepdim=True)
    std = torch.where(std == 0, torch.ones_like(std), std)
    numeric = (numeric - mean) / std
    state_fingerprint = torch.tensor(
        [state_by_id.get(str(row["state_id"]), {}).get("policy_fingerprint", [0.0] * 16) for row in rows],
        dtype=torch.float32,
    )
    candidate_embedding = torch.tensor([row["candidate_embedding"] for row in rows], dtype=torch.float32)
    gradient_sketch = torch.tensor([row["gradient_sketch"] for row in rows], dtype=torch.float32)
    history = torch.tensor(
        [_history_tensor_for_state(history_by_state.get(str(row["state_id"]), [])) for row in rows],
        dtype=torch.float32,
    )
    target = torch.tensor([float(row["utility_points"]) for row in rows], dtype=torch.float32)
    target_mean = target.mean()
    target_std = target.std()
    if float(target_std) == 0.0:
        target_std = torch.tensor(1.0)
    target_z = (target - target_mean) / target_std

    model = build_tap_model(numeric_dim=len(NUMERIC_FEATURES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    huber = nn.HuberLoss()
    groups = _state_groups(rows)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        pred = model(numeric, state_fingerprint, candidate_embedding, gradient_sketch, history)
        loss = huber(pred, target_z)
        ranking_losses = []
        for indices in groups.values():
            if len(indices) < 2:
                continue
            group_pred = pred[indices]
            group_target = target_z[indices]
            pair_losses = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    sign = torch.sign(group_target[i] - group_target[j])
                    if float(sign) != 0.0:
                        pair_losses.append(torch.logaddexp(torch.tensor(0.0), -(group_pred[i] - group_pred[j]) * sign))
            if pair_losses:
                ranking_losses.append(torch.stack(pair_losses).mean())
        if ranking_losses:
            loss = loss + 0.5 * torch.stack(ranking_losses).mean()
        loss.backward()
        optimizer.step()

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "tap_small.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "numeric_mean": mean,
            "numeric_std": std,
            "target_mean": target_mean,
            "target_std": target_std,
            "numeric_features": NUMERIC_FEATURES,
        },
        model_path,
    )
    with torch.no_grad():
        final_loss = float(huber(model(numeric, state_fingerprint, candidate_embedding, gradient_sketch, history), target_z))
    return {"status": "trained", "model_path": str(model_path), "huber_loss": final_loss}


def run_training(run_root: Path) -> dict[str, Any]:
    rows = load_candidate_rows(run_root)
    if not rows:
        raise ValueError(f"no candidate rows found under {run_root / 'parquet'}")
    states = load_table_rows(run_root, "states")
    history = load_table_rows(run_root, "history")
    reports = run_root / "reports"
    models = run_root / "models"
    reports.mkdir(parents=True, exist_ok=True)
    result = {
        "candidate_rows": len(rows),
        "state_rows": len(states),
        "history_rows": len(history),
        "baselines": evaluate_baselines(rows),
        "small_tap": train_small_tap_if_available(rows, models, states=states, history_rows=history),
    }
    report_path = reports / "tap_metrics.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(json.dumps(run_training(args.run_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
