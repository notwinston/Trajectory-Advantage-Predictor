"""End-to-end TAP v1 evaluation + report generation.

    uv run python -m tap.run_all --parquet-dir outputs/tap_synth_72 --out outputs/tap_report

Loads the four Parquet files, evaluates TAP + its three ablations + every baseline
over both leave-one-chain-out directions, and writes ``results.csv`` plus
``report.{md,tex,pdf}`` into ``--out``. Synthetic data is the default source; pass
``--parquet-dir`` to point at a real collection. Exits 0 even if TAP ties or loses
(that outcome is reported, per spec).

Pure CPU. torch is used only if available (SmallTAP); otherwise the TAP_NO_TORCH
sklearn fallback is selected automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from tap import eval as E
from tap import report as R
from tap.baselines import make_baselines
from tap.dataset import TapData, chain_splits, load_parquets
from tap.model import SmallTAP, torch_available
from tap.train import make_tap_models

# Result family labels (for the table's "family" column).
_TAP_NAMES = {"tap", "tap-no-prob", "tap-no-grad", "tap-no-history"}
_LEARNED = {"ridge", "gbt", "no_history_mlp", "numeric_only", "candidate_only"}


def _family(name: str) -> str:
    if name == "tap":
        return "tap"
    if name in _TAP_NAMES:
        return "tap-ablation"
    if name in _LEARNED:
        return "learned-baseline"
    return "heuristic"


def _build_models(seed: int, epochs: int) -> Dict:
    models = {}
    models.update(make_tap_models(seed=seed, epochs=epochs))
    models.update(make_baselines(seed=seed))
    return models


def evaluate_all(
    data: TapData, parquet_dir: str, seed: int = 0, epochs: int = 300
) -> Tuple[pd.DataFrame, Dict]:
    """Run every model over both directions and assemble the results table."""
    splits = chain_splits(data)
    per_model = {name: [] for name in R.MODEL_ORDER}

    for s in splits:
        truth = E.build_truth(s.test)
        models = _build_models(seed, epochs)  # fresh per direction (no state leak)
        for name in R.MODEL_ORDER:
            model = models[name]
            if getattr(model, "trainable", False):
                model.fit(data.states, s.train, data.history)
            scores = model.score(data.states, s.test, data.history)
            per_model[name].append(E.evaluate(scores, truth))

    averaged = {name: E.average_directions(per_model[name]) for name in R.MODEL_ORDER}
    reference_mtu = {
        "random": averaged["random"]["mean_true_utility"],
        "reward": averaged["reward_mean"]["mean_true_utility"],
        "prob": averaged["geo_mean_prob"]["mean_true_utility"],
    }

    rows = []
    for name in R.MODEL_ORDER:
        metrics = E.add_lift(averaged[name], reference_mtu)
        rows.append({"model": name, "family": _family(name), **{c: metrics[c] for c in R.METRIC_COLS}})
    results = pd.DataFrame(rows, columns=["model", "family", *R.METRIC_COLS])

    tap_metrics = E.add_lift(averaged["tap"], reference_mtu)
    tap_label = make_tap_models(seed=seed, epochs=epochs)["tap"].label
    analysis = {
        "parquet_dir": parquet_dir,
        "n_labels": int(data.n_labels),
        "n_states": int(data.states.shape[0]),
        "seed": seed,
        "epochs": epochs,
        "tap_label": tap_label,
        "tap_num_params": SmallTAP().num_params() if torch_available() else None,
        "tap_mtu": tap_metrics["mean_true_utility"],
        "mtu_random": reference_mtu["random"],
        "mtu_reward": reference_mtu["reward"],
        "mtu_prob": reference_mtu["prob"],
        "beat_random": tap_metrics["lift_random"] > 0,
        "beat_reward": tap_metrics["lift_reward"] > 0,
        "beat_prob": tap_metrics["lift_prob"] > 0,
    }
    return results, analysis


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="TAP v1 end-to-end eval + report.")
    parser.add_argument("--parquet-dir", default="outputs/tap_synth_72")
    parser.add_argument("--out", default="outputs/tap_report")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args(argv)

    data = load_parquets(args.parquet_dir)
    results, analysis = evaluate_all(data, args.parquet_dir, seed=args.seed, epochs=args.epochs)
    paths = R.write_all(args.out, results, analysis)

    print(f"Wrote {len(results)} model rows to {Path(args.out)}/results.csv")
    print(f"PDF engine: {paths['pdf_engine']}")
    print(
        "TAP vs selectors (mean true utility): "
        f"TAP={analysis['tap_mtu']:.3f} random={analysis['mtu_random']:.3f} "
        f"reward-only={analysis['mtu_reward']:.3f} prob-only={analysis['mtu_prob']:.3f}"
    )
    verdict = (
        f"beat_random={analysis['beat_random']} beat_reward={analysis['beat_reward']} "
        f"beat_prob={analysis['beat_prob']}"
    )
    print("Verdict:", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
