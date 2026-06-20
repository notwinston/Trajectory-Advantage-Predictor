"""TAP companion predictor: cheap GRPO features -> update utility.

Per the TAP critiques this is deliberately a **small, label-efficient** model
(monotone-constrained GBDT, or a numpy ridge fallback) -- NOT a 300k-param
attention net on ~100 labels. It is evaluated by **selection quality** (within-
anchor ranking, top-k lift) under **leave-one-chain-out**, against the baselines
that matter (difficulty-only, reward-only, redundancy-only, gradient-alignment,
best-single-feature), so we can tell whether the learned model earns its keep.

Inputs are strictly **pre-update** features (leakage guard). The label defaults to
held-out **accuracy** lift (the real target); ``nll``/``utility`` are alternates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Sequence
import warnings

import numpy as np

from tap.features import FEATURE_KEYS, feature_vector
from tap.labels import UtilityWeights, target_from_row
from tap import metrics as M


# Pre-update model-state context (TAP's "policy fingerprint", summarized).
CONTEXT_KEYS: tuple[str, ...] = ("fingerprint_nll", "fingerprint_entropy", "step_frac")

# Theory-aligned monotone priors (only clear, model-invariant directions).
FEATURE_MONOTONICITY: dict[str, int] = {
    "frac_nondegenerate": 1,    # more learnable groups -> more signal
    "frac_all_correct": -1,     # already solved
    "frac_all_wrong": -1,       # no foothold
    "reward_std": 1,            # within-batch reward variance = GRPO signal
    "redundancy_mean": -1,      # policy already confident -> little to learn
    "target_similarity": 1,     # closer to the target distribution -> more relevant
    "fingerprint_nll": 1,       # weaker current policy -> more room to improve
    "step_frac": -1,            # later in training -> diminishing returns
}


def monotone_vector(names: Sequence[str]) -> list[int]:
    return [FEATURE_MONOTONICITY.get(n, 0) for n in names]


# Domain-INVARIANT "mechanism" features (fractions / variances / similarity) — the
# subset expected to transfer across domains. Excludes raw, domain-scaled absolutes
# (length, raw log-prob/entropy/surprisal quantiles) per the transfer analysis.
MECHANISM_KEYS: tuple[str, ...] = (
    "reward_mean", "reward_std", "pass_rate", "group_passrate_mean", "group_passrate_std",
    "frac_nondegenerate", "frac_all_correct", "frac_all_wrong", "redundancy_mean",
    "target_similarity", "n_groups",
)


def _standardize_by_group(X: np.ndarray, groups: Sequence) -> np.ndarray:
    """Z-score each column WITHIN each group (domain) so GBDT thresholds become
    domain-relative -> far better cross-domain transfer (per the analysis)."""

    X = np.array(X, float)
    g = np.asarray(groups)
    for grp in np.unique(g):
        idx = g == grp
        sub = X[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mu = np.nanmean(np.where(np.isfinite(sub), sub, np.nan), axis=0)
            sd = np.nanstd(np.where(np.isfinite(sub), sub, np.nan), axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
        X[idx] = (sub - mu) / sd
    return X


def load_labels(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _anchor_group(row: dict[str, Any]) -> str:
    return f"{row.get('chain_id', 0)}::{row.get('anchor_index', row.get('state_index', 0))}"


def build_xy(
    rows: Sequence[dict[str, Any]],
    *,
    feature_keys: Sequence[str] = FEATURE_KEYS,
    with_context: bool = True,
    label: str = "acc",
    weights: UtilityWeights | None = None,
    standardize_domains: bool = False,
) -> dict[str, Any]:
    """Vectorize label rows. Returns X, y, anchor groups, chain ids, domains, names."""

    xs, ys, anchors, chains, domains = [], [], [], [], []
    for row in rows:
        y = target_from_row(row, mode=label, weights=weights)
        if y is None:
            continue
        summary = row.get("reward_summary") or {}
        extra = {"target_similarity": row.get("target_similarity")}
        vec = list(feature_vector(summary, keys=feature_keys, extra=extra))
        if with_context:
            vec += [row.get(k) if row.get(k) is not None else float("nan") for k in CONTEXT_KEYS]
        xs.append(vec)
        ys.append(y)
        anchors.append(_anchor_group(row))
        chains.append(row.get("chain_id", 0))
        domains.append(row.get("domain", "?"))
    if not xs:
        raise ValueError("no labelled rows with a usable target")
    names = list(feature_keys) + (list(CONTEXT_KEYS) if with_context else [])
    X = np.asarray(xs, float)
    if standardize_domains:  # competence/domain-relative features -> better transfer
        X = _standardize_by_group(X, domains)
    return {
        "X": X,
        "y": np.asarray(ys, float),
        "anchors": anchors,
        "chains": chains,
        "domains": domains,
        "names": names,
    }


def _impute(X: np.ndarray, means: np.ndarray | None = None):
    if means is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            means = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
    return np.where(np.isfinite(X), X, means), means


def _resolve_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    import importlib.util as u

    if u.find_spec("lightgbm") is not None:
        return "lightgbm"
    return "ridge"


@dataclass
class LiftPredictor:
    """Monotone GBDT (LightGBM) with a dependency-free ridge fallback."""

    backend: str = "auto"
    monotone: bool = True
    ridge_lambda: float = 1.0
    _resolved: str = field(default="", init=False)
    _model: Any = field(default=None, init=False)
    _means: np.ndarray | None = field(default=None, init=False)
    _names: list[str] | None = field(default=None, init=False)
    _X_fit: np.ndarray | None = field(default=None, init=False)
    _mu: np.ndarray | None = field(default=None, init=False)
    _sd: np.ndarray | None = field(default=None, init=False)
    _w: np.ndarray | None = field(default=None, init=False)
    _y_mean: float = field(default=0.0, init=False)
    _q: float | None = field(default=None, init=False)  # conformal half-width

    def fit(self, X, y, names: Sequence[str] | None = None) -> "LiftPredictor":
        self._resolved = _resolve_backend(self.backend)
        X, self._means = _impute(np.asarray(X, float))
        y = np.asarray(y, float)
        self._names = list(names) if names is not None else None
        self._X_fit = X
        if self._resolved == "lightgbm":
            import lightgbm as lgb

            params = dict(n_estimators=140, num_leaves=4, learning_rate=0.05,
                          subsample=0.6, colsample_bytree=0.6, min_child_samples=5,
                          reg_lambda=1.0, verbosity=-1)
            cons = monotone_vector(self._names) if (self.monotone and self._names) else None
            if cons and any(cons):
                params["monotone_constraints"] = cons
            self._model = lgb.LGBMRegressor(**params)
            self._model.fit(X, y)
        else:
            self._resolved = "ridge"
            self._mu = X.mean(0)
            sd = X.std(0); sd[sd == 0] = 1.0
            self._sd = sd
            Xs = (X - self._mu) / self._sd
            self._y_mean = float(y.mean())
            self._w = np.linalg.solve(Xs.T @ Xs + self.ridge_lambda * np.eye(Xs.shape[1]), Xs.T @ (y - self._y_mean))
        return self

    def predict(self, X) -> np.ndarray:
        if not self._resolved:
            raise RuntimeError("not fitted")
        X, _ = _impute(np.asarray(X, float), self._means)
        if self._resolved == "ridge":
            return ((X - self._mu) / self._sd) @ self._w + self._y_mean
        return np.asarray(self._model.predict(X), float)

    def calibrate(self, halfwidth: float) -> "LiftPredictor":
        """Store a conformal half-width (typically from OOF residuals)."""
        self._q = float(halfwidth)
        return self

    def predict_interval(self, X, *, halfwidth: float | None = None) -> dict[str, np.ndarray]:
        """Point prediction + conformal interval. ``below_resolution`` flags rows whose
        interval straddles 0 -> we honestly cannot call the sign of the lift (signal <
        noise). This is the right answer for a single trace / tiny cohort."""
        q = self._q if halfwidth is None else float(halfwidth)
        if q is None:
            raise RuntimeError("not calibrated; pass halfwidth or call calibrate()")
        p = self.predict(X)
        return {"point": p, "lower": p - q, "upper": p + q,
                "below_resolution": np.abs(p) < q, "halfwidth": np.full(len(p), q)}

    def explain(self, X=None, top_k: int | None = None) -> dict[str, Any]:
        names = self._names or [f"f{i}" for i in range(len(self._means or []))]
        if self._resolved == "ridge":
            imp, direction, method = np.abs(self._w), np.sign(self._w), "ridge_coef"
        else:
            imp, direction, method = self._tree_attr(self._X_fit if X is None else _impute(np.asarray(X, float), self._means)[0], len(names))
        order = np.argsort(imp)[::-1]
        feats = [{"feature": names[i], "importance": float(imp[i]), "direction": int(np.sign(direction[i]))} for i in order]
        return {"method": method, "features": feats[:top_k] if top_k else feats}

    def _tree_attr(self, X, n):
        import importlib.util as u

        if u.find_spec("shap") is not None:
            try:
                import shap

                sv = np.asarray(shap.TreeExplainer(self._model).shap_values(X), float)
                if sv.ndim == 1:
                    sv = sv.reshape(1, -1)
                return np.abs(sv).mean(0), sv.mean(0), "tree_shap"
            except Exception:
                pass
        imp = np.asarray(getattr(self._model, "feature_importances_", np.zeros(n)), float)
        direction = np.asarray(monotone_vector(self._names), float) if self._names else np.zeros(n)
        return imp, direction, "feature_importance"


# ---- cross-validation: out-of-fold predictions ---------------------------------


def _oof(X, y, fold_ids: Sequence, fit_predict: Callable) -> np.ndarray:
    """Out-of-fold predictions: each fold predicted by a model trained on the rest."""

    fold_ids = list(fold_ids)
    preds = np.full(len(y), np.nan)
    for fold in sorted(set(fold_ids), key=str):
        te = [i for i, f in enumerate(fold_ids) if f == fold]
        tr = [i for i, f in enumerate(fold_ids) if f != fold]
        if not tr or not te:
            continue
        preds[te] = np.asarray(fit_predict(X[tr], y[tr], X[te]), float)
    return preds


def _ridge_single(j: int):
    def fp(Xtr, ytr, Xte):
        return LiftPredictor(backend="ridge", monotone=False).fit(Xtr[:, [j]], ytr).predict(Xte[:, [j]])
    return fp


def conformal_summary(pred, y, *, alpha: float = 0.2) -> dict[str, Any]:
    """Split-conformal interval from OOF residuals (marginal coverage ~= 1-alpha).

    ``halfwidth`` is the finite-sample conformal radius q; an interval [p-q, p+q]
    covers the truth with prob ~1-alpha. ``indeterminate_sign_rate`` is the share of
    cohorts with |pred| < q (interval straddles 0) -> the model honestly cannot call
    the sign: this rises when the lift signal is below the measurement noise floor.
    """

    pred = np.asarray(pred, float); y = np.asarray(y, float)
    res = np.abs(pred - y)
    res = res[np.isfinite(res)]
    n = res.size
    if n == 0:
        return {}
    k = min(n, int(np.ceil((n + 1) * (1 - alpha))))  # conformal rank
    q = float(np.sort(res)[k - 1])
    return {
        "alpha": alpha,
        "halfwidth": q,
        "coverage": float(np.mean(res <= q)),
        "indeterminate_sign_rate": float(np.mean(np.abs(pred) < q)),
        "n": int(n),
    }


def _all_metrics(pred, y, anchors, deadband: float) -> dict[str, Any]:
    out = {
        "rmse": M.rmse(pred, y),
        "pearson": M.pearson(pred, y),
        "spearman": M.spearman(pred, y),
        "sign_accuracy": M.sign_accuracy(pred, y, deadband),
        "within_anchor_spearman": M.within_group_spearman(pred, y, anchors),
        "pairwise_ranking_accuracy": M.pairwise_ranking_accuracy(pred, y, anchors),
        "top1_regret": M.top1_regret(pred, y, anchors),
        "selection": M.selection_lift(pred, y, anchors, k_frac=0.25),
    }
    return out


def evaluate(
    rows: Sequence[dict[str, Any]],
    *,
    scheme: str = "logo",          # "logo" (leave-one-chain-out) | "loocv"
    backend: str = "auto",
    deadband: float = 0.0,
    with_context: bool = True,
    label: str = "acc",
    weights: UtilityWeights | None = None,
    explain: bool = True,
    alpha: float = 0.2,
) -> dict[str, Any]:
    """Headline eval: OOF predictions + selection metrics + baselines.

    ``logo`` folds by chain (the honest split when chains exist); ``loocv`` folds
    by row (for single-anchor batteries). Selection metrics group by anchor.
    """

    d = build_xy(rows, with_context=with_context, label=label, weights=weights)
    X, y, anchors, chains, names = d["X"], d["y"], d["anchors"], d["chains"], d["names"]
    n = len(y)
    if n < 3:
        raise ValueError(f"need >=3 labels, got {n}")

    if scheme == "logo" and len(set(chains)) >= 2:
        folds = chains
    else:
        scheme = "loocv"
        folds = list(range(n))

    def model_fp(Xtr, ytr, Xte):
        return LiftPredictor(backend=backend, monotone=True).fit(Xtr, ytr, names=names).predict(Xte)

    pred = _oof(X, y, folds, model_fp)
    ok = np.isfinite(pred)
    result = _all_metrics(pred[ok], y[ok], [a for a, m in zip(anchors, ok) if m], deadband)
    result.update({"scheme": scheme, "n": int(n), "n_features": int(X.shape[1]),
                   "backend": _resolve_backend(backend), "label": label})
    result["conformal"] = conformal_summary(pred[ok], y[ok], alpha=alpha)

    # Baselines under the identical folds.
    baselines: dict[str, Any] = {}
    mean_pred = _oof(X, y, folds, lambda a, b, c: np.full(len(c), float(np.mean(b))))
    baselines["predict_mean"] = _all_metrics(mean_pred[ok], y[ok], [a for a, m in zip(anchors, ok) if m], deadband)
    for bname, feat in (("difficulty_only", "pass_rate"), ("reward_only", "reward_mean"),
                        ("redundancy_only", "redundancy_mean"), ("gradient_alignment", "grad_align")):
        if feat in names:
            j = names.index(feat)
            if np.nanstd(X[:, j]) == 0:
                continue
            bp = _oof(X, y, folds, _ridge_single(j))
            baselines[bname] = _all_metrics(bp[ok], y[ok], [a for a, m in zip(anchors, ok) if m], deadband)
    # best single feature by within-anchor ranking (the selection-relevant criterion)
    best = {"feature": None, "within_anchor_spearman": -2.0}
    for j, nm in enumerate(names):
        if np.nanstd(X[:, j]) == 0:
            continue
        bp = _oof(X, y, folds, _ridge_single(j))
        s = M.within_group_spearman(bp[ok], y[ok], [a for a, m in zip(anchors, ok) if m])
        if s > best["within_anchor_spearman"]:
            best = {"feature": nm, "within_anchor_spearman": s}
    baselines["best_single_feature"] = best
    result["baselines"] = baselines
    result["beats_difficulty"] = bool(
        result["within_anchor_spearman"] >= baselines.get("difficulty_only", {}).get("within_anchor_spearman", -2)
    )
    result["beats_best_single_feature"] = bool(result["within_anchor_spearman"] >= best["within_anchor_spearman"])

    if explain:
        Xi, _ = _impute(X)
        result["explanation"] = LiftPredictor(backend=backend).fit(Xi, y, names=names).explain(Xi, top_k=10)
    return result


def evaluate_transfer(train_rows, test_rows, *, backend="auto", deadband=0.0,
                      with_context=True, label="acc") -> dict[str, Any]:
    """Fit on one model's labels, predict another's (the reusability test)."""

    a = build_xy(train_rows, with_context=with_context, label=label)
    b = build_xy(test_rows, with_context=with_context, label=label)
    model = LiftPredictor(backend=backend).fit(a["X"], a["y"], names=a["names"])
    pred = model.predict(b["X"])
    out = _all_metrics(pred, b["y"], b["anchors"], deadband)
    out.update({"mode": "transfer", "n_train": len(a["y"]), "n_test": len(b["y"])})
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--transfer-labels", type=Path, default=None)
    p.add_argument("--scheme", choices=("logo", "loocv"), default="logo")
    p.add_argument("--backend", default="auto")
    p.add_argument("--label", choices=("acc", "nll", "utility"), default="acc")
    p.add_argument("--deadband", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=0.2, help="conformal miscoverage (0.2 => 80%% interval)")
    p.add_argument("--no-context", action="store_true")
    p.add_argument("--no-explain", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_labels(args.labels)
    report = {
        "eval": evaluate(rows, scheme=args.scheme, backend=args.backend, deadband=args.deadband,
                         with_context=not args.no_context, label=args.label, explain=not args.no_explain,
                         alpha=args.alpha)
    }
    if args.transfer_labels is not None:
        report["transfer"] = evaluate_transfer(rows, load_labels(args.transfer_labels),
                                               backend=args.backend, deadband=args.deadband,
                                               with_context=not args.no_context, label=args.label)
    print(json.dumps(report, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
