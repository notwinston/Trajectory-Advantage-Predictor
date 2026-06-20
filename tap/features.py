"""Pre-update GRPO features for a candidate batch (the predictor's inputs).

All features are computed from a **no-gradient rollout** of the *current* policy on
the candidate's prompts (available before deciding to train) plus cheap text stats
-- never from the post-update state (leakage guard).

Feature groups (and why):
* learnability   : reward variance, per-group pass-rate dist, ``frac_nondegenerate``
                   -- the GRPO signal (zero-variance groups give no gradient).
* familiarity    : mean log-prob / surprisal, entropy, log-prob quantiles,
                   early-vs-late confidence slope (TAP's fix for "redundancy = mean
                   token prob" -- these measure *familiarity*, not redundancy).
* redundancy     : ``redundancy_mean`` (mean token prob) kept for continuity, plus
                   ``target_similarity`` (cohort vs the common probe) which is the
                   single most relevant feature when forecasting a *specific* target.
* shape          : length stats.

Pure stdlib so it imports/tests on a laptop; the battery fills the rollout dicts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
from typing import Any, Iterable, Mapping, Sequence

_EPS = 1e-9


def _f(x: Any) -> float | None:
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _std(xs: Sequence[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def _mean(xs: Sequence[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


@dataclass(frozen=True)
class RolloutStats:
    n_rollouts: int
    n_groups: int
    # learnability
    reward_mean: float
    reward_std: float
    pass_rate: float
    group_passrate_mean: float | None
    group_passrate_std: float | None
    frac_nondegenerate: float | None
    frac_all_correct: float | None
    frac_all_wrong: float | None
    # familiarity / confidence
    mean_logprob: float | None
    surprisal_mean: float | None
    entropy_mean: float | None
    logprob_p10: float | None
    logprob_p50: float | None
    logprob_p90: float | None
    confidence_slope: float | None
    redundancy_mean: float | None
    min_logprob_mean: float | None
    # shape
    len_mean: float | None
    len_std: float | None
    len_max: float | None
    len_cv: float | None
    # advantage = GRPO signal strength. adv_std is the single strongest NLL-lift
    # predictor we have (~0.79, marginally > frac_nondegenerate; r=0.98 to it --
    # same latent axis "how much differentiated learning signal the cohort carries").
    adv_std: float | None
    adv_absmean: float | None
    # cheap curiosity / diversity (weak independent signal, recorded for completeness)
    reward_gap: float | None
    distinct_frac: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _q(xs: Sequence[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    idx = p * (len(s) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def summarize_rollouts(rollouts: Sequence[Mapping[str, Any]]) -> RolloutStats:
    """Aggregate per-completion rollout dicts into a candidate feature record.

    Expected (optional) per-rollout keys: ``group_id``, ``reward``, ``advantage``,
    ``completion_tokens``, ``mean_logprob``, ``mean_entropy``, ``early_logprob``,
    ``late_logprob``, ``min_logprob``, ``comp_hash``.
    """

    rewards = [v for v in (_f(r.get("reward")) for r in rollouts) if v is not None]
    lengths = [v for v in (_f(r.get("completion_tokens")) for r in rollouts) if v is not None]
    logps = [v for v in (_f(r.get("mean_logprob")) for r in rollouts) if v is not None]
    ents = [v for v in (_f(r.get("mean_entropy")) for r in rollouts) if v is not None]
    early = [v for v in (_f(r.get("early_logprob")) for r in rollouts) if v is not None]
    late = [v for v in (_f(r.get("late_logprob")) for r in rollouts) if v is not None]
    advs = [v for v in (_f(r.get("advantage")) for r in rollouts) if v is not None]
    mins = [v for v in (_f(r.get("min_logprob")) for r in rollouts) if v is not None]
    hashes = [r.get("comp_hash") for r in rollouts if r.get("comp_hash") is not None]

    reward_mean = _mean(rewards)
    pass_rate = _mean([1.0 if r > 0.5 else 0.0 for r in rewards]) if rewards else 0.0

    groups: dict[str, list[float]] = {}
    for r in rollouts:
        rw = _f(r.get("reward"))
        gid = r.get("group_id")
        if rw is None or gid is None:
            continue
        groups.setdefault(str(gid), []).append(rw)
    multi = {k: v for k, v in groups.items() if len(v) >= 2}

    gpm = gps = fnd = fac = faw = None
    if multi:
        prs = [_mean([1.0 if x > 0.5 else 0.0 for x in v]) for v in multi.values()]
        gpm, gps = _mean(prs), _std(prs)
        fnd = _mean([1.0 if _std(v) > _EPS else 0.0 for v in multi.values()])
        fac = _mean([1.0 if all(x > 0.5 for x in v) else 0.0 for v in multi.values()])
        faw = _mean([1.0 if all(x <= 0.5 for x in v) else 0.0 for v in multi.values()])

    mean_lp = _mean(logps) if logps else None
    slope = None
    if early and late and len(early) == len(late):
        slope = _mean([b - a for a, b in zip(early, late)])

    return RolloutStats(
        n_rollouts=len(rollouts),
        n_groups=len(groups),
        reward_mean=reward_mean,
        reward_std=_std(rewards) if rewards else 0.0,
        pass_rate=pass_rate,
        group_passrate_mean=gpm,
        group_passrate_std=gps,
        frac_nondegenerate=fnd,
        frac_all_correct=fac,
        frac_all_wrong=faw,
        mean_logprob=mean_lp,
        surprisal_mean=(-mean_lp) if mean_lp is not None else None,
        entropy_mean=_mean(ents) if ents else None,
        logprob_p10=_q(logps, 0.10),
        logprob_p50=_q(logps, 0.50),
        logprob_p90=_q(logps, 0.90),
        confidence_slope=slope,
        redundancy_mean=(_mean([math.exp(x) for x in logps]) if logps else None),
        min_logprob_mean=_mean(mins) if mins else None,
        len_mean=_mean(lengths) if lengths else None,
        len_std=_std(lengths) if lengths else None,
        len_max=max(lengths) if lengths else None,
        len_cv=(_std(lengths) / (_mean(lengths) + _EPS)) if lengths else None,
        adv_std=_std(advs) if advs else None,
        adv_absmean=_mean([abs(a) for a in advs]) if advs else None,
        reward_gap=(max(rewards) - min(rewards)) if rewards else None,
        distinct_frac=(len(set(hashes)) / len(hashes)) if hashes else None,
    )


# ---- target similarity (cohort vs the common probe) ---------------------------


def _unigrams(token_id_lists: Iterable[Sequence[int]]) -> dict[int, float]:
    counts: dict[int, float] = {}
    total = 0
    for ids in token_id_lists:
        for t in ids:
            counts[t] = counts.get(t, 0.0) + 1.0
            total += 1
    if total:
        for k in counts:
            counts[k] /= total
    return counts


def cosine(a: Mapping[int, float], b: Mapping[int, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b.get(k, 0.0) for k in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb + _EPS)


def target_similarity(cohort_token_lists: Iterable[Sequence[int]], probe_unigrams: Mapping[int, float]) -> float:
    """Unigram cosine between a cohort's prompts and the (fixed) common probe.

    Cheap, static, model-agnostic proxy for 'is this data relevant to the target'.
    """

    return cosine(_unigrams(cohort_token_lists), probe_unigrams)


# Canonical ordered feature names for the predictor (rollout-derived).
FEATURE_KEYS: tuple[str, ...] = (
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
    "entropy_mean",
    "logprob_p10",
    "logprob_p50",
    "logprob_p90",
    "confidence_slope",
    "redundancy_mean",
    "len_mean",
    "len_std",
    "target_similarity",
    "n_groups",
    # advantage-spread = GRPO signal strength; single strongest NLL-lift predictor
    # (within-domain spearman 0.92/0.68/0.78/0.68; ~ceiling for 3 of 4 domains).
    "adv_std",
    "adv_absmean",
    # weak but partially-orthogonal to adv_std in the headroom domain (codemmlu).
    "min_logprob_mean",
)


def feature_vector(
    stats: RolloutStats | Mapping[str, Any],
    *,
    keys: Sequence[str] = FEATURE_KEYS,
    extra: Mapping[str, Any] | None = None,
    missing: float = float("nan"),
) -> list[float]:
    """Flatten stats (+ optional extra scalars like ``target_similarity``)."""

    data = stats.as_dict() if isinstance(stats, RolloutStats) else dict(stats)
    if extra:
        data = {**data, **dict(extra)}
    out: list[float] = []
    for k in keys:
        v = data.get(k)
        out.append(missing if v is None else float(v))
    return out
