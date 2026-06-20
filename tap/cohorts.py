"""Cohort construction for the TAP battery.

A *cohort* is a fixed-size set of training prompt ids whose one-update lift we
measure. Each cohort becomes one TAP example. Cohort kinds:

* ``random``              -- volume / uncontrolled variation.
* ``passband``           -- difficulty bands by baseline pass-rate (for *this* model).
* ``variance_decoupled`` -- the decisive pair: equal mean pass-rate, different
                            within-group variance (does lift track variance, not
                            difficulty?).
* ``by_level`` / ``by_subject`` -- MATH metadata slices (also gives on/off-target).
* ``label_noise``        -- a fraction of prompts get a CORRUPTED gold answer, so
                            the verifier reward is poisoned. The RL-native "bad
                            data" axis (the head should down-rank these).
* ``duplication``        -- a few prompts repeated to fill the cohort (high
                            redundancy; little new signal).

Pure stdlib so it imports and unit-tests on a laptop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Cohort:
    name: str
    kind: str
    prompt_ids: tuple[str, ...]
    meta: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "prompt_ids": list(self.prompt_ids),
                "size": len(self.prompt_ids), "meta": dict(self.meta)}

    @staticmethod
    def from_json(row: Mapping[str, Any]) -> "Cohort":
        return Cohort(row["name"], row["kind"], tuple(row["prompt_ids"]), row.get("meta", {}))


def _ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    ids = [str(r["id"]) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("rows contain duplicate ids")
    return ids


def _chunk(items: Sequence[str], size: int) -> list[tuple[str, ...]]:
    return [tuple(items[i : i + size]) for i in range(0, len(items) - size + 1, size)]


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def cohort_pass_stats(cohort: Cohort, pass_rates: Mapping[str, float]) -> dict[str, float]:
    rates = [float(pass_rates[p]) for p in cohort.prompt_ids if p in pass_rates]
    m = _mean(rates)
    return {
        "passrate_mean": m,                                  # difficulty
        "passrate_var": _mean((r - m) ** 2 for r in rates),  # across-prompt spread
        "within_group_var_mean": _mean(r * (1 - r) for r in rates),  # GRPO signal
        "n_with_rate": float(len(rates)),
    }


def random_cohorts(rows, *, n_cohorts: int, size: int, seed: int = 1729, prefix: str = "random") -> list[Cohort]:
    if size <= 0 or n_cohorts <= 0:
        raise ValueError("size and n_cohorts must be positive")
    ids = _ids(rows)
    if len(ids) < n_cohorts * size:
        raise ValueError(f"need {n_cohorts * size} prompts, have {len(ids)}")
    rng = random.Random(seed)
    sh = ids[:]
    rng.shuffle(sh)
    return [Cohort(f"{prefix}-{i:03d}", "random", c) for i, c in enumerate(_chunk(sh, size)[:n_cohorts])]


def grouped_cohorts(rows, *, key: str, size: int, seed: int = 1729, max_per_group: int | None = None) -> list[Cohort]:
    if size <= 0:
        raise ValueError("size must be positive")
    buckets: dict[str, list[str]] = {}
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        buckets.setdefault(str(v), []).append(str(r["id"]))
    out: list[Cohort] = []
    for v in sorted(buckets):
        ids = buckets[v]
        random.Random(f"{seed}:{v}").shuffle(ids)
        chunks = _chunk(ids, size)
        if max_per_group is not None:
            chunks = chunks[:max_per_group]
        for i, c in enumerate(chunks):
            safe = str(v).replace(" ", "_").replace("/", "-")
            out.append(Cohort(f"by_{key}-{safe}-{i:03d}", f"by_{key}", c, {key: v}))
    return out


def passband_cohorts(rows, pass_rates, *, size: int, seed: int = 1729,
                     bands=((0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0)),
                     max_per_band: int | None = None) -> list[Cohort]:
    if size <= 0:
        raise ValueError("size must be positive")
    ids = [str(r["id"]) for r in rows if str(r["id"]) in pass_rates]
    out: list[Cohort] = []
    for lo, hi in bands:
        in_band = [p for p in ids if lo <= pass_rates[p] < hi or (hi >= 1.0 and pass_rates[p] == 1.0)]
        random.Random(f"{seed}:{lo}:{hi}").shuffle(in_band)
        chunks = _chunk(in_band, size)
        if max_per_band is not None:
            chunks = chunks[:max_per_band]
        for i, c in enumerate(chunks):
            base = Cohort(f"passband-{lo:.1f}_{hi:.1f}-{i:03d}", "passband", c, {"band": [lo, hi]})
            out.append(Cohort(base.name, base.kind, base.prompt_ids, {**base.meta, **cohort_pass_stats(base, pass_rates)}))
    return out


def variance_decoupled_pair(rows, pass_rates, *, size: int, seed: int = 1729,
                            mid_tol: float = 0.15, low: float = 0.15, high: float = 0.85) -> list[Cohort]:
    """Equal mean pass-rate, opposite within-group variance (the decisive test)."""

    if size <= 0 or size % 2 != 0:
        raise ValueError("size must be a positive even number")
    half = size // 2
    ids = [str(r["id"]) for r in rows if str(r["id"]) in pass_rates]
    mid = [p for p in ids if abs(pass_rates[p] - 0.5) <= mid_tol]
    lows = [p for p in ids if pass_rates[p] <= low]
    highs = [p for p in ids if pass_rates[p] >= high]
    if len(mid) < size or len(lows) < half or len(highs) < half:
        raise ValueError(f"insufficient prompts: mid={len(mid)} low={len(lows)} high={len(highs)}")
    rng = random.Random(seed)
    rng.shuffle(mid); rng.shuffle(lows); rng.shuffle(highs)
    hv = Cohort("vardecouple-highvar", "variance_decoupled", tuple(mid[:size]), {"design": "mid_passrate"})
    lv = Cohort("vardecouple-lowvar", "variance_decoupled", tuple(lows[:half] + highs[:half]),
                {"design": "half_impossible_half_trivial"})
    return [Cohort(c.name, c.kind, c.prompt_ids, {**c.meta, **cohort_pass_stats(c, pass_rates)}) for c in (hv, lv)]


def label_noise_cohorts(rows, *, size: int, fracs=(0.0, 0.25, 0.5, 1.0), seed: int = 1729,
                        max_per_frac: int = 2) -> list[Cohort]:
    """Cohorts with a fraction of prompts whose gold answer is corrupted.

    The battery poisons the verifier reward for ``noisy_ids`` (treats a wrong
    answer as the target), the RL analog of corrupted training data.
    """

    ids = _ids(rows)
    rng = random.Random(seed)
    sh = ids[:]
    rng.shuffle(sh)
    chunks = _chunk(sh, size)
    out: list[Cohort] = []
    ci = 0
    # round-robin over fracs (one of each, then seconds...) so a small --max-cohorts
    # still spans the full 0->1 dose-response range (incl. the all-poison endpoint).
    for rep in range(max_per_frac):
        for frac in fracs:
            if ci >= len(chunks):
                break
            chunk = chunks[ci]; ci += 1
            k = int(round(frac * size))
            noisy = list(chunk[:k])
            out.append(Cohort(f"labelnoise-{int(frac*100):03d}-{ci:03d}", "label_noise", chunk,
                              {"label_noise_frac": frac, "noisy_ids": noisy}))
    return out


def duplication_cohorts(rows, *, size: int, ks=(1, 2, 4, 8), seed: int = 1729, max_per_k: int = 2) -> list[Cohort]:
    """High-redundancy cohorts: ``size`` slots filled by repeating ``k`` distinct prompts."""

    ids = _ids(rows)
    rng = random.Random(seed)
    out: list[Cohort] = []
    for k in ks:
        for j in range(max_per_k):
            pool = ids[:]
            random.Random(f"{seed}:{k}:{j}").shuffle(pool)
            if len(pool) < k:
                continue
            base = pool[:k]
            filled = tuple(base[i % k] for i in range(size))
            out.append(Cohort(f"dup-k{k:02d}-{j:03d}", "duplication", filled, {"dup_k": k}))
    return out


def build_all_cohorts(rows, *, size: int, pass_rates: Mapping[str, float] | None = None,
                      n_random: int = 8, seed: int = 1729, max_per_group: int = 3,
                      max_per_band: int = 3) -> list[Cohort]:
    out: list[Cohort] = []
    # Most lift-DIFFERENTIATING cohorts first, so a small --max-cohorts stays
    # diverse (and gives a built-in ground truth: lift should fall as quality drops).
    out += label_noise_cohorts(rows, size=size, seed=seed)   # data-quality dose-response
    out += duplication_cohorts(rows, size=size, seed=seed)    # redundancy
    if pass_rates:
        out += passband_cohorts(rows, pass_rates, size=size, seed=seed, max_per_band=max_per_band)
        try:
            out += variance_decoupled_pair(rows, pass_rates, size=size, seed=seed)
        except ValueError:
            pass
    out += grouped_cohorts(rows, key="level", size=size, seed=seed, max_per_group=max_per_group)
    out += grouped_cohorts(rows, key="subject", size=size, seed=seed, max_per_group=max_per_group)
    try:
        out += random_cohorts(rows, n_cohorts=n_random, size=size, seed=seed)
    except ValueError:
        pass
    return out


def write_cohorts(path: Path, cohorts: Iterable[Cohort]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        for c in cohorts:
            h.write(json.dumps(c.to_json(), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def read_cohorts(path: Path) -> list[Cohort]:
    out: list[Cohort] = []
    with Path(path).open("r", encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if line:
                out.append(Cohort.from_json(json.loads(line)))
    return out
