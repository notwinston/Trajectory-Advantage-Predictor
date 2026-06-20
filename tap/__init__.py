"""TAP v1 — Trajectory Advantage Predictor (Wave 0 foundation: schema + synth).

This package holds the frozen Parquet schema contract (:mod:`tap.schema`) and a
synthetic data generator (:mod:`tap.synth`) that the parallel engine/model waves
build against. Heavy/optional imports (torch, transformers, peft) must live
inside functions so ``py_compile`` and unittest work on CPU-only ARM64 without
GPU libraries installed.
"""

__all__: list[str] = []
