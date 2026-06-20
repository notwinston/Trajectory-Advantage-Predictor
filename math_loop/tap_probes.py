"""TAP v1 probe math: NLL, per-token entropy, generic incremental-KL, fingerprint.

The pure aggregation helpers (:func:`softmax`, :func:`entropy_from_logits`,
:func:`token_nll_from_logprobs`, :func:`sequence_kl`,
:func:`assemble_policy_fingerprint`) use only the stdlib ``math`` module so this
file imports with no numpy/torch present and is unit-testable on CPU.

The model-driven probe runners (:func:`teacher_forced_probe_nll`,
:func:`generic_incremental_kl`, :func:`compute_policy_fingerprint`) defer all
``torch``/``transformers`` imports inside the function body — they execute only
on the Wave 2 GPU pod, never during Wave 1 CPU validation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

from math_loop.data import FINGERPRINT_PROBE_SIZE


# --- pure aggregation (stdlib only) -----------------------------------------

def softmax(logits: Sequence[float]) -> list[float]:
    """Numerically stable softmax over a 1-D logit row."""
    if not logits:
        return []
    hi = max(logits)
    exps = [math.exp(value - hi) for value in logits]
    total = sum(exps)
    if total <= 0.0:
        n = len(logits)
        return [1.0 / n] * n
    return [value / total for value in exps]


def entropy_from_logits(logits: Sequence[float]) -> float:
    """Shannon entropy (nats) of the softmax distribution over ``logits``."""
    probs = softmax(logits)
    acc = 0.0
    for p in probs:
        if p > 0.0:
            acc -= p * math.log(p)
    return acc


def sequence_entropy(logits_2d: Sequence[Sequence[float]]) -> list[float]:
    """Per-token entropy for a [T, V] logit matrix."""
    return [entropy_from_logits(row) for row in logits_2d]


def token_nll_from_logprobs(token_logprobs: Sequence[float]) -> float:
    """Mean negative log-likelihood (nats/token) from realized-token logprobs."""
    if not token_logprobs:
        return 0.0
    return -sum(token_logprobs) / len(token_logprobs)


def nll_from_logits_and_targets(
    logits_2d: Sequence[Sequence[float]], target_ids: Sequence[int]
) -> float:
    """Teacher-forced NLL (nats/token) given per-step logits and target ids."""
    if not target_ids:
        return 0.0
    total = 0.0
    count = 0
    for logits, target in zip(logits_2d, target_ids):
        probs = softmax(logits)
        if 0 <= target < len(probs):
            p = probs[target]
            total -= math.log(p if p > 0.0 else 1e-12)
            count += 1
    return total / max(count, 1)


def kl_divergence(p_logits: Sequence[float], q_logits: Sequence[float]) -> float:
    """KL(softmax(p) || softmax(q)) in nats for one token position."""
    p = softmax(p_logits)
    q = softmax(q_logits)
    acc = 0.0
    for pi, qi in zip(p, q):
        if pi > 0.0:
            acc += pi * (math.log(pi) - math.log(qi if qi > 0.0 else 1e-12))
    return acc


def sequence_kl(
    base_logits_2d: Sequence[Sequence[float]], branch_logits_2d: Sequence[Sequence[float]]
) -> float:
    """Mean per-token KL(base || branch) over a generic-prompt sequence."""
    rows = list(zip(base_logits_2d, branch_logits_2d))
    if not rows:
        return 0.0
    return sum(kl_divergence(base, branch) for base, branch in rows) / len(rows)


def assemble_policy_fingerprint(
    nll_values: Sequence[float], entropy_values: Sequence[float], *, size: int = FINGERPRINT_PROBE_SIZE
) -> list[float]:
    """16-value fingerprint = ``size`` NLLs followed by ``size`` entropies.

    Pads with 0.0 (and truncates) so the result is exactly ``2 * size`` long,
    matching the frozen ``policy_fingerprint`` width (16).
    """
    nll = list(nll_values[:size]) + [0.0] * max(0, size - len(nll_values))
    ent = list(entropy_values[:size]) + [0.0] * max(0, size - len(entropy_values))
    return [float(v) for v in (nll[:size] + ent[:size])]


def percentiles(values: Sequence[float], points: Sequence[float] = (10, 50, 90)) -> list[float]:
    """Linear-interpolation percentiles using only the stdlib (no numpy)."""
    if not values:
        return [0.0 for _ in points]
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    out: list[float] = []
    for q in points:
        if n == 1:
            out.append(ordered[0])
            continue
        rank = (q / 100.0) * (n - 1)
        low = int(math.floor(rank))
        high = min(low + 1, n - 1)
        frac = rank - low
        out.append(ordered[low] * (1.0 - frac) + ordered[high] * frac)
    return out


# --- model-driven runners (Wave 2; torch deferred) --------------------------

def teacher_forced_probe_nll(
    checkpoint: Path,
    probe_rows: Sequence[dict[str, Any]],
    *,
    model_name: str = "Qwen/Qwen3-8B",
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_length: int = 4096,
) -> float:
    """Teacher-forced NLL (nats/non-pad token) over ``probe_rows`` on the GPU pod.

    Thin wrapper over :func:`math_loop.probe_loss.compute_probe_loss` writing the
    rows to a temporary split. Torch lives entirely inside ``compute_probe_loss``.
    """
    import json
    import tempfile

    from math_loop.probe_loss import compute_probe_loss

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as handle:
        for row in probe_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        split_path = Path(handle.name)
    result = compute_probe_loss(
        checkpoint,
        split_path,
        model_name=model_name,
        max_length=max_length,
        device=device,
        dtype=dtype,
    )
    return float(result.nll)


def generic_incremental_kl(
    base_checkpoint: Path,
    branch_checkpoint: Path,
    generic_prompts: Sequence[dict[str, Any]],
    *,
    model_name: str = "Qwen/Qwen3-8B",
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_new_tokens: int = 64,
) -> float:
    """Mean per-token KL(base || branch) on the generic drift prompts (Wave 2).

    Generates continuations under the base policy, then compares the per-token
    next-token distributions of base vs branch via :func:`sequence_kl`. All
    torch/transformers usage is local to this function.
    """
    import torch  # noqa: F401  (deferred — GPU only)

    from math_loop.probe_loss import load_model_and_tokenizer

    base_model, tokenizer = load_model_and_tokenizer(
        base_checkpoint, model_name=model_name, device=device, dtype=dtype
    )
    branch_model, _ = load_model_and_tokenizer(
        branch_checkpoint, model_name=model_name, device=device, dtype=dtype
    )
    kls: list[float] = []
    for row in generic_prompts:
        prompt = row.get("prompt") or row.get("question") or row.get("problem") or ""
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = base_model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False
            )
            base_out = base_model(gen)
            branch_out = branch_model(gen)
        base_logits = base_out.logits[0].float().tolist()
        branch_logits = branch_out.logits[0].float().tolist()
        kls.append(sequence_kl(base_logits, branch_logits))
    return sum(kls) / max(len(kls), 1)


def compute_policy_fingerprint(
    checkpoint: Path,
    fingerprint_rows: Sequence[dict[str, Any]],
    *,
    model_name: str = "Qwen/Qwen3-8B",
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_length: int = 4096,
) -> list[float]:
    """16-value fingerprint (per-prompt NLL + entropy) on the 8 fixed prompts."""
    import torch

    from math_loop.answers import NON_THINKING_SYSTEM_PROMPT, render_prompt
    from math_loop.probe_loss import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(
        checkpoint, model_name=model_name, device=device, dtype=dtype
    )
    nll_values: list[float] = []
    entropy_values: list[float] = []
    for row in fingerprint_rows[:FINGERPRINT_PROBE_SIZE]:
        prompt = render_prompt(
            tokenizer, row.get("question") or row["problem"], system_prompt=NON_THINKING_SYSTEM_PROMPT
        )
        target = (row.get("solution") or f"\\boxed{{{row['answer']}}}").strip()
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids = tokenizer(prompt + target, add_special_tokens=False).input_ids
        target_ids = full_ids[len(prompt_ids) :]
        input_tensor = torch.tensor([full_ids[:max_length]], device=device)
        with torch.no_grad():
            logits = model(input_tensor).logits[0].float()
        # next-token logits aligned to the target positions
        start = max(len(prompt_ids) - 1, 0)
        step_logits = logits[start : start + len(target_ids)].tolist()
        nll_values.append(nll_from_logits_and_targets(step_logits, target_ids))
        entropy_values.append(
            sum(sequence_entropy(step_logits)) / max(len(step_logits), 1)
        )
    return assemble_policy_fingerprint(nll_values, entropy_values)
