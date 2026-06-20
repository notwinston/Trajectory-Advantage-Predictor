"""Fixed-rollout LoRA/GRPO branch update primitives for TAP collection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class FixedRolloutUpdateConfig:
    model_name: str = "Qwen/Qwen3-8B"
    learning_rate: float = 1e-5
    clip_range: float = 0.2
    grpo_beta: float = 0.0
    dtype: str = "bfloat16"
    device: str = "cuda"
    max_length: int = 4096


def grpo_surrogate_loss(
    current_log_probs,
    old_log_probs,
    advantages,
    *,
    clip_range: float = 0.2,
    reference_log_probs=None,
    beta: float = 0.0,
):
    """Torch GRPO clipped surrogate loss for already-tokenized fixed rollouts."""

    import torch

    ratios = torch.exp(current_log_probs - old_log_probs)
    clipped = torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range)
    token_loss = -torch.minimum(ratios * advantages, clipped * advantages)
    loss = token_loss.mean()
    if reference_log_probs is not None and beta:
        loss = loss + beta * (current_log_probs - reference_log_probs).mean()
    return loss


def apply_fixed_rollout_lora_update(
    *,
    before_checkpoint: Path,
    optimizer_state: Path | None,
    trajectories: Sequence[dict[str, Any]],
    output_checkpoint: Path,
    config: FixedRolloutUpdateConfig,
) -> dict[str, float | str]:
    """Apply one LoRA-only optimizer step from fixed candidate trajectories.

    This function is intended for the Prime pod. It imports torch/transformers/peft
    lazily so local tests can run without the heavy training stack.
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from math_loop.probe_loss import find_adapter_path

    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except Exception as exc:  # pragma: no cover - remote dependency path
        raise RuntimeError("peft is required for TAP fixed-rollout LoRA updates") from exc

    if not trajectories:
        raise ValueError("fixed rollout update requires at least one trajectory")
    tokenizer_source = before_checkpoint if (before_checkpoint / "tokenizer_config.json").exists() else config.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[config.dtype]
    base = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=dtype, trust_remote_code=True)
    adapter_path = find_adapter_path(before_checkpoint)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    else:
        lora = LoraConfig(r=16, lora_alpha=32.0, lora_dropout=0.0, task_type="CAUSAL_LM")
        model = get_peft_model(base, lora)
    model.to(config.device)
    model.train()
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=config.learning_rate)
    if optimizer_state and optimizer_state.exists():
        optimizer.load_state_dict(torch.load(optimizer_state, map_location=config.device))

    losses = []
    optimizer.zero_grad(set_to_none=True)
    for row in trajectories:
        input_ids = torch.tensor([row["input_ids"][: config.max_length]], dtype=torch.long, device=config.device)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        prompt_tokens = int(row.get("prompt_token_count", 0))
        if prompt_tokens:
            labels[:, :prompt_tokens] = -100
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits[:, :-1, :]
        target = input_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
        mask = labels[:, 1:] != -100
        current = log_probs[mask]
        old_values = row.get("old_token_log_probabilities")
        if old_values is None:
            old = current.detach()
        else:
            old = torch.tensor(old_values[: current.numel()], dtype=current.dtype, device=config.device)
        reference_values = row.get("reference_token_log_probabilities")
        reference = None
        if reference_values is not None:
            reference = torch.tensor(reference_values[: current.numel()], dtype=current.dtype, device=config.device)
        advantage_value = float(row.get("advantage", 0.0))
        advantages = torch.full_like(current, advantage_value)
        loss = grpo_surrogate_loss(
            current,
            old,
            advantages,
            clip_range=config.clip_range,
            reference_log_probs=reference,
            beta=config.grpo_beta,
        )
        loss.backward()
        losses.append(float(loss.detach().cpu()))
    optimizer.step()

    output_checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_checkpoint)
    tokenizer.save_pretrained(output_checkpoint)
    torch.save(optimizer.state_dict(), output_checkpoint / "optimizer.pt")
    return {
        "checkpoint": str(output_checkpoint),
        "optimizer_state": str(output_checkpoint / "optimizer.pt"),
        "training_loss": float(sum(losses) / max(len(losses), 1)),
    }
