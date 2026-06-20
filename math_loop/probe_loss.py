"""Teacher-forced probe NLL for held-out MATH training problems."""

from __future__ import annotations

import argparse
import gc
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from math_loop.answers import NON_THINKING_SYSTEM_PROMPT, render_prompt
from math_loop.data import read_jsonl


@dataclass(frozen=True)
class ProbeLossResult:
    checkpoint: str
    examples: int
    tokens: int
    nll: float


def find_adapter_path(path: Path) -> Path | None:
    if (path / "adapter_config.json").exists():
        return path
    for candidate in path.rglob("adapter_config.json"):
        return candidate.parent
    return None


def _torch_dtype(torch: Any, dtype: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]


def _cleanup_torch_cuda(torch: Any, device: str) -> None:
    """Release GPU allocations before prime-rl/vLLM starts in another process."""
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_model_and_tokenizer(
    checkpoint: Path,
    *,
    model_name: str = "Qwen/Qwen3-8B",
    device: str = "cuda",
    dtype: str = "bfloat16",
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint path does not exist: {checkpoint}")

    tokenizer_source = checkpoint if (checkpoint / "tokenizer_config.json").exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = _torch_dtype(torch, dtype)
    if (checkpoint / "config.json").exists():
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        adapter_path = find_adapter_path(checkpoint)
        if adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)

    model.to(device)
    model.eval()
    return model, tokenizer


def _encode_example(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    max_length: int,
    system_prompt: str,
) -> dict[str, list[int]] | None:
    prompt = render_prompt(tokenizer, row.get("question") or row["problem"], system_prompt=system_prompt)
    target = (row.get("solution") or f"\\boxed{{{row['answer']}}}").strip()
    eos = tokenizer.eos_token or ""
    full = prompt + target + eos

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    input_ids = tokenizer(full, add_special_tokens=False).input_ids
    labels = list(input_ids)
    labels[: len(prompt_ids)] = [-100] * len(prompt_ids)

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    if not any(label != -100 for label in labels):
        return None
    return {"input_ids": input_ids, "labels": labels}


def _batches(items: list[dict[str, list[int]]], size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def compute_probe_loss(
    checkpoint: Path,
    split_path: Path,
    *,
    model_name: str = "Qwen/Qwen3-8B",
    batch_size: int = 1,
    max_length: int = 4096,
    device: str = "cuda",
    dtype: str = "bfloat16",
    system_prompt: str = NON_THINKING_SYSTEM_PROMPT,
) -> ProbeLossResult:
    import torch

    rows = read_jsonl(split_path)
    if not rows:
        raise ValueError(f"probe split is empty: {split_path}")
    model = None
    tokenizer = None
    try:
        model, tokenizer = load_model_and_tokenizer(
            checkpoint,
            model_name=model_name,
            device=device,
            dtype=dtype,
        )
        encoded = [
            item
            for item in (
                _encode_example(tokenizer, row, max_length=max_length, system_prompt=system_prompt)
                for row in rows
            )
            if item is not None
        ]
        if not encoded:
            raise ValueError("all probe examples were truncated before target tokens")

        pad_token_id = tokenizer.pad_token_id
        total_loss = 0.0
        total_tokens = 0
        for batch in _batches(encoded, batch_size):
            width = max(len(item["input_ids"]) for item in batch)
            input_ids = []
            labels = []
            attention_mask = []
            for item in batch:
                pad = width - len(item["input_ids"])
                input_ids.append(item["input_ids"] + [pad_token_id] * pad)
                labels.append(item["labels"] + [-100] * pad)
                attention_mask.append([1] * len(item["input_ids"]) + [0] * pad)
            input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
            label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
            mask_tensor = torch.tensor(attention_mask, dtype=torch.long, device=device)
            token_count = int((label_tensor != -100).sum().item())
            with torch.no_grad():
                output = model(input_ids=input_tensor, attention_mask=mask_tensor, labels=label_tensor)
            total_loss += float(output.loss.item()) * token_count
            total_tokens += token_count
            del input_tensor, label_tensor, mask_tensor, output

        return ProbeLossResult(
            checkpoint=str(checkpoint),
            examples=len(encoded),
            tokens=total_tokens,
            nll=total_loss / max(total_tokens, 1),
        )
    finally:
        del model, tokenizer
        _cleanup_torch_cuda(torch, device)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = compute_probe_loss(
        args.checkpoint,
        args.split,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        dtype=args.dtype,
    )
    print(json.dumps(asdict(result), sort_keys=True))


if __name__ == "__main__":
    main()
