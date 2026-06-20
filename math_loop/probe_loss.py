"""Teacher-forced probe NLL for held-out MATH training problems."""

from __future__ import annotations

import argparse
import gc
from dataclasses import asdict, dataclass
import json
import os
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


def _probe_loss_over_model(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    max_length: int,
    device: str,
    system_prompt: str,
) -> tuple[float, int, int]:
    """Teacher-forced NLL over an ALREADY-LOADED model. Returns (nll, examples, tokens).

    Shared by :func:`compute_probe_loss` (load -> here -> cleanup) and
    :meth:`ProbeSession.probe_nll` (resident base + active adapter). Padding/masking
    and the token-weighted mean are identical regardless of ``batch_size``.
    """
    import torch

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

    return total_loss / max(total_tokens, 1), len(encoded), total_tokens


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
        nll, examples, tokens = _probe_loss_over_model(
            model,
            tokenizer,
            rows,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            system_prompt=system_prompt,
        )
        return ProbeLossResult(
            checkpoint=str(checkpoint),
            examples=examples,
            tokens=tokens,
            nll=nll,
        )
    finally:
        del model, tokenizer
        _cleanup_torch_cuda(torch, device)


class ProbeSession:
    """Resident Qwen3-8B base with hot-swappable LoRA adapters for the probe phase.

    Loads the base model ONCE and attaches each state/branch LoRA adapter via PEFT
    ``load_adapter``/``set_adapter`` instead of re-running ``from_pretrained`` per
    probe (which previously reloaded ~15GB per call). One session serves every probe
    for a single state; ``close()`` frees the GPU before the next state's prime-rl
    subprocesses run. All torch/transformers/peft imports are deferred to method
    bodies so the module stays importable on a CPU-only host.
    """

    def __init__(
        self,
        *,
        model_name: str = "Qwen/Qwen3-8B",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_length: int = 4096,
        system_prompt: str = NON_THINKING_SYSTEM_PROMPT,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.max_length = max_length
        self.system_prompt = system_prompt
        self._base = None
        self._model = None  # PeftModel once the first adapter is attached
        self._tokenizer = None
        self._adapters: set[str] = set()
        self._active: str | None = None

    # -- lifecycle --------------------------------------------------------
    def _ensure_base(self) -> None:
        if self._base is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer = tokenizer
        self._base = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=_torch_dtype(torch, self.dtype),
            trust_remote_code=True,
        )
        self._base.to(self.device)
        self._base.eval()

    def use_adapter(self, checkpoint: Path) -> str:
        """Activate the LoRA adapter at ``checkpoint``; load it once, else ``set_adapter``."""
        import hashlib

        self._ensure_base()
        adapter_path = find_adapter_path(Path(checkpoint))
        if adapter_path is None:
            raise FileNotFoundError(f"no LoRA adapter found under checkpoint: {checkpoint}")
        key = str(Path(adapter_path).resolve())
        name = "ad_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        if name in self._adapters:
            self._model.set_adapter(name)
            self._active = name
            return name
        if self._model is None:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._base, str(adapter_path), adapter_name=name)
            self._model.to(self.device)
            self._model.eval()
        else:
            self._model.load_adapter(str(adapter_path), adapter_name=name)
        self._adapters.add(name)
        self._model.set_adapter(name)
        self._active = name
        return name

    def _active_model(self):
        return self._model if self._model is not None else self._base

    def close(self) -> None:
        import torch

        self._model = None
        self._base = None
        self._tokenizer = None
        self._adapters.clear()
        self._active = None
        _cleanup_torch_cuda(torch, self.device)

    # -- probes (the active adapter is already set by use_adapter) ---------
    def probe_nll(self, probe_rows: list[dict[str, Any]], *, batch_size: int = 8) -> float:
        model = self._active_model()
        model.eval()
        nll, _, _ = _probe_loss_over_model(
            model,
            self._tokenizer,
            list(probe_rows),
            batch_size=batch_size,
            max_length=self.max_length,
            device=self.device,
            system_prompt=self.system_prompt,
        )
        return float(nll)

    def fingerprint(self, fingerprint_rows: list[dict[str, Any]]) -> list[float]:
        import torch

        from math_loop.answers import NON_THINKING_SYSTEM_PROMPT as _SYS, render_prompt
        from math_loop.data import FINGERPRINT_PROBE_SIZE
        from math_loop.tap_probes import (
            assemble_policy_fingerprint,
            entropy_mean_vectorized,
            nll_mean_vectorized,
        )

        model = self._active_model()
        model.eval()
        tokenizer = self._tokenizer
        nll_values: list[float] = []
        entropy_values: list[float] = []
        for row in list(fingerprint_rows)[:FINGERPRINT_PROBE_SIZE]:
            prompt = render_prompt(
                tokenizer, row.get("question") or row["problem"], system_prompt=_SYS
            )
            target = (row.get("solution") or f"\\boxed{{{row['answer']}}}").strip()
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            full_ids = tokenizer(prompt + target, add_special_tokens=False).input_ids
            target_ids = full_ids[len(prompt_ids):]
            input_tensor = torch.tensor([full_ids[: self.max_length]], device=self.device)
            with torch.no_grad():
                logits = model(input_tensor).logits[0].float()
                start = max(len(prompt_ids) - 1, 0)
                step_logits = logits[start : start + len(target_ids)]
                nll_values.append(nll_mean_vectorized(step_logits, target_ids))
                entropy_values.append(entropy_mean_vectorized(step_logits))
            del input_tensor, logits
        return assemble_policy_fingerprint(nll_values, entropy_values)

    def logits_for_generic(self, generic_prompts: list[dict[str, Any]], *, max_new_tokens: int = 64):
        """Greedy-generate + base-forward under the CURRENT (state) adapter, once per state.

        Returns ``[(gen_ids, base_logits), ...]`` reused by :meth:`sequence_kl_vs` for
        every candidate (the state policy + greedy generation are identical across the
        state's candidates, so this work is done once instead of 8x).
        """
        import torch

        model = self._active_model()
        model.eval()
        tokenizer = self._tokenizer
        cached = []
        for row in generic_prompts:
            prompt = row.get("prompt") or row.get("question") or row.get("problem") or ""
            enc = tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
                base_logits = model(gen).logits[0].float()
            cached.append((gen.detach(), base_logits.detach()))
            del enc
        return cached

    def sequence_kl_vs(self, cached_base) -> float:
        """Mean KL(state || branch) over the cached generic sequences (CURRENT = branch)."""
        import torch

        from math_loop.tap_probes import kl_mean_vectorized

        model = self._active_model()
        model.eval()
        kls: list[float] = []
        for gen, base_logits in cached_base:
            with torch.no_grad():
                branch_logits = model(gen).logits[0].float()
                kls.append(kl_mean_vectorized(base_logits, branch_logits))
            del branch_logits
        return sum(kls) / max(len(kls), 1)

    def grad_sketch(
        self, candidate_rows: list[dict[str, Any]], *, sketch_dim: int = 64, seed: int = 1729
    ) -> list[float]:
        """64-dim random projection of the active adapter's gradient (matches
        :func:`math_loop.branch.compute_lora_gradient_sketch` exactly, sans reload)."""
        if os.environ.get("TAP_NO_TORCH") == "1":
            return [0.0] * sketch_dim
        import numpy as np
        import torch

        model = self._active_model()
        tokenizer = self._tokenizer
        try:
            model.train()
            grads: list[Any] = []
            for row in candidate_rows:
                text = (row.get("prompt_text") or "") + (row.get("completion_text") or "")
                enc = tokenizer(text, return_tensors="pt").to(self.device)
                out = model(**enc, labels=enc["input_ids"])
                model.zero_grad(set_to_none=True)
                out.loss.backward()
                flat = [
                    p.grad.detach().reshape(-1).float()
                    for _, p in model.named_parameters()
                    if p.requires_grad and p.grad is not None
                ]
                if flat:
                    grads.append(torch.cat(flat))
            if not grads:
                return [0.0] * sketch_dim
            gradient = torch.stack(grads).mean(0).cpu().numpy()
            rng = np.random.default_rng(seed)
            projection = rng.standard_normal((sketch_dim, gradient.shape[0])).astype(np.float32)
            projection /= np.sqrt(gradient.shape[0])
            sketch = projection @ gradient.astype(np.float32)
            return [float(v) for v in np.nan_to_num(sketch)]
        except Exception:
            return [0.0] * sketch_dim
        finally:
            model.zero_grad(set_to_none=True)
            model.eval()


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
