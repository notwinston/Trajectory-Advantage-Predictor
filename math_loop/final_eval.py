"""Final exact-match evaluation on the sealed MATH-500 split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from math_loop.answers import exact_match, extract_boxed_answer, render_prompt
from math_loop.data import prepare_final_split, read_jsonl
from math_loop.probe_loss import load_model_and_tokenizer


def evaluate(
    checkpoint: Path,
    split: Path,
    *,
    output_path: Path,
    model_name: str = "Qwen/Qwen3-8B",
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> dict[str, float | int | str]:
    import torch

    rows = read_jsonl(split)
    if not rows:
        raise ValueError(f"final eval split is empty: {split}")

    model, tokenizer = load_model_and_tokenizer(
        checkpoint,
        model_name=model_name,
        device=device,
        dtype=dtype,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            prompt = render_prompt(tokenizer, row.get("question") or row["problem"])
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if temperature > 0:
                kwargs.update({"do_sample": True, "temperature": temperature})
            else:
                kwargs.update({"do_sample": False})
            with torch.no_grad():
                generated = model.generate(**inputs, **kwargs)
            new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
            prediction = extract_boxed_answer(completion, strict=True)
            is_correct = exact_match(prediction, row["answer"])
            correct += int(is_correct)
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "answer": row["answer"],
                        "prediction": prediction,
                        "completion": completion,
                        "correct": is_correct,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    accuracy = correct / len(rows)
    summary = {
        "checkpoint": str(checkpoint),
        "split": str(split),
        "examples": len(rows),
        "correct": correct,
        "exact_match": accuracy,
        "predictions": str(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("data/math_loop/math500.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/qwen3_math_loop/math500_predictions.jsonl"))
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--prepare-final-split",
        action="store_true",
        help="download/write MATH-500 if --split does not exist",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.prepare_final_split and not args.split.exists():
        prepare_final_split(args.split.parent)
    summary = evaluate(
        args.checkpoint,
        args.split,
        output_path=args.output,
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
