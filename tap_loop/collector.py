"""TAP v1 fixed-rollout collection controller."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Protocol

import numpy as np

from tap_loop.artifacts import TapArtifactWriter, atomic_write_text
from tap_loop.data import GENERIC_DRIFT_PROMPTS, prepare_tap_training_splits, read_jsonl, write_jsonl
from tap_loop.fixed_update import FixedRolloutUpdateConfig, apply_fixed_rollout_lora_update
from tap_loop.layout import SUBDIRS
from tap_loop.probes import geometric_mean_probability, select_global_probe, select_matched_probe, utility_points
from tap_loop.schedule import build_tap_candidate_schedule, latest_history, select_main_candidate


@dataclass(frozen=True)
class CollectorConfig:
    run_root: Path
    chains: int = 2
    states_per_chain: int = 6
    candidates_per_state: int = 6
    batch_prompts: int = 2
    group_size: int = 4
    max_completion_tokens: int = 192
    gpu_count: int = 4
    seed: int = 1729
    model_name: str = "Qwen/Qwen3-8B"
    learning_rate: float = 1e-5
    grpo_beta: float = 0.0
    clip_range: float = 0.2
    backend: str = "dry-run"


class PolicyBackend(Protocol):
    def initial_state(self, chain_id: int) -> dict[str, Any]:
        ...

    def generate_candidate_trajectories(
        self, state: dict[str, Any], candidate_id: str, prompts: list[dict[str, Any]], config: CollectorConfig
    ) -> list[dict[str, Any]]:
        ...

    def evaluate_before_state(self, state: dict[str, Any], matched_probe: list[dict[str, Any]], global_probe: list[dict[str, Any]]) -> dict[str, float]:
        ...

    def apply_branch(
        self,
        state: dict[str, Any],
        candidate_id: str,
        trajectories: list[dict[str, Any]],
        matched_probe: list[dict[str, Any]],
        global_probe: list[dict[str, Any]],
        config: CollectorConfig,
    ) -> dict[str, Any]:
        ...


class DryRunPolicyBackend:
    """Deterministic lightweight backend used for local smoke tests and dry-runs."""

    def __init__(self, run_root: Path, seed: int = 1729):
        self.run_root = run_root
        self.seed = seed

    def initial_state(self, chain_id: int) -> dict[str, Any]:
        checkpoint = self.run_root / "checkpoints" / "chains" / f"chain_{chain_id:02d}" / "state_000"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "adapter_model.safetensors").write_text("dry-run\n", encoding="utf-8")
        return {
            "checkpoint": str(checkpoint),
            "optimizer_state": str(checkpoint / "optimizer.pt"),
            "step": 0,
            "policy_quality": 0.0,
        }

    def _rng(self, *parts: object) -> random.Random:
        digest = hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()
        return random.Random(self.seed + int(digest[:8], 16))

    def generate_candidate_trajectories(
        self, state: dict[str, Any], candidate_id: str, prompts: list[dict[str, Any]], config: CollectorConfig
    ) -> list[dict[str, Any]]:
        rng = self._rng(candidate_id, state["step"])
        rows: list[dict[str, Any]] = []
        for prompt_index, prompt in enumerate(prompts):
            rewards = [rng.choice([0.0, 1.0]) for _ in range(config.group_size)]
            mean_reward = float(np.mean(rewards))
            std_reward = float(np.std(rewards)) or 1.0
            for completion_index, reward in enumerate(rewards):
                seq_len = rng.randint(24, config.max_completion_tokens)
                mean_logp = rng.uniform(-3.5, -0.2)
                entropy = rng.uniform(0.2, 2.5)
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "trajectory_id": f"{candidate_id}_p{prompt_index:02d}_t{completion_index:02d}",
                        "prompt_id": prompt["id"],
                        "subject": prompt["subject"],
                        "difficulty": prompt["level"],
                        "prompt_text": prompt["question"],
                        "completion_text": "\\boxed{dry-run}",
                        "reward_total": reward,
                        "reward_exact_answer": reward,
                        "reward_format": 1.0,
                        "advantage": (reward - mean_reward) / std_reward,
                        "sequence_length": seq_len,
                        "mean_token_log_probability": mean_logp,
                        "geometric_mean_probability": geometric_mean_probability(mean_logp),
                        "arithmetic_mean_probability": min(1.0, geometric_mean_probability(mean_logp) * rng.uniform(0.8, 1.2)),
                        "mean_token_entropy": entropy,
                        "entropy_p10": max(0.0, entropy - 0.2),
                        "entropy_p50": entropy,
                        "entropy_p90": entropy + 0.2,
                        "early_mean_log_probability": mean_logp + rng.uniform(-0.2, 0.2),
                        "late_mean_log_probability": mean_logp + rng.uniform(-0.2, 0.2),
                        "confidence_slope": rng.uniform(-0.1, 0.1),
                        "mean_old_to_current_log_ratio": 0.0,
                        "mean_current_to_reference_log_ratio": rng.uniform(-0.1, 0.1),
                        "clipped_token_fraction": rng.uniform(0.0, 0.1),
                        "trajectory_embedding": [rng.uniform(-1.0, 1.0) for _ in range(256)],
                    }
                )
        return rows

    def evaluate_before_state(self, state: dict[str, Any], matched_probe: list[dict[str, Any]], global_probe: list[dict[str, Any]]) -> dict[str, float]:
        quality = float(state.get("policy_quality", 0.0))
        return {
            "matched_probe_nll": 2.0 - quality,
            "global_probe_nll": 2.2 - 0.5 * quality,
            "generic_kl": max(0.0, 0.01 * state["step"]),
        }

    def apply_branch(
        self,
        state: dict[str, Any],
        candidate_id: str,
        trajectories: list[dict[str, Any]],
        matched_probe: list[dict[str, Any]],
        global_probe: list[dict[str, Any]],
        config: CollectorConfig,
    ) -> dict[str, Any]:
        rng = self._rng(candidate_id, "branch")
        reward_mean = float(np.mean([row["reward_total"] for row in trajectories]))
        branch_quality = float(state.get("policy_quality", 0.0)) + 0.01 * (reward_mean - 0.5) + rng.uniform(-0.005, 0.005)
        checkpoint = self.run_root / "checkpoints" / "branches" / candidate_id
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "adapter_model.safetensors").write_text("dry-run-branch\n", encoding="utf-8")
        after = {
            "matched_probe_nll": 2.0 - branch_quality,
            "global_probe_nll": 2.2 - 0.5 * branch_quality,
            "generic_kl": max(0.0, 0.01 * (state["step"] + 1) + rng.uniform(0.0, 0.005)),
        }
        return {
            "checkpoint": str(checkpoint),
            "optimizer_state": str(checkpoint / "optimizer.pt"),
            "step": int(state["step"]) + 1,
            "policy_quality": branch_quality,
            "after": after,
            "training_loss_change": rng.uniform(-0.05, 0.05),
            "update_norm": rng.uniform(0.01, 0.2),
        }


class TorchPolicyBackend:
    """Single-process torch backend for fixed-rollout TAP collection on Prime pods."""

    def __init__(self, run_root: Path, seed: int = 1729, device: str = "cuda", dtype: str = "bfloat16"):
        self.run_root = run_root
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.reference_checkpoint_by_chain: dict[int, Path] = {}

    def initial_state(self, chain_id: int) -> dict[str, Any]:
        checkpoint = self.run_root / "checkpoints" / "chains" / f"chain_{chain_id:02d}" / "state_000"
        checkpoint.mkdir(parents=True, exist_ok=True)
        self.reference_checkpoint_by_chain[chain_id] = checkpoint
        return {
            "checkpoint": str(checkpoint),
            "optimizer_state": str(checkpoint / "optimizer.pt"),
            "step": 0,
            "policy_quality": 0.0,
            "chain_id": chain_id,
        }

    def _load_model(self, checkpoint: Path, model_name: str):
        from math_loop.probe_loss import load_model_and_tokenizer

        return load_model_and_tokenizer(checkpoint, model_name=model_name, device=self.device, dtype=self.dtype)

    @staticmethod
    def _project_embedding(values: Any, dim: int = 256) -> list[float]:
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if array.size >= dim:
            return array[:dim].astype(float).tolist()
        output = np.zeros(dim, dtype=np.float32)
        output[: array.size] = array
        return output.astype(float).tolist()

    def generate_candidate_trajectories(
        self, state: dict[str, Any], candidate_id: str, prompts: list[dict[str, Any]], config: CollectorConfig
    ) -> list[dict[str, Any]]:
        import torch

        from math_loop.answers import exact_match, extract_boxed_answer, render_prompt

        checkpoint = Path(state["checkpoint"])
        model, tokenizer = self._load_model(checkpoint, config.model_name)
        rows: list[dict[str, Any]] = []
        for prompt_index, prompt in enumerate(prompts):
            rendered = render_prompt(tokenizer, prompt["question"])
            prompt_ids = tokenizer(rendered, add_special_tokens=False).input_ids
            inputs = tokenizer(rendered, return_tensors="pt").to(self.device)
            generated = model.generate(
                **inputs,
                do_sample=True,
                temperature=1.0,
                max_new_tokens=config.max_completion_tokens,
                num_return_sequences=config.group_size,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            rewards: list[float] = []
            completions: list[str] = []
            for sequence in generated:
                new_tokens = sequence[inputs["input_ids"].shape[-1] :]
                completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
                completions.append(completion)
                rewards.append(1.0 if exact_match(extract_boxed_answer(completion, strict=True), prompt["answer"]) else 0.0)
            reward_mean = float(np.mean(rewards))
            reward_std = float(np.std(rewards)) or 1.0
            for completion_index, sequence in enumerate(generated):
                input_ids = sequence.tolist()
                input_tensor = sequence.unsqueeze(0).to(self.device)
                attention_mask = torch.ones_like(input_tensor)
                with torch.no_grad():
                    output = model(
                        input_ids=input_tensor,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                logits = output.logits[:, :-1, :]
                target = input_tensor[:, 1:]
                log_probs_all = torch.log_softmax(logits, dim=-1)
                token_log_probs = log_probs_all.gather(-1, target.unsqueeze(-1)).squeeze(-1).detach().cpu().float().numpy()[0]
                entropy = (-(log_probs_all.exp() * log_probs_all).sum(dim=-1)).detach().cpu().float().numpy()[0]
                completion_slice = slice(max(len(prompt_ids) - 1, 0), None)
                completion_log_probs = token_log_probs[completion_slice]
                completion_entropy = entropy[completion_slice]
                hidden = output.hidden_states[-1].mean(dim=1).detach().cpu().float().numpy()[0]
                mean_logp = float(np.mean(completion_log_probs)) if completion_log_probs.size else 0.0
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "trajectory_id": f"{candidate_id}_p{prompt_index:02d}_t{completion_index:02d}",
                        "prompt_id": prompt["id"],
                        "subject": prompt["subject"],
                        "difficulty": prompt["level"],
                        "prompt_text": prompt["question"],
                        "completion_text": completions[completion_index],
                        "reward_total": rewards[completion_index],
                        "reward_exact_answer": rewards[completion_index],
                        "reward_format": 1.0,
                        "advantage": (rewards[completion_index] - reward_mean) / reward_std,
                        "input_ids": input_ids,
                        "prompt_token_count": len(prompt_ids),
                        "old_token_log_probabilities": completion_log_probs.astype(float).tolist(),
                        "sequence_length": len(input_ids) - len(prompt_ids),
                        "mean_token_log_probability": mean_logp,
                        "geometric_mean_probability": geometric_mean_probability(mean_logp),
                        "arithmetic_mean_probability": float(np.mean(np.exp(completion_log_probs))) if completion_log_probs.size else 0.0,
                        "token_log_probability_p10": float(np.quantile(completion_log_probs, 0.1)) if completion_log_probs.size else 0.0,
                        "token_log_probability_p50": float(np.quantile(completion_log_probs, 0.5)) if completion_log_probs.size else 0.0,
                        "token_log_probability_p90": float(np.quantile(completion_log_probs, 0.9)) if completion_log_probs.size else 0.0,
                        "mean_token_entropy": float(np.mean(completion_entropy)) if completion_entropy.size else 0.0,
                        "entropy_p10": float(np.quantile(completion_entropy, 0.1)) if completion_entropy.size else 0.0,
                        "entropy_p50": float(np.quantile(completion_entropy, 0.5)) if completion_entropy.size else 0.0,
                        "entropy_p90": float(np.quantile(completion_entropy, 0.9)) if completion_entropy.size else 0.0,
                        "early_mean_log_probability": float(np.mean(completion_log_probs[: max(1, len(completion_log_probs) // 3)]))
                        if completion_log_probs.size
                        else 0.0,
                        "late_mean_log_probability": float(np.mean(completion_log_probs[-max(1, len(completion_log_probs) // 3) :]))
                        if completion_log_probs.size
                        else 0.0,
                        "confidence_slope": 0.0,
                        "mean_old_to_current_log_ratio": 0.0,
                        "mean_current_to_reference_log_ratio": 0.0,
                        "clipped_token_fraction": 0.0,
                        "trajectory_embedding": self._project_embedding(hidden),
                    }
                )
        del model
        return rows

    def evaluate_before_state(self, state: dict[str, Any], matched_probe: list[dict[str, Any]], global_probe: list[dict[str, Any]]) -> dict[str, float]:
        from math_loop.probe_loss import compute_probe_loss

        checkpoint = Path(state["checkpoint"])
        probe_dir = self.run_root / "probes" / f"tmp_state_{state['step']}"
        matched_path = probe_dir / "matched.jsonl"
        global_path = probe_dir / "global.jsonl"
        generic_path = probe_dir / "generic.jsonl"
        write_jsonl(matched_path, matched_probe)
        write_jsonl(global_path, global_probe)
        write_jsonl(
            generic_path,
            [
                {"id": f"generic-{index}", "question": prompt, "problem": prompt, "solution": "OK.", "answer": "OK."}
                for index, prompt in enumerate(GENERIC_DRIFT_PROMPTS)
            ],
        )
        matched = compute_probe_loss(checkpoint, matched_path, model_name="Qwen/Qwen3-8B", device=self.device, dtype=self.dtype)
        global_result = compute_probe_loss(checkpoint, global_path, model_name="Qwen/Qwen3-8B", device=self.device, dtype=self.dtype)
        generic = compute_probe_loss(checkpoint, generic_path, model_name="Qwen/Qwen3-8B", device=self.device, dtype=self.dtype)
        return {
            "matched_probe_nll": matched.nll,
            "global_probe_nll": global_result.nll,
            "generic_kl": generic.nll,
        }

    def apply_branch(
        self,
        state: dict[str, Any],
        candidate_id: str,
        trajectories: list[dict[str, Any]],
        matched_probe: list[dict[str, Any]],
        global_probe: list[dict[str, Any]],
        config: CollectorConfig,
    ) -> dict[str, Any]:
        output_checkpoint = self.run_root / "checkpoints" / "branches" / candidate_id
        result = apply_fixed_rollout_lora_update(
            before_checkpoint=Path(state["checkpoint"]),
            optimizer_state=Path(state["optimizer_state"]),
            trajectories=trajectories,
            output_checkpoint=output_checkpoint,
            config=FixedRolloutUpdateConfig(
                model_name=config.model_name,
                learning_rate=config.learning_rate,
                clip_range=config.clip_range,
                grpo_beta=config.grpo_beta,
                dtype=self.dtype,
                device=self.device,
            ),
        )
        branch_state = {
            "checkpoint": result["checkpoint"],
            "optimizer_state": result["optimizer_state"],
            "step": int(state["step"]) + 1,
            "policy_quality": state.get("policy_quality", 0.0),
        }
        after = self.evaluate_before_state(branch_state, matched_probe, global_probe)
        return {
            **branch_state,
            "after": after,
            "training_loss_change": float(result["training_loss"]),
            "update_norm": 0.0,
        }


def _hash_path(path: Path) -> str:
    text = str(path)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _chain_state_manifest(run_root: Path, chain_id: int, state_index: int) -> Path:
    return run_root / "checkpoints" / "chains" / f"chain_{chain_id:02d}" / f"state_{state_index:03d}" / "state.json"


def _write_chain_state_manifest(
    run_root: Path,
    *,
    chain_id: int,
    completed_state_index: int,
    state: dict[str, Any],
    selected_ids: list[str],
    selected_history: list[dict[str, Any]],
) -> None:
    next_state_index = completed_state_index + 1
    path = _chain_state_manifest(run_root, chain_id, next_state_index)
    payload = {
        "chain_id": chain_id,
        "ready_state_index": next_state_index,
        "completed_state_index": completed_state_index,
        "state": state,
        "selected_ids": selected_ids,
        "selected_history": selected_history[-4:],
    }
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_chain_state_manifest(run_root: Path, chain_id: int, state_index: int) -> dict[str, Any] | None:
    path = _chain_state_manifest(run_root, chain_id, state_index)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_collection_status(run_root: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(run_root / "collection_status.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row.get(key, 0.0) or 0.0) for row in rows])) if rows else 0.0


def _std(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.std([float(row.get(key, 0.0) or 0.0) for row in rows])) if rows else 0.0


def _avg_vector(rows: list[dict[str, Any]], key: str, dim: int) -> list[float]:
    values = [row[key] for row in rows if key in row]
    if not values:
        return [0.0] * dim
    return np.asarray(values, dtype=np.float64).mean(axis=0).astype(float).tolist()


def _cosine(a: list[float], b: list[float]) -> float:
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return float(av @ bv / denom) if denom else 0.0


def _candidate_rows(
    *,
    config: CollectorConfig,
    chain_id: int,
    state_index: int,
    candidate_index: int,
    state_id: str,
    candidate_id: str,
    trajectories: list[dict[str, Any]],
    branch: dict[str, Any],
    before: dict[str, float],
    history: list[dict[str, Any]],
    selected: bool,
) -> dict[str, Any]:
    digest = hashlib.sha1(candidate_id.encode("utf-8")).hexdigest()
    rng = random.Random(config.seed + int(digest[:8], 16))
    candidate_embedding = _avg_vector(trajectories, "trajectory_embedding", 256)
    gradient_sketch = [rng.uniform(-0.5, 0.5) for _ in range(64)]
    history_embeddings = [row["candidate_embedding"] for row in history]
    history_gradients = [row["gradient_sketch"] for row in history]
    semantic_sims = [_cosine(candidate_embedding, vector) for vector in history_embeddings]
    gradient_sims = [_cosine(gradient_sketch, vector) for vector in history_gradients]
    utility = utility_points(
        before["matched_probe_nll"],
        branch["after"]["matched_probe_nll"],
        before["global_probe_nll"],
        branch["after"]["global_probe_nll"],
        before["generic_kl"],
        branch["after"]["generic_kl"],
    )
    return {
        "state_id": state_id,
        "candidate_id": candidate_id,
        "chain_id": chain_id,
        "step": state_index,
        "candidate_index": candidate_index,
        "trajectory_ids": [row["trajectory_id"] for row in trajectories],
        "candidate_reward_mean": _mean(trajectories, "reward_total"),
        "candidate_reward_std": _std(trajectories, "reward_total"),
        "candidate_advantage_mean": _mean(trajectories, "advantage"),
        "candidate_advantage_std": _std(trajectories, "advantage"),
        "candidate_mean_log_probability": _mean(trajectories, "mean_token_log_probability"),
        "candidate_geometric_mean_probability": geometric_mean_probability(_mean(trajectories, "mean_token_log_probability")),
        "candidate_arithmetic_mean_probability": _mean(trajectories, "arithmetic_mean_probability"),
        "candidate_mean_entropy": _mean(trajectories, "mean_token_entropy"),
        "candidate_mean_sequence_length": _mean(trajectories, "sequence_length"),
        "candidate_embedding": candidate_embedding,
        "gradient_sketch": gradient_sketch,
        "gradient_norm": float(np.linalg.norm(np.asarray(gradient_sketch, dtype=np.float64))),
        "estimated_update_norm": float(branch.get("update_norm", 0.0)),
        "max_semantic_similarity_to_history": max(semantic_sims) if semantic_sims else 0.0,
        "mean_semantic_similarity_to_history": float(np.mean(semantic_sims)) if semantic_sims else 0.0,
        "max_gradient_similarity_to_history": max(gradient_sims) if gradient_sims else 0.0,
        "mean_gradient_similarity_to_history": float(np.mean(gradient_sims)) if gradient_sims else 0.0,
        "matched_probe_nll_after": branch["after"]["matched_probe_nll"],
        "global_probe_nll_after": branch["after"]["global_probe_nll"],
        "generic_kl_after": branch["after"]["generic_kl"],
        "matched_probe_gradient_alignment": rng.uniform(-1.0, 1.0),
        "candidate_log_probability_change": rng.uniform(-0.2, 0.2),
        "matched_exact_match_before": 0.0,
        "matched_exact_match_after": 0.0,
        "is_selected_for_main_chain": selected,
        **utility,
    }


def _history_rows(state_id: str, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, candidate in enumerate(reversed(history[-4:])):
        rows.append(
            {
                "state_id": state_id,
                "history_position": position,
                "relative_age": position + 1,
                "historical_candidate_id": candidate["candidate_id"],
                "historical_candidate_embedding": candidate["candidate_embedding"],
                "historical_gradient_sketch": candidate["gradient_sketch"],
                "historical_reward_mean": candidate["candidate_reward_mean"],
                "historical_advantage_mean": candidate["candidate_advantage_mean"],
                "historical_mean_log_probability": candidate["candidate_mean_log_probability"],
                "historical_mean_entropy": candidate["candidate_mean_entropy"],
                "historical_update_norm": candidate["estimated_update_norm"],
                "historical_training_loss_change": candidate.get("training_loss_change", 0.0),
                "historical_candidate_log_probability_change": candidate["candidate_log_probability_change"],
            }
        )
    return rows


def run_collection(config: CollectorConfig, backend: PolicyBackend) -> dict[str, Any]:
    for subdir in SUBDIRS:
        (config.run_root / subdir).mkdir(parents=True, exist_ok=True)
    data = prepare_tap_training_splits(config.run_root / "data")
    train_rows = read_jsonl(data.train_pool)
    heldout_rows = read_jsonl(data.heldout_pool)
    prompts_by_id = {row["id"]: row for row in train_rows}
    schedule = build_tap_candidate_schedule(
        list(prompts_by_id),
        chains=config.chains,
        states_per_chain=config.states_per_chain,
        candidates_per_state=config.candidates_per_state,
        batch_prompts=config.batch_prompts,
        seed=config.seed,
    )
    by_state: dict[tuple[int, int], list[Any]] = {}
    for candidate in schedule:
        by_state.setdefault((candidate.chain_id, candidate.state_index), []).append(candidate)
    writer = TapArtifactWriter(config.run_root, require_parquet=False)
    completed_candidates = 0

    for chain_id in range(config.chains):
        state = backend.initial_state(chain_id)
        selected_history: list[dict[str, Any]] = []
        selected_ids: list[str] = []
        for state_index in range(config.states_per_chain):
            resume_manifest = _load_chain_state_manifest(config.run_root, chain_id, state_index)
            if resume_manifest is not None:
                state = resume_manifest["state"]
                selected_ids = list(resume_manifest.get("selected_ids", []))
                selected_history = list(resume_manifest.get("selected_history", []))

            completed_manifest = _load_chain_state_manifest(config.run_root, chain_id, state_index + 1)
            if completed_manifest is not None:
                state = completed_manifest["state"]
                selected_ids = list(completed_manifest.get("selected_ids", []))
                selected_history = list(completed_manifest.get("selected_history", []))
                _write_collection_status(
                    config.run_root,
                    {
                        "status": "resumed_skip_completed_state",
                        "chain_id": chain_id,
                        "state_index": state_index,
                        "completed_candidates_this_process": completed_candidates,
                    },
                )
                continue

            state_id = f"chain_{chain_id:02d}_state_{state_index:03d}"
            batches = by_state[(chain_id, state_index)]
            _write_collection_status(
                config.run_root,
                {
                    "status": "running_state",
                    "chain_id": chain_id,
                    "state_index": state_index,
                    "completed_candidates_this_process": completed_candidates,
                },
            )
            state_global_probe = select_global_probe(heldout_rows, seed=config.seed + state_index)
            state_probe = backend.evaluate_before_state(state, state_global_probe, state_global_probe)
            writer.write_fragment(
                "states",
                [
                    {
                        "schema_version": "tap_v1",
                        "state_id": state_id,
                        "chain_id": chain_id,
                        "step": state_index,
                        "seed": config.seed,
                        "checkpoint_hash": _hash_path(Path(state["checkpoint"])),
                        "optimizer_state_hash": _hash_path(Path(state["optimizer_state"])),
                        "learning_rate": config.learning_rate,
                        "grpo_beta": config.grpo_beta,
                        "clip_range": config.clip_range,
                        "lora_rank": 16,
                        "matched_probe_nll_before": state_probe["matched_probe_nll"],
                        "global_probe_nll_before": state_probe["global_probe_nll"],
                        "generic_kl_before": state_probe["generic_kl"],
                        "adam_first_moment_norm": 0.0,
                        "adam_second_moment_norm": 0.0,
                        "policy_fingerprint": [0.0] * 16,
                        "history_candidate_ids": latest_history(selected_ids),
                    }
                ],
                fragment_id=state_id,
            )
            writer.write_fragment("history", _history_rows(state_id, selected_history), fragment_id=state_id)
            selected_index = select_main_candidate(
                config.candidates_per_state,
                chain_id=chain_id,
                state_index=state_index,
                seed=config.seed,
            )
            state_candidates: list[dict[str, Any]] = []
            for candidate in batches:
                prompts = [prompts_by_id[prompt_id] for prompt_id in candidate.prompt_ids]
                matched_probe = select_matched_probe(prompts, heldout_rows, seed=config.seed + state_index)
                global_probe = select_global_probe(heldout_rows, seed=config.seed + state_index)
                before = backend.evaluate_before_state(state, matched_probe, global_probe)
                trajectories = backend.generate_candidate_trajectories(state, candidate.candidate_id, prompts, config)
                branch = backend.apply_branch(state, candidate.candidate_id, trajectories, matched_probe, global_probe, config)
                selected = candidate.candidate_index == selected_index
                candidate_row = _candidate_rows(
                    config=config,
                    chain_id=chain_id,
                    state_index=state_index,
                    candidate_index=candidate.candidate_index,
                    state_id=state_id,
                    candidate_id=candidate.candidate_id,
                    trajectories=[{**row, "state_id": state_id} for row in trajectories],
                    branch=branch,
                    before=before,
                    history=selected_history,
                    selected=selected,
                )
                writer.write_fragment(
                    "trajectories",
                    [{**row, "state_id": state_id} for row in trajectories],
                    fragment_id=candidate.candidate_id,
                )
                writer.write_fragment("candidates", [candidate_row], fragment_id=candidate.candidate_id)
                state_candidates.append({**candidate_row, "branch_state": branch})
                completed_candidates += 1
                _write_collection_status(
                    config.run_root,
                    {
                        "status": "running_candidate",
                        "chain_id": chain_id,
                        "state_index": state_index,
                        "candidate_index": candidate.candidate_index,
                        "completed_candidates_this_process": completed_candidates,
                    },
                )
            selected_candidate = next(row for row in state_candidates if row["candidate_index"] == selected_index)
            branch_state = selected_candidate.pop("branch_state")
            state = {
                "checkpoint": branch_state["checkpoint"],
                "optimizer_state": branch_state["optimizer_state"],
                "step": branch_state["step"],
                "policy_quality": branch_state.get("policy_quality", state.get("policy_quality", 0.0)),
            }
            selected_ids.append(selected_candidate["candidate_id"])
            selected_history.append(selected_candidate)
            writer.compact_all()
            _write_chain_state_manifest(
                config.run_root,
                chain_id=chain_id,
                completed_state_index=state_index,
                state=state,
                selected_ids=selected_ids,
                selected_history=selected_history,
            )
            _write_collection_status(
                config.run_root,
                {
                    "status": "completed_state",
                    "chain_id": chain_id,
                    "state_index": state_index,
                    "next_state_index": state_index + 1,
                    "selected_candidate_id": selected_candidate["candidate_id"],
                    "completed_candidates_this_process": completed_candidates,
                },
            )
    _write_collection_status(
        config.run_root,
        {"status": "complete", "completed_candidates_this_process": completed_candidates},
    )
    return {
        "run_root": str(config.run_root),
        "completed_candidates": completed_candidates,
        "parquet": {key: str(path) for key, path in writer.compact_all().items()},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--states-per-chain", type=int, default=6)
    parser.add_argument("--candidates-per-state", type=int, default=6)
    parser.add_argument("--batch-prompts", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-completion-tokens", type=int, default=192)
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--backend", choices=("dry-run", "torch"), default="dry-run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = CollectorConfig(
        run_root=args.run_root,
        chains=args.chains,
        states_per_chain=args.states_per_chain,
        candidates_per_state=args.candidates_per_state,
        batch_prompts=args.batch_prompts,
        group_size=args.group_size,
        max_completion_tokens=args.max_completion_tokens,
        gpu_count=args.gpu_count,
        seed=args.seed,
        backend=args.backend,
    )
    if args.backend == "dry-run":
        backend: PolicyBackend = DryRunPolicyBackend(args.run_root, seed=args.seed)
    else:
        backend = TorchPolicyBackend(args.run_root, seed=args.seed)
    result = run_collection(config, backend)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
