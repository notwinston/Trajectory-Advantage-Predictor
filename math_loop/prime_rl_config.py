"""prime-rl TOML rendering for the Qwen3 MATH loop."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from math_loop.answers import NON_THINKING_SYSTEM_PROMPT


@dataclass(frozen=True)
class PrimeRLConfigSpec:
    output_dir: Path
    split_path: Path
    max_steps: int
    batch_size: int = 4
    group_size: int = 4
    seq_len: int = 4096
    max_completion_tokens: int = 1024
    model_name: str = "Qwen/Qwen3-8B"
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    learning_rate: float = 1e-5
    num_train_gpus: int = 1
    num_infer_gpus: int = 1
    gpus_per_node: int = 2
    clean_output_dir: bool = False
    run_name: str = "qwen3-math-loop"
    system_prompt: str = NON_THINKING_SYSTEM_PROMPT


def _toml_str(value: str | Path) -> str:
    return json.dumps(str(value))


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"


def render_prime_rl_config(spec: PrimeRLConfigSpec) -> str:
    """Render a complete prime-rl config for either state generation or branch update."""

    lora_name = f"qwen3-math-r{spec.lora_rank}"
    return f"""max_steps = {spec.max_steps}
seq_len = {spec.seq_len}
output_dir = {_toml_str(spec.output_dir)}
clean_output_dir = {_toml_bool(spec.clean_output_dir)}

[ckpt]
interval = 1

[deployment]
type = "single_node"
gpus_per_node = {spec.gpus_per_node}
num_train_gpus = {spec.num_train_gpus}
num_infer_gpus = {spec.num_infer_gpus}

[model]
name = {_toml_str(spec.model_name)}

[wandb]
project = "qwen3-math-loop"
name = {_toml_str(spec.run_name)}

[trainer.model]
impl = "auto"
seq_len = {spec.seq_len}

[trainer.model.ac]
freq = 1

[trainer.model.lora]
rank = {spec.lora_rank}
alpha = {spec.lora_alpha}
dropout = {spec.lora_dropout}

[trainer.optim]
lr = {spec.learning_rate}

[trainer.ckpt.weights]
save_adapter_separately = true

[orchestrator]
batch_size = {spec.batch_size}
group_size = {spec.group_size}
seq_len = {spec.seq_len}

[orchestrator.advantage]
type = "custom"
import_path = "math_loop.advantage.normalized_advantage"
kwargs = {{ eps = 1e-8 }}

[orchestrator.model.lora]
name = {_toml_str(lora_name)}
rank = {spec.lora_rank}
alpha = {spec.lora_alpha}

[orchestrator.train.sampling]
temperature = 1.0
max_completion_tokens = {spec.max_completion_tokens}

[orchestrator.train.sampling.extra_body]
chat_template_kwargs = {{ enable_thinking = false }}

[[orchestrator.train.env]]
id = "math-loop"
name = "math-loop"
group_size = {spec.group_size}
num_workers = 1
args = {{ split_path = {_toml_str(spec.split_path)}, system_prompt = {_toml_str(spec.system_prompt)} }}

[inference]
enable_lora = true
gpu_memory_utilization = 0.80

[orchestrator.renderer]
name = "auto"
"""


def write_prime_rl_config(path: Path, spec: PrimeRLConfigSpec) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_prime_rl_config(spec), encoding="utf-8")
    return path
