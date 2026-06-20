"""Frozen 4-Parquet schema contract for TAP v1.

Column lists are transcribed VERBATIM from ``TAP_v1_3_4_hour_plan.txt`` section
"DATA FORMAT". The spec field names are authoritative — do not rename/add/drop
columns. Vector widths are frozen here so the parallel waves (engine, model) all
agree on shapes.

Frozen vector widths
--------------------
- ``policy_fingerprint``            : 16   (spec: 16-value NLL+entropy vector)
- ``candidate_embedding``           : 256  (spec: 256-dim pooled Qwen hidden)
- ``gradient_sketch``               : 64   (spec: 64-dim random projection)
- ``trajectory_embedding``          : 128  (spec is silent — frozen here at 128)
- ``historical_candidate_embedding``: 256  (same contract as candidate_embedding)
- ``historical_gradient_sketch``    : 64   (same contract as gradient_sketch)

Vectors are stored as ``list<float32>``; ids are strings
(``state_id = "{chain}-{state}"``, ``candidate_id = "{state_id}-{k}"``).
``schema_version`` is the constant ``"tap-v1"``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "tap-v1"

# --- Frozen vector widths ---------------------------------------------------
POLICY_FINGERPRINT_DIM = 16
CANDIDATE_EMBEDDING_DIM = 256
GRADIENT_SKETCH_DIM = 64
TRAJECTORY_EMBEDDING_DIM = 128  # contract's frozen choice; spec is silent here

FILE_NAMES = (
    "states.parquet",
    "trajectories.parquet",
    "candidates.parquet",
    "history.parquet",
)

# Logical type tags used by the schema below.
#   "str"      -> pa.string()
#   "int"      -> pa.int64()
#   "float"    -> pa.float64()   (scalars kept in float64 for label precision)
#   "bool"     -> pa.bool_()
#   "vec:N"    -> pa.list_(pa.float32()) with frozen width N
#   "list_str" -> pa.list_(pa.string())

# 1. states.parquet — one row per main-chain policy state.
STATES_SCHEMA: List[Tuple[str, str]] = [
    ("schema_version", "str"),
    ("state_id", "str"),
    ("chain_id", "str"),
    ("step", "int"),
    ("seed", "int"),
    ("checkpoint_hash", "str"),
    ("optimizer_state_hash", "str"),
    ("learning_rate", "float"),
    ("grpo_beta", "float"),
    ("clip_range", "float"),
    ("lora_rank", "int"),
    ("matched_probe_nll_before", "float"),
    ("global_probe_nll_before", "float"),
    ("generic_kl_before", "float"),
    ("adam_first_moment_norm", "float"),
    ("adam_second_moment_norm", "float"),
    ("policy_fingerprint", f"vec:{POLICY_FINGERPRINT_DIM}"),
    ("history_candidate_ids", "list_str"),
]

# 2. trajectories.parquet — one row per generated completion.
TRAJECTORIES_SCHEMA: List[Tuple[str, str]] = [
    ("state_id", "str"),
    ("candidate_id", "str"),
    ("trajectory_id", "str"),
    ("prompt_id", "str"),
    ("subject", "str"),
    ("difficulty", "str"),
    ("prompt_text", "str"),
    ("completion_text", "str"),
    ("reward_total", "float"),
    ("reward_exact_answer", "float"),
    ("reward_format", "float"),
    ("advantage", "float"),
    ("sequence_length", "int"),
    ("mean_token_log_probability", "float"),
    ("geometric_mean_probability", "float"),
    ("arithmetic_mean_probability", "float"),
    ("token_log_probability_p10", "float"),
    ("token_log_probability_p50", "float"),
    ("token_log_probability_p90", "float"),
    ("mean_token_entropy", "float"),
    ("entropy_p10", "float"),
    ("entropy_p50", "float"),
    ("entropy_p90", "float"),
    ("early_mean_log_probability", "float"),
    ("late_mean_log_probability", "float"),
    ("confidence_slope", "float"),
    ("mean_old_to_current_log_ratio", "float"),
    ("mean_current_to_reference_log_ratio", "float"),
    ("clipped_token_fraction", "float"),
    ("trajectory_embedding", f"vec:{TRAJECTORY_EMBEDDING_DIM}"),
]

# 3. candidates.parquet — one row per candidate branch (main TAP training table).
CANDIDATES_SCHEMA: List[Tuple[str, str]] = [
    ("state_id", "str"),
    ("candidate_id", "str"),
    ("chain_id", "str"),
    ("step", "int"),
    ("trajectory_ids", "list_str"),
    ("candidate_reward_mean", "float"),
    ("candidate_reward_std", "float"),
    ("candidate_advantage_mean", "float"),
    ("candidate_advantage_std", "float"),
    ("candidate_mean_log_probability", "float"),
    ("candidate_geometric_mean_probability", "float"),
    ("candidate_arithmetic_mean_probability", "float"),
    ("candidate_mean_entropy", "float"),
    ("candidate_mean_sequence_length", "float"),
    ("candidate_embedding", f"vec:{CANDIDATE_EMBEDDING_DIM}"),
    ("gradient_sketch", f"vec:{GRADIENT_SKETCH_DIM}"),
    ("gradient_norm", "float"),
    ("estimated_update_norm", "float"),
    ("max_semantic_similarity_to_history", "float"),
    ("mean_semantic_similarity_to_history", "float"),
    ("max_gradient_similarity_to_history", "float"),
    ("mean_gradient_similarity_to_history", "float"),
    ("matched_probe_nll_after", "float"),
    ("global_probe_nll_after", "float"),
    ("generic_kl_after", "float"),
    ("matched_gain", "float"),
    ("global_gain", "float"),
    ("incremental_generic_kl", "float"),
    ("utility_points", "float"),
    ("candidate_log_probability_change", "float"),
    ("matched_exact_match_before", "float"),
    ("matched_exact_match_after", "float"),
    ("is_selected_for_main_chain", "bool"),
]

# 4. history.parquet — one row per historical update attached to a state.
HISTORY_SCHEMA: List[Tuple[str, str]] = [
    ("state_id", "str"),
    ("history_position", "int"),
    ("relative_age", "int"),
    ("historical_candidate_id", "str"),
    ("historical_candidate_embedding", f"vec:{CANDIDATE_EMBEDDING_DIM}"),
    ("historical_gradient_sketch", f"vec:{GRADIENT_SKETCH_DIM}"),
    ("historical_reward_mean", "float"),
    ("historical_advantage_mean", "float"),
    ("historical_mean_log_probability", "float"),
    ("historical_mean_entropy", "float"),
    ("historical_update_norm", "float"),
    ("historical_training_loss_change", "float"),
    ("historical_candidate_log_probability_change", "float"),
]

SCHEMAS: Dict[str, List[Tuple[str, str]]] = {
    "states.parquet": STATES_SCHEMA,
    "trajectories.parquet": TRAJECTORIES_SCHEMA,
    "candidates.parquet": CANDIDATES_SCHEMA,
    "history.parquet": HISTORY_SCHEMA,
}


def _tag_to_arrow(tag: str) -> pa.DataType:
    if tag == "str":
        return pa.string()
    if tag == "int":
        return pa.int64()
    if tag == "float":
        return pa.float64()
    if tag == "bool":
        return pa.bool_()
    if tag == "list_str":
        return pa.list_(pa.string())
    if tag.startswith("vec:"):
        return pa.list_(pa.float32())
    raise ValueError(f"unknown type tag: {tag!r}")


def vector_width(tag: str) -> int | None:
    """Return the frozen width for a ``vec:N`` tag, else ``None``."""
    if tag.startswith("vec:"):
        return int(tag.split(":", 1)[1])
    return None


def arrow_schema(file_name: str) -> pa.Schema:
    """Return the pyarrow ``Schema`` to WRITE ``file_name`` with."""
    fields = SCHEMAS[file_name]
    return pa.schema([(name, _tag_to_arrow(tag)) for name, tag in fields])


def column_names(file_name: str) -> List[str]:
    return [name for name, _ in SCHEMAS[file_name]]


def _check_scalar_type(tag: str, arrow_type: pa.DataType) -> bool:
    if tag == "str":
        return pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type)
    if tag == "int":
        return pa.types.is_integer(arrow_type)
    if tag == "float":
        return pa.types.is_floating(arrow_type)
    if tag == "bool":
        return pa.types.is_boolean(arrow_type)
    return False


def _validate_file(path: Path, file_name: str) -> None:
    table = pq.read_table(path)
    schema = SCHEMAS[file_name]
    expected_cols = [name for name, _ in schema]

    # 1. EXACT set + order of column names.
    actual_cols = list(table.schema.names)
    if actual_cols != expected_cols:
        missing = set(expected_cols) - set(actual_cols)
        extra = set(actual_cols) - set(expected_cols)
        raise ValueError(
            f"{file_name}: column mismatch. "
            f"missing={sorted(missing)} extra={sorted(extra)} "
            f"(order_ok={actual_cols == expected_cols})"
        )

    # 2. dtypes + vector widths.
    for name, tag in schema:
        field = table.schema.field(name)
        arrow_type = field.type
        width = vector_width(tag)
        if width is not None:
            if not (pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type)):
                raise ValueError(f"{file_name}.{name}: expected list type, got {arrow_type}")
            if not pa.types.is_float32(arrow_type.value_type):
                raise ValueError(
                    f"{file_name}.{name}: expected list<float32>, got list<{arrow_type.value_type}>"
                )
            column = table.column(name)
            for row_index, value in enumerate(column.to_pylist()):
                if value is None or len(value) != width:
                    got = "null" if value is None else len(value)
                    raise ValueError(
                        f"{file_name}.{name}: row {row_index} width {got} != frozen {width}"
                    )
        elif tag == "list_str":
            if not (pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type)):
                raise ValueError(f"{file_name}.{name}: expected list<string>, got {arrow_type}")
            if not (
                pa.types.is_string(arrow_type.value_type)
                or pa.types.is_large_string(arrow_type.value_type)
            ):
                raise ValueError(
                    f"{file_name}.{name}: expected list<string>, got list<{arrow_type.value_type}>"
                )
        else:
            if not _check_scalar_type(tag, arrow_type):
                raise ValueError(
                    f"{file_name}.{name}: expected {tag}, got arrow type {arrow_type}"
                )

    # 3. gradient_sketch must have no NaN (sanity for the random projection).
    if file_name == "candidates.parquet":
        import math

        for row_index, value in enumerate(table.column("gradient_sketch").to_pylist()):
            if value is None or any(v is None or math.isnan(v) for v in value):
                raise ValueError(f"candidates.parquet.gradient_sketch: NaN/None in row {row_index}")


def validate_parquet_dir(path: str | Path) -> None:
    """Validate that ``path`` holds the four schema-valid TAP Parquet files.

    Raises ``ValueError`` / ``FileNotFoundError`` on any mismatch.
    """
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"not a directory: {directory}")
    for file_name in FILE_NAMES:
        file_path = directory / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"missing required parquet: {file_path}")
        _validate_file(file_path, file_name)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TAP v1 frozen Parquet schema contract.")
    parser.add_argument(
        "--validate",
        metavar="DIR",
        help="validate that DIR holds the four schema-valid TAP parquet files",
    )
    args = parser.parse_args(argv)
    if args.validate:
        validate_parquet_dir(args.validate)
        print(f"OK: {args.validate} passes the TAP v1 schema contract "
              f"({', '.join(FILE_NAMES)}).")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
