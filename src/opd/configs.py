"""Configuration classes for OPD training.

Follows the open-r1 pattern exactly:
- ``OPDScriptArguments`` extends ``GRPOScriptArguments`` with OPD-specific fields.
- ``GRPOConfig`` and ``ModelConfig`` are trl's built-in dataclasses (parsed by TrlParser).
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml
from trl import ScriptArguments


# ============================================================================
# Script arguments (extends trl.ScriptArguments, parsed by TrlParser)
# ============================================================================

@dataclass
class OPDScriptArguments(ScriptArguments):
    """Project-specific script arguments for OPD training.

    Extends trl's ``ScriptArguments`` with fields for JSONL-based local datasets,
    teacher model configuration, and OPD-specific generation parameters.
    """

    # Override dataset_name — we use JSONL files, not HF datasets
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Not used. Provide train_file / eval_file instead."},
    )

    # ---- Local JSONL data files ----
    train_file: str = field(
        default="data/processed/openr1_math_grpo_train.jsonl",
        metadata={"help": "Path to JSONL training data file."},
    )
    eval_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to JSONL eval data file."},
    )
    dataset_prompt_column: str = field(
        default="prompt",
        metadata={"help": "Column name for the prompt text."},
    )
    dataset_answer_column: str = field(
        default="answer",
        metadata={"help": "Column name for the reference answer."},
    )

    # ---- OPD-specific parameters ----
    max_prompt_length: int = field(
        default=2048,
        metadata={"help": "Max token length for prompt truncation."},
    )
    clip_epsilon: float = field(
        default=0.2,
        metadata={"help": "PPO clipping epsilon for importance sampling loss."},
    )

    # ---- Teacher model ----
    teacher_model_path: str = field(
        default="/home/zcy/OPD/models/Qwen3-32B-Base",
        metadata={"help": "Path to teacher model for OPD logprob supervision."},
    )
    teacher_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Torch dtype for teacher model inference."},
    )
    teacher_device_map: str = field(
        default="auto",
        metadata={"help": "Device map strategy for teacher model (auto, sequential, etc.)."},
    )
    teacher_load_in_4bit: bool = field(
        default=True,
        metadata={"help": "Load teacher model in 4-bit (NF4) quantization to save GPU memory."},
    )

    # ---- Data limits ----
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Cap on training samples (None = all)."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Cap on eval samples (None = all)."},
    )

    # ---- Training control ----
    dry_run: bool = field(
        default=False,
        metadata={"help": "Print config summary and exit without training."},
    )
    skip_confirmation: bool = field(
        default=False,
        metadata={"help": "Skip the 'Start training?' prompt."},
    )


# ============================================================================
# YAML loading helper (used by dry-run, which bypasses TrlParser)
# ============================================================================

def load_yaml_config(path: str) -> dict:
    """Load a YAML config file, resolving relative paths against the project root."""
    if not os.path.isabs(path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        path = os.path.join(project_root, path)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
