"""Configuration classes for GRPO training.
- ``GRPOScriptArguments`` extends ``trl.ScriptArguments`` with project-specific fields.
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
class GRPOScriptArguments(ScriptArguments):
    """Project-specific script arguments for GRPO training.

    Extends trl's ``ScriptArguments`` (which provides ``dataset_name``,
    ``dataset_config``, etc.) with fields for JSONL-based local datasets
    and reward configuration.
    """

    # Override dataset_name — we use JSONL files, not HF datasets
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Not used. Provide train_file / eval_file instead."},
    )

    # Local JSONL data files
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

    # Reward function selection (not used yet — always [format, correctness])
    reward_funcs: list[str] = field(
        default_factory=lambda: ["format", "correctness"],
        metadata={"help": "Reward function names to use."},
    )

    # Data limits
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Cap on training samples (None = all)."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Cap on eval samples (None = all)."},
    )

    # Training control
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
