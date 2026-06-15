"""Utility functions for OPD training.

Data loading, model/tokenizer loading, logging, and DeepSpeed compatibility.
Mirrors ``src/grpo/utils.py`` — shared functions are imported from grpo.utils
to avoid duplication.
"""

import json
import os
import sys
from typing import List, Optional

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from trl import ModelConfig

# Re-export shared utilities from grpo.utils
from src.grpo.utils import (  # noqa: F401 (re-exported for convenience)
    Tee,
    load_jsonl,
    patch_lr_scheduler_for_deepspeed,
)
from src.grpo.utils import dry_run as _grpo_dry_run


# ============================================================================
# Data loading
# ============================================================================

def get_dataset(script_args) -> DatasetDict:
    """Load train and eval datasets from JSONL files.

    Same signature as open-r1 / grpo.utils.
    """
    train_path = script_args.train_file
    if not os.path.isabs(train_path):
        train_path = os.path.join(os.getcwd(), train_path)
    train_dataset = Dataset.from_list(
        load_jsonl(train_path, limit=script_args.max_train_samples)
    )

    eval_dataset = None
    if script_args.eval_file:
        eval_path = script_args.eval_file
        if not os.path.isabs(eval_path):
            eval_path = os.path.join(os.getcwd(), eval_path)
        eval_dataset = Dataset.from_list(
            load_jsonl(eval_path, limit=script_args.max_eval_samples)
        )

    result = {"train": train_dataset}
    if eval_dataset is not None:
        result["eval"] = eval_dataset
    return DatasetDict(result)


# ============================================================================
# Model / tokenizer loading
# ============================================================================

def get_tokenizer(model_config: ModelConfig) -> PreTrainedTokenizer:
    """Load tokenizer and set pad_token + left-padding for correct generation."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    # Left-padding is required for decoder-only generation:
    # the model must see the prompt immediately before generating, not PAD tokens.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def get_model(model_config: ModelConfig) -> AutoModelForCausalLM:
    """Load the base CausalLM model for LoRA training.

    Same signature as open-r1's ``get_model(model_args, training_args)``.
    """
    dtype = model_config.dtype
    torch_dtype = getattr(torch, dtype) if dtype not in ("auto", None) else None

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=model_config.trust_remote_code,
        device_map=None,
    )

    if getattr(model_config, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return model


# ============================================================================
# OPD-specific logprob helpers
# ============================================================================

@torch.no_grad()
def compute_student_logprobs(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token log-probabilities from the student model.

    Parameters
    ----------
    model : AutoModelForCausalLM
        Student model (LoRA adapter applied).
    input_ids : (B, L)
        Full sequences (prompt + completion).
    attention_mask : (B, L)

    Returns
    -------
    (B, L) per-token logprobs, with first position padded to 0.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, L, V)

    logprobs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    token_logprobs = logprobs[:, :-1, :].gather(
        dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)

    bsz = token_logprobs.shape[0]
    return torch.nn.functional.pad(token_logprobs, (1, 0), value=0.0)


def extract_completion_slice(
    input_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Slice out the completion-only portion of a tensor.

    Given a tensor of shape ``(B, L)`` (or ``(B, L, *)``), returns the slice
    corresponding to completion tokens (positions >= *prompt_len*).

    Parameters
    ----------
    input_ids : Tensor
    prompt_len : int
        Number of prompt tokens (the completion starts at this index).

    Returns
    -------
    Tensor of same shape but trimmed to completion positions.
    """
    return input_ids[:, prompt_len:]


# ============================================================================
# Dry-run (extended with OPD-specific info)
# ============================================================================

def dry_run(config: dict):
    """Print config summary including OPD-specific fields (teacher, reverse KL, etc.).

    Calls the GRPO dry_run for shared fields, then prints OPD additions.
    """
    # Shared fields via GRPO dry_run
    _grpo_dry_run(config)

    # OPD-specific additions
    teacher_path = config.get("teacher_model_path", "N/A")
    teacher_dtype = config.get("teacher_dtype", "bfloat16")
    teacher_device = config.get("teacher_device_map", "auto")
    clip_eps = config.get("clip_epsilon", 0.2)

    print(f"\n  --- OPD-specific ---")
    print(f"  teacher_model_path:             {teacher_path}")
    print(f"  teacher_dtype:                  {teacher_dtype}")
    print(f"  teacher_device_map:             {teacher_device}")
    print(f"  clip_epsilon:                   {clip_eps}")
    print(f"  loss: per-token reverse KL → clipped importance sampling")
    print("=" * 60)
