"""Utility functions for GRPO training.
Data loading, model/tokenizer loading, logging, and DeepSpeed compatibility.
"""

import json
import os
import sys
from typing import List, Optional

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from trl import ModelConfig


# ============================================================================
# Data loading
# ============================================================================

def load_jsonl(path: str, limit: Optional[int] = None) -> List[dict]:
    """Load records from a JSONL file, optionally capped at *limit*."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def get_dataset(script_args) -> DatasetDict:
    """Load train and eval datasets from JSONL files.
       all data config lives in *script_args*.
    """
    train_path = script_args.train_file
    if not os.path.isabs(train_path):
        train_path = os.path.join(os.getcwd(), train_path)
    train_dataset = Dataset.from_list(load_jsonl(train_path, limit=script_args.max_train_samples))

    eval_dataset = None
    if script_args.eval_file:
        eval_path = script_args.eval_file
        if not os.path.isabs(eval_path):
            eval_path = os.path.join(os.getcwd(), eval_path)
        eval_dataset = Dataset.from_list(load_jsonl(eval_path, limit=script_args.max_eval_samples))

    result = {"train": train_dataset}
    if eval_dataset is not None:
        result["eval"] = eval_dataset
    return DatasetDict(result)


# ============================================================================
# Model / tokenizer loading
# ============================================================================

def get_tokenizer(model_config: ModelConfig) -> PreTrainedTokenizer:
    """Load tokenizer and set pad_token."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def get_model(model_config: ModelConfig) -> AutoModelForCausalLM:
    """Load the base CausalLM model for LoRA training.
    *training_args* is accepted but not needed for LoRA (gradient checkpointing
    is read from *model_config*).
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
# Logging
# ============================================================================

class Tee:
    """Write to a file and the original stdout/stderr simultaneously."""

    def __init__(self, file_path: str):
        self.file = open(file_path, "w", buffering=1)
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()


# ============================================================================
# DeepSpeed + PyTorch >=2.12 scheduler compatibility
# ============================================================================

def patch_lr_scheduler_for_deepspeed():
    """Patch LRScheduler._update_lr for PyTorch >=2.12 + DeepSpeed compat.

    PyTorch >=2.12 made LRScheduler._update_lr use strict=True in zip(),
    which crashes when DeepSpeed replaces the optimizer after scheduler
    creation.
    """
    import torch.optim.lr_scheduler as _lrsched_module

    def _patched_update_lr(self, values):
        if not getattr(self, "optimizer", None):
            return
        if values is not None:
            for param_group, lr in zip(self.optimizer.param_groups, values):
                param_group["lr"] = lr
        self._last_lr = [
            group.get("lr", 0.0) for group in self.optimizer.param_groups
        ]

    _lrsched_module.LRScheduler._update_lr = _patched_update_lr


# ============================================================================
# Dry-run
# ============================================================================

def dry_run(config: dict):
    """Print config summary and dataset statistics without starting training.

    Reads a flat YAML config dict (open-r1 style).
    """
    from transformers import AutoTokenizer

    model_name_or_path = config["model_name_or_path"]
    train_file = config["train_file"]
    eval_file = config.get("eval_file", "")
    output_dir = config.get("output_dir", "outputs/grpo/default")
    use_vllm = config.get("use_vllm", False)

    print("=" * 60)
    print("GRPO DRY RUN")
    print(f"  Model:       {model_name_or_path}")
    print(f"  Train data:  {train_file}")
    print(f"  Eval data:   {eval_file}")
    print(f"  Output dir:  {output_dir}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True)

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    train_path = train_file
    if not os.path.isabs(train_path):
        train_path = os.path.join(project_root, train_path)

    max_train = config.get("max_train_samples")
    records = load_jsonl(train_path, limit=max_train)
    print(f"\n  Train records loaded: {len(records)}")
    if records:
        print(f"  Fields: {list(records[0].keys())}")
        print(f"\n  Sample 0 prompt (first 200 chars):")
        print(f"    {records[0]['prompt'][:200]}...")
        print(f"  Sample 0 answer:")
        print(f"    {records[0]['answer']}")

    if records:
        lengths = []
        for r in records:
            lengths.append(len(tokenizer.encode(r["prompt"])))
        sorted_lens = sorted(lengths)
        def pct(arr, p):
            return arr[min(int(len(arr) * p / 100), len(arr) - 1)]
        print(f"\n  Prompt token stats (n={len(records)}): "
              f"min={min(lengths)} p50={pct(sorted_lens, 50)} "
              f"p95={pct(sorted_lens, 95)} max={max(lengths)}")

    per_device = config.get("per_device_train_batch_size", 2)
    grad_acc = config.get("gradient_accumulation_steps", 1)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    num_gen = config.get("num_generations", 8)
    max_steps_val = config.get("max_steps", -1)
    gen_batch = config.get("generation_batch_size", per_device * world_size)

    effective_train_batch = per_device * grad_acc * world_size
    distinct_prompts_per_update = effective_train_batch // num_gen

    print(f"\n  --- Training budget ---")
    print(f"  world_size:                     {world_size}")
    print(f"  per_device_train_batch_size:    {per_device}")
    print(f"  gradient_accumulation_steps:    {grad_acc}")
    print(f"  num_generations:                {num_gen}")
    print(f"  generation_batch_size:          {gen_batch}")
    print(f"  effective_train_batch:          {effective_train_batch} completions/update")
    print(f"    = {world_size} x {per_device} x {grad_acc}")
    print(f"  distinct_prompts_per_update:    {distinct_prompts_per_update}")
    print(f"    = {effective_train_batch} / {num_gen}")

    if effective_train_batch % num_gen != 0:
        print(f"  [WARN] effective_train_batch ({effective_train_batch}) must be "
              f"divisible by num_generations ({num_gen}).")

    if max_steps_val > 0:
        import math as _math
        num_samples = len(records) if records else config.get("max_train_samples", 0)
        estimated_prompts = distinct_prompts_per_update * max_steps_val
        steps_per_epoch = _math.ceil(num_samples / distinct_prompts_per_update) if distinct_prompts_per_update > 0 and num_samples > 0 else "N/A"
        epoch_fraction = estimated_prompts / num_samples if num_samples > 0 else 0

        print(f"  max_steps:                      {max_steps_val}")
        print(f"  estimated_distinct_prompts:     {estimated_prompts}")
        print(f"  num_train_samples:              {num_samples}")
        print(f"  steps_per_epoch:                {steps_per_epoch}")
        if isinstance(epoch_fraction, float):
            print(f"  estimated_epoch_fraction:       {epoch_fraction:.2f}")

    print(f"  max_completion_length:          {config.get('max_completion_length', 'N/A')}")

    print(f"\n  --- vLLM ---")
    print(f"  use_vllm:                       {use_vllm}")
    if use_vllm:
        vllm_mode = config.get("vllm_mode", "colocate")
        print(f"  vllm_mode:                      {vllm_mode}")
        if vllm_mode == "server":
            print(f"  vllm_server_host:               {config.get('vllm_server_host', '127.0.0.1')}")
            print(f"  vllm_server_port:               {config.get('vllm_server_port', 8000)}")
        print(f"  vllm_gpu_memory_utilization:    {config.get('vllm_gpu_memory_utilization', 0.3)}")
        print(f"  vllm_tensor_parallel_size:      {config.get('vllm_tensor_parallel_size', 1)}")

    if config.get("use_peft"):
        print(f"\n  --- LoRA ---")
        print(f"  lora_r:                         {config.get('lora_r', 'N/A')}")
        print(f"  lora_alpha:                     {config.get('lora_alpha', 'N/A')}")
        print(f"  lora_dropout:                   {config.get('lora_dropout', 'N/A')}")
        print(f"  lora_target_modules:            {config.get('lora_target_modules', 'N/A')}")

    ds_path = config.get("deepspeed")
    if ds_path:
        print(f"\n  --- DeepSpeed ---")
        print(f"  config:                         {ds_path}")

    print("=" * 60)
