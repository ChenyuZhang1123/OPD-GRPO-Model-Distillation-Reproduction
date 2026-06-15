#!/usr/bin/env python
"""Qwen3-1.7B-Base LoRA OPD training with Qwen3-32B teacher.

On-Policy Distillation: student generates rollouts → teacher grades every token
via reverse KL → student updates via importance-sampled policy gradient.

Usage
-----
● # ===== 终端 1: Teacher Server (GPUs 0-3) =====
  tmux new -s teacher
  conda activate opd-train
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/opd_teacher_server.py \
      --model /home/zcy/OPD/models/Qwen3-32B-Base --port 8100
  # Ctrl+B D 断开，等模型加载完（约 35s）

  # ===== 终端 2: OPD 训练 (GPUs 4-7) =====
  tmux new -s train
  conda activate opd-train
  rm -rf outputs/opd/qwen3_1.7b_opd
  deepspeed --include localhost:4,5,6,7 src/opd/opd.py \
      --config configs/opd/qwen3_1.7b_opd.yaml --skip_confirmation \
      --teacher_server_url http://127.0.0.1:8100 \
      2>&1 | tee logs/train_opd_qwen3_1.7b.log
  # Ctrl+B D 断开

  状态检查：
  curl -s http://127.0.0.1:8100/health  # teacher 是否就绪
  tmux ls                                # 查看所有 session
  tmux attach -t train                   # 重新连接训练窗口
  
Create tmux session:
    tmux new -s opd

    # Dry-run
    python src/opd/opd.py --dry-run --config configs/opd/qwen3_1.7b_opd.yaml

    # 8-GPU training with DeepSpeed ZeRO-2
    deepspeed --num_gpus=8 src/opd/opd.py \
        --config configs/opd/qwen3_1.7b_opd.yaml --skip_confirmation

    deepspeed --include localhost:4,5,6,7 src/opd/opd.py \
        --config configs/opd/qwen3_1.7b_opd.yaml --skip_confirmation \
        --teacher_server_url http://127.0.0.1:8100 \
        2>&1 | tee logs/train_opd_qwen3_1.7b.log

    # From checkpoint
    deepspeed --num_gpus=8 src/opd/opd.py \
        --config configs/opd/qwen3_1.7b_opd.yaml --skip_confirmation \
        --output_dir outputs/opd/qwen3_1.7b_opd_v2 \
        --resume_from_checkpoint outputs/opd/qwen3_1.7b_opd/checkpoint-150

Reference
---------
Thinking Machines Lab, "On-Policy Distillation" (Oct 2025)
https://thinkingmachines.ai/blog/on-policy-distillation/

  对每步训练:
    1. 从 DataLoader 获取 prompts (list[str])
    2. Tokenize prompts
    3. Student 生成 completions (num_generations 个/prompt, 无梯度)
    4. Student forward → old_logprobs (无梯度)
    5. Teacher forward → teacher_logprobs (无梯度)
    6. Student forward → new_logprobs (有梯度)
    7. advantages = teacher_logprobs - old_logprobs (逐 token reverse KL)
    8. loss = -min(ratio * A, clip(ratio) * A)  (standard PPO clipped objective)
    9. Backward + optimizer step
"""

from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# HF mirror — server cannot reach huggingface.co
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/home/zcy/OPD/models/hf_home")
os.environ.setdefault("HF_HUB_CACHE", "/home/zcy/OPD/models/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/home/zcy/OPD/models/transformers")

from dataclasses import dataclass
from typing import Any

from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, ModelConfig, TrlParser, get_peft_config

from src.opd.configs import OPDScriptArguments, load_yaml_config
from src.opd.teacher import OPDTeacher, OPDTeacherClient
from src.opd.trainer import OPDTrainer
from src.opd.utils import (
    Tee,
    get_dataset,
    get_model,
    get_tokenizer,
    patch_lr_scheduler_for_deepspeed,
)


# ============================================================================
# Data collator — preserves string prompt column
# ============================================================================

@dataclass
class OPDDataCollator:
    """Data collator that preserves string columns for OPD.

    The HF default collator would stack everything into tensors.
    We need prompt strings to survive as Python strings for tokenization
    inside ``compute_loss``.
    """

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = {}
        for key in features[0].keys():
            values = [f[key] for f in features]
            # Keep strings / lists as-is; numerical values become tensors
            if isinstance(values[0], str):
                batch[key] = values  # list of strings
            elif isinstance(values[0], (int, float)):
                import torch
                batch[key] = torch.tensor(values)
            else:
                batch[key] = values
        return batch


# ============================================================================
# Main
# ============================================================================

def main(script_args: OPDScriptArguments, training_args: GRPOConfig, model_args: ModelConfig):
    # ---- Seed ----
    set_seed(training_args.seed)

    # ---- Patch DeepSpeed / PyTorch >=2.12 compat ----
    patch_lr_scheduler_for_deepspeed()

    # ---- Output dir & logging ----
    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    is_main = training_args.local_rank in (0, -1)
    tee = None
    if is_main:
        console_log_path = os.path.join(output_dir, "console.log")
        tee = Tee(console_log_path)
        sys.stdout = tee
        sys.stderr = tee
        print(f"Console log: {console_log_path}")

    print("=" * 60)
    print(f"OPD Training  |  bf16={training_args.bf16}")
    print(f"  Student: {model_args.model_name_or_path}")
    print(f"  Teacher: {script_args.teacher_model_path}")
    print(f"  Output:  {output_dir}")
    print("=" * 60)

    # ---- Dataset ----
    dataset = get_dataset(script_args)

    # ---- Tokenizer ----
    tokenizer = get_tokenizer(model_args)

    # ---- Student model (with LoRA) ----
    print("*** Loading student model ***")
    model = get_model(model_args)

    # Apply PEFT / LoRA before passing to Trainer (base Trainer doesn't accept peft_config)
    peft_config = get_peft_config(model_args)
    if peft_config is not None:
        from peft import get_peft_model
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # ---- Teacher model (local or remote) ----
    if script_args.teacher_server_url:
        print("*** Connecting to remote teacher server ***")
        teacher = OPDTeacherClient(script_args.teacher_server_url)
    else:
        print("*** Loading teacher model (local) ***")
        teacher = OPDTeacher(
            model_path=script_args.teacher_model_path,
            torch_dtype=script_args.teacher_dtype,
            device_map=script_args.teacher_device_map,
            load_in_4bit=script_args.teacher_load_in_4bit,
        )

    # ---- Generation kwargs ----
    generation_kwargs = {
        "max_new_tokens": getattr(training_args, "max_completion_length", 1792),
        "max_prompt_length": script_args.max_prompt_length,
        "num_generations": getattr(training_args, "num_generations", 4),
        "temperature": getattr(training_args, "temperature", 1.0),
        "top_p": getattr(training_args, "top_p", 1.0),
        "clip_epsilon": script_args.clip_epsilon,
    }

    # ---- Checkpoint detection (before Trainer init) ----
    checkpoint = training_args.resume_from_checkpoint
    if checkpoint is None and os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is not None:
            print(f"  [INFO] Checkpoint detected, resuming at {last_checkpoint}")
            checkpoint = last_checkpoint

    # ---- Data collator ----
    data_collator = OPDDataCollator()

    # ---- OPD Trainer ----
    trainer = OPDTrainer(
        teacher=teacher,
        tokenizer=tokenizer,
        generation_kwargs=generation_kwargs,
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        data_collator=data_collator,
    )

    # ---- Train ----
    print("*** Train ***")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # ---- Log & save metrics ----
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # ---- Save final model ----
    if is_main:
        print("*** Save model ***")
        trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
        final_path = os.path.join(output_dir, "final_model")
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"  Model saved to: {final_path}")

        # Restore cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(output_dir)

        # Export training logs
        log_history = trainer.state.log_history
        if log_history:
            with open(os.path.join(output_dir, "train_log.json"), "w") as f:
                json.dump(log_history, f, indent=2)

            fieldnames = list(dict.fromkeys(k for row in log_history for k in row))
            with open(os.path.join(output_dir, "train_log.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(log_history)

            print(f"  Logs saved: {output_dir}/train_log.json, train_log.csv")

    if tee is not None:
        sys.stdout = tee.stdout
        sys.stderr = tee.stderr
        tee.close()

    print("Done.")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    # --dry-run bypasses TrlParser (only needs YAML, not training deps)
    if "--dry-run" in sys.argv:
        from src.opd.utils import dry_run as _dry_run_opd

        import argparse as _ap
        _p = _ap.ArgumentParser()
        _p.add_argument("--dry-run", action="store_true")
        _p.add_argument("--config", required=True)
        _dry_args, _ = _p.parse_known_args()
        cfg = load_yaml_config(_dry_args.config)

        print("=" * 60)
        print("OPD DRY RUN")
        print(f"  Config:       {_dry_args.config}")
        print("=" * 60)
        _dry_run_opd(cfg)
        sys.exit(0)

    parser = TrlParser((OPDScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Confirmation (unless --skip_confirmation)
    if not script_args.skip_confirmation:
        yn = input("Start OPD training? (yes/no): ")
        if yn.lower() != "yes":
            sys.exit(0)

    main(script_args, training_args, model_args)
