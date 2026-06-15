"""Qwen3-1.7B-Base LoRA GRPO training on OpenR1-Math-220k.

Usage
-----
tensorboard --logdir outputs/grpo/qwen3_1.7b_openr1_v2/runs/ --port 6006

Create tmux session:
    tmux new -s grpo_vllm

vLLM server:
    conda activate opd-vllm-trl
    CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
      --model /home/zcy/OPD/models/Qwen3-1.7B-Base \
      --host 127.0.0.1 --port 8000

    # 7-GPU training
    conda activate opd-train-vllm
    deepspeed --include localhost:1,2,3,4,5,6,7 \
      src/grpo/grpo.py \
      --config configs/grpo/qwen3_1.7b_openr1_flat.yaml --skip_confirmation \
      --output_dir outputs/grpo/qwen3_1.7b_openr1_v2

    # Dry-run
    python src/grpo/grpo.py --dry-run --config configs/grpo/qwen3_1.7b_openr1_flat.yaml

    # Single-GPU debug
    CUDA_VISIBLE_DEVICES=0 python src/grpo/grpo.py \\
        --config configs/grpo/qwen3_1.7b_openr1_flat.yaml \\
        --max_steps 5 --skip_confirmation
"""

import os
import sys
import csv
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# HF mirror
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/home/zcy/OPD/models/hf_home")
os.environ.setdefault("HF_HUB_CACHE", "/home/zcy/OPD/models/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/home/zcy/OPD/models/transformers")

from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from src.grpo.configs import GRPOScriptArguments, load_yaml_config
from src.grpo.rewards import get_reward_funcs
from src.grpo.utils import Tee, dry_run, get_dataset, get_model, get_tokenizer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser, get_peft_config

def main(script_args: GRPOScriptArguments, training_args: GRPOConfig, model_args: ModelConfig):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    is_main = training_args.local_rank == 0 or training_args.local_rank == -1
    tee = None
    if is_main:
        console_log_path = os.path.join(output_dir, "console.log")
        tee = Tee(console_log_path)
        sys.stdout = tee
        sys.stderr = tee
        print(f"Console log: {console_log_path}")

    print("=" * 60)
    print(f"LoRA GRPO Training  |  bf16={training_args.bf16}  fp16={training_args.fp16}")
    print(f"Model: {model_args.model_name_or_path}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # vLLM mode
    if training_args.use_vllm:
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            print(f"  [vLLM] mode={training_args.vllm_mode} "
                  f"server={training_args.vllm_server_host}:{training_args.vllm_server_port}")

    # Check for last checkpoint
    checkpoint = training_args.resume_from_checkpoint
    if checkpoint is None and os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is not None:
            print(f"  [INFO] Checkpoint detected, resuming at {last_checkpoint}")
            checkpoint = last_checkpoint
            
    # Load the dataset
    dataset = get_dataset(script_args)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args)

    ############
    # Load model
    ############
    print("*** Loading model ***")
    model = get_model(model_args)

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)

    #############################
    # Initialize the GRPO trainer
    #############################
    eval_available = "eval" in dataset
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval") if eval_available else None,
        peft_config=get_peft_config(model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    print("*** Train ***")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # Log and save final metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ############
    # Save model
    ############
    print("*** Save model ***")
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    final_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"  Model saved to: {final_path}")

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(output_dir)

    ######################
    # Export training logs
    ######################
    log_history = trainer.state.log_history
    if log_history and (training_args.local_rank == 0 or training_args.local_rank == -1):
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

if __name__ == "__main__":
    # --dry-run bypasses TrlParser (only needs YAML, not training deps)
    if "--dry-run" in sys.argv:
        import argparse as _ap
        _p = _ap.ArgumentParser()
        _p.add_argument("--dry-run", action="store_true")
        _p.add_argument("--config", required=True)
        _dry_args, _ = _p.parse_known_args()
        dry_run(load_yaml_config(_dry_args.config))
        sys.exit(0)

    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # vLLM fail-fast check
    if training_args.use_vllm:
        try:
            import vllm  # noqa: F401
        except ImportError:
            print("=" * 60)
            print("[ERROR] use_vllm=True but vllm is not importable.")
            print("  Install: pip install vllm")
            print("=" * 60)
            sys.exit(1)

    # Confirmation (unless --skip_confirmation)
    if not script_args.skip_confirmation:
        yn = input("Start GRPO training? (yes/no): ")
        if yn.lower() != "yes":
            sys.exit(0)

    main(script_args, training_args, model_args)
