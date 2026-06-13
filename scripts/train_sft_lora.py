"""Qwen3-1.7B-Base LoRA SFT 训练脚本。

用法:
    # Dry-run
    python scripts/train_sft_lora.py --config configs/sft/qwen3_1.7b_lora_stage1_v1.yaml --dry-run

    # 单 GPU
    CUDA_VISIBLE_DEVICES=0 python scripts/train_sft_lora.py \
        --config configs/sft/qwen3_1.7b_lora_stage1_v1.yaml --yes

    # 3 GPU DeepSpeed
    CUDA_VISIBLE_DEVICES=5,6,7 deepspeed --num_gpus=3 scripts/train_sft_lora.py \
        --config configs/sft/qwen3_1.7b_lora_stage1_v1.yaml --yes

    # 3 GPU DeepSpeed (指定 master_port)
    deepspeed --include localhost:5,6,7 --master_port 29501 scripts/train_sft_lora.py \
        --config configs/sft/qwen3_1.7b_lora_stage1_v1.yaml --yes

    # 8 GPU DeepSpeed
    deepspeed --num_gpus=8 scripts/train_sft_lora.py \
    --config configs/sft/qwen3_1.7b_lora_stage1_v3.yaml --yes

TensorBoard:
        服务器 tensorboard --logdir outputs/sft/qwen3_1.7b_lora_stage1_v1 --bind_all
        本机运行ssh -L 6006:localhost:6006 210.45.70.163
        本地浏览器访问 http://localhost:6006。
"""

import argparse
import json
import os
import sys
import yaml

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/home/zcy/OPD/models/hf_home"
os.environ["HF_HUB_CACHE"] = "/home/zcy/OPD/models/hub"
os.environ["TRANSFORMERS_CACHE"] = "/home/zcy/OPD/models/transformers"


def load_jsonl(path: str, limit: int = None) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


# ---- Shared formatting (dry-run and train use the same logic) ----

def _prepare_record(record: dict, tokenizer) -> tuple[str, str]:
    """Return (prompt, completion) ready for training.

    A space is always prepended to completion to ensure a BPE token boundary
    between prompt and completion.  When the space lives on the completion
    side, tokenize(prompt) matches the prefix of tokenize(prompt + completion)
    because the space is absorbed into the first completion token.
    """
    prompt = record.get("prompt", "")
    response = record.get("response", record.get("completion", ""))
    completion = " " + response.rstrip() + tokenizer.eos_token
    return prompt, completion


def _format_dataset(records: list, tokenizer) -> "Dataset":
    """Apply _prepare_record to all records and return a HuggingFace Dataset."""
    from datasets import Dataset
    for r in records:
        r["prompt"], r["completion"] = _prepare_record(r, tokenizer)
    return Dataset.from_list(records)


def _print_token_stats(records, tokenizer, max_seq):
    lengths = []
    mismatches = 0
    for r in records:
        prompt, completion = _prepare_record(r, tokenizer)
        full = prompt + completion
        lengths.append(len(tokenizer.encode(full)))

        prompt_ids = tokenizer.encode(prompt)
        full_ids = tokenizer.encode(full)
        if full_ids[:len(prompt_ids)] != prompt_ids:
            mismatches += 1

    sorted_lens = sorted(lengths)
    pct = lambda arr, p: arr[min(int(len(arr) * p / 100), len(arr) - 1)]

    over = sum(1 for t in lengths if t > max_seq)
    print(f"  Token stats (n={len(records)}): "
          f"min={min(lengths)} p50={pct(sorted_lens, 50)} "
          f"p95={pct(sorted_lens, 95)} max={max(lengths)}")
    print(f"  Samples > max_seq ({max_seq}): {over}/{len(records)} ({over/len(records)*100:.1f}%)")
    print(f"  Prompt-token mismatch: {mismatches}/{len(records)} ({mismatches/len(records)*100:.1f}%)")


def dry_run(config: dict, max_samples: int = 100):
    from transformers import AutoTokenizer

    model_cfg = config["model"]
    data_cfg = config["data"]
    training_cfg = config["training"]

    print("=" * 60)
    print("DRY RUN")
    print(f"  Tokenizer: {model_cfg['name_or_path']}")
    print(f"  Data:      {data_cfg['train_file']}")
    print(f"  Samples:   {max_samples}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"], trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = load_jsonl(data_cfg["train_file"], limit=max_samples)
    print(f"  Loaded {len(records)} records")

    for i, rec in enumerate(records[:3]):
        prompt, completion = _prepare_record(rec, tokenizer)
        n_prompt = len(tokenizer.encode(prompt))
        n_comp = len(tokenizer.encode(completion))
        print(f"\n  sample {i+1}: source={rec.get('source')} "
              f"dataset={rec.get('dataset')} subject={rec.get('subject','-')} level={rec.get('level','-')}")
        print(f"  tokens: prompt={n_prompt} completion={n_comp} total={n_prompt+n_comp}")
        print(f"  prompt:     {prompt[:100]}...")
        print(f"  completion: {completion[:100]}...")

    _print_token_stats(records, tokenizer, data_cfg.get("max_seq_length", 2048))

    per_device = training_cfg["per_device_train_batch_size"]
    grad_acc = training_cfg["gradient_accumulation_steps"]
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_bs = per_device * grad_acc * world_size
    total_samples = len(load_jsonl(data_cfg["train_file"]))
    steps = total_samples // global_bs
    print(f"\n  Full dataset: {total_samples} samples")
    print(f"  Global batch: {per_device} x {grad_acc} x {world_size} GPU(s) = {global_bs}")
    print(f"  Steps/epoch:  ~{steps}")
    print("=" * 60)


def train(config: dict, max_samples: int = None):
    import torch
    import tempfile
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, TaskType
    from trl import SFTTrainer, SFTConfig

    model_cfg = config["model"]
    data_cfg = config["data"]
    lora_cfg = config["lora"]
    training_cfg = config["training"]
    output_cfg = config["output"]
    ds_cfg = config.get("deepspeed", {})

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ds = bool(ds_cfg and "zero_optimization" in ds_cfg and world_size > 1)

    # ---- dtype ----
    want_bf16 = training_cfg.get("bf16", True)
    bf16_ok = want_bf16 and torch.cuda.is_bf16_supported()
    use_bf16 = bf16_ok
    use_fp16 = not bf16_ok
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print("=" * 60)
    print(f"LoRA SFT Training  |  bf16={use_bf16}  fp16={use_fp16}  world_size={world_size}  deepspeed={use_ds}")
    print("=" * 60)

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- model ----
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"], torch_dtype=torch_dtype, trust_remote_code=True,
        device_map=None)

    if model_cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # ---- data ----
    limit = max_samples or data_cfg.get("max_samples") or training_cfg.get("max_samples")
    records = load_jsonl(data_cfg["train_file"], limit=limit)
    train_dataset = _format_dataset(records, tokenizer)
    tokenizer.model_max_length = data_cfg["max_seq_length"]

    eval_file = data_cfg.get("eval_file")
    if eval_file:
        if not os.path.isabs(eval_file):
            eval_file = os.path.join(project_root, eval_file)
        eval_records = load_jsonl(eval_file)
        eval_dataset = _format_dataset(eval_records, tokenizer)
    else:
        eval_dataset = None

    print(f"  Train: {len(train_dataset)} samples, max_seq_length={data_cfg['max_seq_length']}")
    if eval_dataset is not None:
        print(f"  Eval:  {len(eval_dataset)} samples ({len(eval_dataset)/len(train_dataset)*100:.1f}% of train)")

    # ---- LoRA ----
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"], target_modules=lora_cfg["target_modules"])

    # ---- DeepSpeed config (sync dtype with actual) ----
    ds_config_path = None
    if use_ds:
        ds_cfg = json.loads(json.dumps(ds_cfg))  # deep copy
        ds_cfg["bf16"]["enabled"] = use_bf16
        ds_cfg["fp16"] = ds_cfg.get("fp16", {})
        ds_cfg["fp16"]["enabled"] = use_fp16
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(ds_cfg, f, indent=2)
            ds_config_path = f.name

    # ---- training args (SFTConfig) ----
    training_args = SFTConfig(
        output_dir=output_cfg["dir"],
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        warmup_steps=training_cfg.get("warmup_steps", 0),
        weight_decay=training_cfg["weight_decay"],
        optim=training_cfg["optim"],
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=training_cfg["logging_steps"],
        eval_strategy=training_cfg.get("eval_strategy", "no"),
        eval_steps=training_cfg.get("eval_steps"),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 8),
        save_strategy=training_cfg.get("save_strategy", "steps"),
        save_steps=training_cfg.get("save_steps", 500),
        save_total_limit=training_cfg.get("save_total_limit", 2),
        max_steps=training_cfg.get("max_steps", -1),
        max_grad_norm=training_cfg["max_grad_norm"],
        seed=training_cfg["seed"],
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 2),
        deepspeed=ds_config_path,
        report_to="tensorboard",
        remove_unused_columns=False,
        max_length=data_cfg["max_seq_length"],
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset, peft_config=peft_config)

    # ---- param count (after PEFT applied) ----
    total = sum(p.numel() for p in trainer.model.parameters())
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"  LoRA r={lora_cfg['r']} alpha={lora_cfg['alpha']} "
          f"trainable={trainable/1e6:.2f}M / total={total/1e9:.2f}B "
          f"({100*trainable/total:.2f}%)")

    trainer.train()

    # ---- save training logs (rank 0 only) ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    log_history = trainer.state.log_history
    if log_history and local_rank == 0:
        import csv
        log_dir = output_cfg["dir"]
        os.makedirs(log_dir, exist_ok=True)

        # JSON
        with open(os.path.join(log_dir, "train_log.json"), "w") as f:
            json.dump(log_history, f, indent=2)

        # CSV (auto-detect fields from all rows)
        fieldnames = list(dict.fromkeys(k for row in log_history for k in row))
        with open(os.path.join(log_dir, "train_log.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(log_history)

        print(f"  Logs saved: {log_dir}/train_log.json, train_log.csv")

    final_path = os.path.join(output_cfg["dir"], "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"  Model saved to: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-1.7B-Base LoRA SFT")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--local-rank", type=int, default=-1, dest="local_rank")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(project_root, args.config)
    config = yaml.safe_load(open(config_path))

    data_file = config["data"]["train_file"]
    if not os.path.isabs(data_file):
        config["data"]["train_file"] = os.path.join(project_root, data_file)

    if args.dry_run:
        dry_run(config, max_samples=args.max_samples or 100)
    else:
        if not args.yes:
            yn = input("Start full training? (yes/no): ")
            if yn.lower() != "yes":
                sys.exit(0)
        train(config, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
