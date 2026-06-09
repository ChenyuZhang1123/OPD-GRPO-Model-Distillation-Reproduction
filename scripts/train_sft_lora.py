"""Qwen3-8B-Base LoRA SFT 训练脚本。

用法:
    python -u scripts/train_sft_lora.py --config configs/sft/qwen3_8b_lora_stage1.yaml --yes
    python -u scripts/train_sft_lora.py --config ...  --dry-run --max-samples 100
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


def format_sample(record: dict) -> str:
    prompt = record.get("prompt", "")
    response = record.get("response", record.get("completion", ""))
    return prompt + response


def _print_token_stats(records, tokenizer, max_seq):
    lengths = [len(tokenizer.encode(format_sample(r))) for r in records]
    sorted_lens = sorted(lengths)
    pct = lambda arr, p: arr[min(int(len(arr) * p / 100), len(arr) - 1)]

    over = sum(1 for t in lengths if t > max_seq)
    print(f"  Token stats (n={len(records)}): "
          f"min={min(lengths)} p50={pct(sorted_lens, 50)} "
          f"p95={pct(sorted_lens, 95)} max={max(lengths)}")
    print(f"  Samples > max_seq ({max_seq}): {over}/{len(records)} ({over/len(records)*100:.1f}%)")


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
        n_prompt = len(tokenizer.encode(rec.get("prompt", "")))
        n_resp = len(tokenizer.encode(rec.get("response", "")))
        print(f"\n  sample {i+1}: source={rec.get('source')} "
              f"dataset={rec.get('dataset')} subject={rec.get('subject','-')} level={rec.get('level','-')}")
        print(f"  tokens: prompt={n_prompt} response={n_resp} total={n_prompt+n_resp}")
        print(f"  prompt:   {rec.get('prompt','')[:100]}...")
        print(f"  response: {rec.get('response','')[:100]}...")

    _print_token_stats(records, tokenizer, data_cfg.get("max_seq_length", 2048))

    bs = training_cfg["per_device_train_batch_size"] * training_cfg["gradient_accumulation_steps"]
    total_samples = len(load_jsonl(data_cfg["train_file"]))
    steps = total_samples // bs
    print(f"\n  Full dataset: {total_samples} samples, batch_size={bs}, ~{steps} steps/epoch")
    print("=" * 60)


def train(config: dict):
    import torch
    import tempfile
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, TaskType
    from trl import SFTTrainer
    from datasets import Dataset

    model_cfg = config["model"]
    data_cfg = config["data"]
    lora_cfg = config["lora"]
    training_cfg = config["training"]
    output_cfg = config["output"]
    ds_cfg = config.get("deepspeed", {})

    torch_dtype = torch.bfloat16 if training_cfg.get("bf16", True) else torch.float16
    use_ds = bool(ds_cfg and "zero_optimization" in ds_cfg)

    print("=" * 60)
    print(f"LoRA SFT Training  |  bf16={training_cfg.get('bf16')}  deepspeed={use_ds}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"], torch_dtype=torch_dtype, trust_remote_code=True,
        device_map=None if use_ds else "auto")
    if model_cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"], target_modules=lora_cfg["target_modules"])
    print(f"  LoRA r={lora_cfg['r']} alpha={lora_cfg['alpha']} "
          f"trainable≈{lora_cfg['r'] * 2 * len(lora_cfg['target_modules']) / 1000:.0f}M params")

    ds_config_path = None
    if use_ds:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(ds_cfg, f, indent=2)
            ds_config_path = f.name

    training_args = TrainingArguments(
        output_dir=output_cfg["dir"],
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        warmup_steps=training_cfg.get("warmup_steps", 0),
        weight_decay=training_cfg["weight_decay"],
        optim=training_cfg["optim"],
        bf16=training_cfg.get("bf16", True),
        fp16=training_cfg.get("fp16", False),
        logging_steps=training_cfg["logging_steps"],
        save_strategy=training_cfg["save_strategy"],
        save_steps=training_cfg["save_steps"],
        save_total_limit=training_cfg["save_total_limit"],
        max_steps=training_cfg.get("max_steps", -1),
        max_grad_norm=training_cfg["max_grad_norm"],
        seed=training_cfg["seed"],
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 2),
        deepspeed=ds_config_path,
        report_to="none",
        remove_unused_columns=False,
    )

    limit = data_cfg.get("max_samples") or training_cfg.get("max_samples")
    records = load_jsonl(data_cfg["train_file"], limit=limit)
    for r in records:
        r["completion"] = r.pop("response", "")
    dataset = Dataset.from_list(records)
    tokenizer.model_max_length = data_cfg["max_seq_length"]

    print(f"  Dataset: {len(dataset)} samples, max_seq_length={data_cfg['max_seq_length']}")
    trainer = SFTTrainer(model=model, processing_class=tokenizer, args=training_args,
                         train_dataset=dataset, peft_config=peft_config)

    trainer.train()

    final_path = os.path.join(output_cfg["dir"], "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"  Model saved to: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-8B-Base LoRA SFT")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--yes", action="store_true")
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
        train(config)


if __name__ == "__main__":
    main()
