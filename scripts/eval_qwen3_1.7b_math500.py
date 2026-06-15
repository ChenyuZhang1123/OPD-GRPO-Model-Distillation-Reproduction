"""
Qwen3-1.7B-Base MATH500 批量推理评测。
支持 base model 直接推理，也支持加载 LoRA adapter 评测 SFT/OPD/GRPO checkpoint。

Prompt 格式与 GRPO train prompt 一致:
  Question:
  {problem}

  Please solve the problem step by step and put the final answer in \\boxed{}.

  Answer:

用法
  # 1. 评估 base model
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_qwen3_1.7b_math500.py \
    --batch-size 32 --max-new-tokens 2048 --output-dir outputs/eval/qwen3_1.7b_openr1_v2/base_model \
    > outputs/eval/qwen3_1.7b_openr1_v2/base_model/eval.log 2>&1
    
  # 2. 评估 LoRA checkpoint
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_qwen3_1.7b_math500.py \
    --adapter outputs/sft/qwen3_1.7b_lora_stage1_v3/final_model \
    --batch-size 32 --max-new-tokens 2048

  # 3. 评估 GRPO checkpoint
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_qwen3_1.7b_math500.py \
    --adapter outputs/grpo/qwen3_1.7b_openr1_v2/final_model \
    --batch-size 32 --max-new-tokens 2048

  # 4. 评估中间 checkpoint
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_qwen3_1.7b_math500.py \
    --adapter outputs/grpo/qwen3_1.7b_openr1_v2/checkpoint-900 \
    --batch-size 32 --max-new-tokens 2048
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval.answer_extraction import extract_and_match

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
MODEL_PATH = "/home/zcy/OPD/models/Qwen3-1.7B-Base"
# GRPO train prompt format (must match data/processed/openr1_math_grpo_train.jsonl)
PROMPT_PREFIX = "Question:\n"
PROMPT_SUFFIX = "\n\nPlease solve the problem step by step and put the final answer in \\boxed{}.\n\nAnswer:"


def main():
    parser = argparse.ArgumentParser(description="Qwen3-1.7B-Base MATH500 Evaluation")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--adapter", default=None, help="LoRA adapter path (optional)")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--output-name", default=None,
                        help="Custom output filename prefix. Default: auto-detect "
                             "from --adapter or use 'qwen3_1.7b_math500' for base model.")
    args = parser.parse_args()

    use_shard = args.shard_id is not None and args.num_shards is not None

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ---- output paths (with optional shard suffix) ----
    # Auto-detect output name from adapter path, or use custom name
    if args.output_name:
        output_name = args.output_name
    elif args.adapter:
        # Derive name from adapter path: strip trailing / and take last component
        adapter_tag = args.adapter.rstrip('/').split('/')[-1]
        output_name = f"qwen3_1.7b_adapter_{adapter_tag}_math500"
    else:
        output_name = "qwen3_1.7b_math500"

    suffix = f"_s{args.shard_id}of{args.num_shards}" if use_shard else ""
    raw_path = os.path.join(out_dir, f"{output_name}_raw{suffix}.jsonl")
    scored_path = os.path.join(out_dir, f"{output_name}_scored{suffix}.jsonl")

    print("=" * 60)
    print("Qwen3-1.7B-Base MATH500 Evaluation (batched)")
    print(f"  Model:       {args.model}")
    print(f"  Adapter:     {args.adapter or 'none'}")
    print(f"  Samples:     {args.num_samples}")
    print(f"  Max tokens:  {args.max_new_tokens}")
    print(f"  Batch size:  {args.batch_size}")
    if use_shard:
        print(f"  Shard:       {args.shard_id}/{args.num_shards}")
    print(f"  Output:      {scored_path}")
    print("=" * 60)

    # ---- Load dataset ----
    print("\n[1/4] Loading MATH-500 dataset ...")
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    num_samples = min(args.num_samples, len(ds))
    problems = ds.select(range(num_samples))
    if use_shard:
        problems = problems.select(
            range(args.shard_id, len(problems), args.num_shards))
    n_total = len(problems)
    print(f"  {n_total} problems to evaluate")

    # ---- Load tokenizer ----
    print("\n[2/4] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for batch generation

    # ---- Load model ----
    print("[3/4] Loading model (float16) ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
        local_files_only=True)
    model.eval()

    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # ---- Build all prompts ----
    print("\n[4/4] Batched inference ...\n")
    prompts = [PROMPT_PREFIX + p["problem"] + PROMPT_SUFFIX for p in problems]
    gold_answers = [p.get("answer", "") for p in problems]
    subjects = [p.get("subject", "Unknown") for p in problems]
    levels = [str(p.get("level", "?")) for p in problems]
    unique_ids = [p.get("unique_id", "") for p in problems]

    # ---- Inference loop ----
    correct = 0
    extraction_failures = 0
    has_boxed_count = 0  # outputs containing \boxed{}
    total_time = 0.0
    total_output_tokens = 0
    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    level_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    f_raw = open(raw_path, "w", encoding="utf-8")
    f_scored = open(scored_path, "w", encoding="utf-8")

    with torch.inference_mode():
        for batch_start in range(0, n_total, args.batch_size):
            batch_end = min(batch_start + args.batch_size, n_total)
            batch_prompts = prompts[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))
            batch_size = len(batch_indices)

            # Tokenize with left-padding
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                               truncation=True).to(model.device)

            torch.cuda.synchronize()
            t_start = time.time()

            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False, num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            torch.cuda.synchronize()
            elapsed = time.time() - t_start
            total_time += elapsed

            # Decode each sample in the batch
            for j, idx in enumerate(batch_indices):
                pad_len = inputs.input_ids.shape[1]
                # Extract generated tokens (exclude left-padded input, then strip
                # right-padding tokens).  pad_token_id == eos_token_id for Qwen3,
                # so stripping all pad tokens also removes the trailing EOS, which
                # is desirable — we only want the text content.
                gen_ids = outputs[j][pad_len:]
                gen_ids = gen_ids[gen_ids != tokenizer.pad_token_id]
                generated = tokenizer.decode(gen_ids, skip_special_tokens=True)

                real_prompt_tokens = int(inputs.attention_mask[j].sum().item())
                gen_tokens = len(gen_ids)

                pred_answer, is_correct, method = extract_and_match(generated, gold_answers[idx])

                if method == "none":
                    extraction_failures += 1
                if is_correct:
                    correct += 1
                if r'\boxed{' in generated or r'\boxed ' in generated:
                    has_boxed_count += 1

                total_output_tokens += gen_tokens
                subject_stats[subjects[idx]]["total"] += 1
                level_stats[levels[idx]]["total"] += 1
                if is_correct:
                    subject_stats[subjects[idx]]["correct"] += 1
                    level_stats[levels[idx]]["correct"] += 1

                result = {
                    "index": idx, "unique_id": unique_ids[idx],
                    "subject": subjects[idx], "level": levels[idx],
                    "problem": problems[idx]["problem"],
                    "gold_answer": gold_answers[idx],
                    "model_output": generated,
                    "extracted_answer": pred_answer,
                    "extraction_method": method,
                    "exact_match": is_correct,
                    "prompt_tokens": real_prompt_tokens,
                    "output_tokens": gen_tokens,
                    "elapsed_sec": round(elapsed / batch_size, 2),
                }

                line = json.dumps(result, ensure_ascii=False) + "\n"
                f_raw.write(line)
                f_scored.write(line)

            f_raw.flush()
            f_scored.flush()

            # Progress
            n_done = batch_end
            avg_time = total_time / n_done
            eta = avg_time * (n_total - n_done)
            tps_batch = batch_size / elapsed
            acc_sofar = correct / n_done * 100
            print(f"  [{n_done:3d}/{n_total}] "
                  f"batch={elapsed:.1f}s ({tps_batch:.1f} it/s)  "
                  f"avg={avg_time:.1f}s/sample  "
                  f"ETA={eta:.0f}s  "
                  f"acc={acc_sofar:5.1f}%", flush=True)

    f_raw.close()
    f_scored.close()

    # ---- Summary ----
    acc = correct / n_total * 100
    avg_elapsed = total_time / n_total
    avg_tokens = total_output_tokens / n_total

    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total problems:        {n_total}")
    print(f"  Correct (exact match): {correct}")
    print(f"  Accuracy:              {acc:.1f}%")
    print(f"  Boxed rate:            {has_boxed_count}/{n_total} ({has_boxed_count/n_total*100:.1f}%)")
    print(f"  Extraction failures:   {extraction_failures}")
    print(f"  Avg output tokens:     {avg_tokens:.0f}")
    print(f"  Avg time/sample:       {avg_elapsed:.1f}s")
    print(f"  Total inference time:  {total_time:.0f}s")

    print(f"\n  Accuracy by Subject:")
    for subj in sorted(subject_stats.keys()):
        s = subject_stats[subj]
        print(f"    {subj:25s}  {s['correct']:2d}/{s['total']:2d}  "
              f"({s['correct']/s['total']*100:5.1f}%)")

    print(f"\n  Accuracy by Level:")
    for lvl in sorted(level_stats.keys(), key=int):
        s = level_stats[lvl]
        print(f"    Level {lvl}:  {s['correct']:2d}/{s['total']:2d}  "
              f"({s['correct']/s['total']*100:5.1f}%)")

    print(f"\n  Raw outputs:    {raw_path}")
    print(f"  Scored outputs: {scored_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
