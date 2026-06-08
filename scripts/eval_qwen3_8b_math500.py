"""
Qwen3-8B-Base MATH500 零训练评估。

用法:
    python scripts/eval_qwen3_8b_math500.py --num_samples 50
    python scripts/eval_qwen3_8b_math500.py --num_samples 100 --max_new_tokens 2048
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

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Qwen3-8B-Base MATH500 Evaluation")
parser.add_argument("--num_samples", type=int, default=50,
                    help="Number of MATH500 problems to evaluate (default: 50)")
parser.add_argument("--max_new_tokens", type=int, default=1024,
                    help="Max tokens to generate per problem (default: 1024)")
parser.add_argument("--output_path", type=str,
                    default="outputs/eval/qwen3_8b_math500_50.jsonl",
                    help="Path for raw outputs")
parser.add_argument("--scored_output_path", type=str,
                    default="outputs/eval/qwen3_8b_math500_50_scored.jsonl",
                    help="Path for scored outputs")
args = parser.parse_args()

# Resolve relative paths from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args.output_path = os.path.join(PROJECT_ROOT, args.output_path)
args.scored_output_path = os.path.join(PROJECT_ROOT, args.scored_output_path)

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
MODEL_PATH = "/home/zcy/OPD/models/Qwen3-8B-Base"

print("=" * 60)
print("Qwen3-8B-Base MATH500 Zero-shot Evaluation")
print(f"Model:       {MODEL_PATH}")
print(f"Samples:     {args.num_samples}")
print(f"Max tokens:  {args.max_new_tokens}")
print(f"Output:      {args.output_path}")
print(f"Scored:      {args.scored_output_path}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
print("\n[1/4] Loading MATH-500 dataset ...")
from datasets import load_dataset

ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
num_available = len(ds)
num_samples = min(args.num_samples, num_available)
problems = ds.select(range(num_samples))
print(f"  Loaded {num_samples} / {num_available} problems.")
print(f"  Fields: {list(problems[0].keys())}")
print(f"  Subjects: {sorted(set(problems['subject']))}")
print(f"  Levels:   {sorted(set(problems['level']), key=int)}")

# ---------------------------------------------------------------------------
# 2. Load tokenizer & model
# ---------------------------------------------------------------------------
print("\n[2/4] Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[3/4] Loading model (bf16, device_map=auto) ...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)
print(f"  Loaded in {time.time() - t0:.1f}s")
print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.memory_allocated(i) / 1024**3
    if mem > 0.01:
        print(f"  GPU {i}: {mem:.2f} GB allocated")

# ---------------------------------------------------------------------------
# 4. Inference loop
# ---------------------------------------------------------------------------
print(f"\n[4/4] Evaluating {num_samples} problems ...\n")

# Open output files for incremental writing
f_raw = open(args.output_path, "w")
f_scored = open(args.scored_output_path, "w")

correct = 0
extraction_failures = 0
total_time = 0.0
total_output_tokens = 0
subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})
level_stats   = defaultdict(lambda: {"correct": 0, "total": 0})

for idx in range(num_samples):
    example = problems[idx]
    problem_text = example["problem"]
    gold_answer  = example.get("answer", "")
    subject      = example.get("subject", "Unknown")
    level        = str(example.get("level", "?"))
    unique_id    = example.get("unique_id", "")

    prompt = (
        "Solve the following math problem. Show your reasoning and "
        "put the final answer in \\boxed{}.\n\n"
        f"Problem: {problem_text}\n\n"
        "Solution:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    torch.cuda.synchronize()
    t_start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    elapsed = time.time() - t_start

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = full_output[len(prompt):] if full_output.startswith(prompt) else full_output

    prompt_tokens = inputs.input_ids.shape[1]
    output_tokens = outputs.shape[1] - prompt_tokens

    # Extract answer & match
    pred_answer, is_correct, method = extract_and_match(generated, gold_answer)

    if method == "none":
        extraction_failures += 1
    if is_correct:
        correct += 1

    total_time += elapsed
    total_output_tokens += output_tokens
    subject_stats[subject]["total"] += 1
    level_stats[level]["total"] += 1
    if is_correct:
        subject_stats[subject]["correct"] += 1
        level_stats[level]["correct"] += 1

    gpu_mem = {}
    for i in range(torch.cuda.device_count()):
        a = torch.cuda.memory_allocated(i) / 1024**3
        if a > 0.01:
            gpu_mem[f"gpu_{i}_gb"] = round(a, 2)

    result = {
        "index": idx,
        "unique_id": unique_id,
        "subject": subject,
        "level": level,
        "problem": problem_text,
        "gold_answer": gold_answer,
        "model_output": generated,
        "extracted_answer": pred_answer,
        "extraction_method": method,
        "exact_match": is_correct,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "elapsed_sec": round(elapsed, 2),
        "gpu_memory": gpu_mem,
    }

    # Write incrementally
    line = json.dumps(result, ensure_ascii=False) + "\n"
    f_raw.write(line)
    f_scored.write(line)
    f_raw.flush()
    f_scored.flush()

    # Progress every 5
    if (idx + 1) % 5 == 0 or idx == num_samples - 1:
        acc_sofar = correct / (idx + 1) * 100
        avg_time = total_time / (idx + 1)
        avg_tps = total_output_tokens / total_time if total_time > 0 else 0
        msg = (f"  [{idx + 1:3d}/{num_samples}] "
               f"acc={acc_sofar:5.1f}% "
               f"avg_time={avg_time:.1f}s "
               f"avg_tps={avg_tps:.1f} "
               f"fail={extraction_failures}")
        print(msg, flush=True)

    del inputs, outputs

# Close files
f_raw.close()
f_scored.close()

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
acc = correct / num_samples * 100
avg_output_tokens = total_output_tokens / num_samples
avg_elapsed = total_time / num_samples

print(f"\n{'=' * 60}")
print(f"EVALUATION SUMMARY")
print(f"{'=' * 60}")
print(f"  Total problems:        {num_samples}")
print(f"  Correct (exact match): {correct}")
print(f"  Accuracy:              {acc:.1f}%")
print(f"  Extraction failures:   {extraction_failures}")
print(f"  Avg output tokens:     {avg_output_tokens:.0f}")
print(f"  Avg time/problem:      {avg_elapsed:.1f}s")
print(f"  Total inference time:  {total_time:.0f}s")

print(f"\n  Accuracy by Subject:")
for subj in sorted(subject_stats.keys()):
    s = subject_stats[subj]
    print(f"    {subj:25s}  {s['correct']:2d}/{s['total']:2d}  ({s['correct']/s['total']*100:5.1f}%)")

print(f"\n  Accuracy by Level:")
for lvl in sorted(level_stats.keys(), key=int):
    s = level_stats[lvl]
    print(f"    Level {lvl}:  {s['correct']:2d}/{s['total']:2d}  ({s['correct']/s['total']*100:5.1f}%)")

print(f"\n  Raw outputs:    {args.output_path}")
print(f"  Scored outputs: {args.scored_output_path}")
print(f"{'=' * 60}")
