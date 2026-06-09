"""评估 LoRA SFT checkpoint on MATH500。"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval.answer_extraction import extract_and_match

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_PATH = "/home/zcy/OPD/models/Qwen3-8B-Base"


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA SFT checkpoint")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--output", default="outputs/eval/sft_eval_scored.jsonl")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args.output = os.path.join(project_root, args.output)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Adapter: {args.adapter}  Samples: {args.num_samples}")

    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = ds.select(range(args.num_samples))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    correct = 0
    failures = 0
    total_time = 0.0
    total_tokens = 0
    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    level_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    results = []

    for idx in range(len(problems)):
        ex = problems[idx]
        problem_text = ex["problem"]
        gold = ex.get("answer", "")
        subject = ex.get("subject", "Unknown")
        level = str(ex.get("level", "?"))

        prompt = (
            "Solve the following math problem. Show your reasoning and "
            "put the final answer in \\boxed{}.\n\n"
            f"Problem: {problem_text}\n\nSolution:")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        p_tokens = inputs.input_ids.shape[1]

        torch.cuda.synchronize()
        t_start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, temperature=1.0,
                                     pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        elapsed = time.time() - t_start

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if generated.startswith(prompt):
            generated = generated[len(prompt):]

        pred, is_correct, method = extract_and_match(generated, gold)
        if method == "none":
            failures += 1
        if is_correct:
            correct += 1

        total_time += elapsed
        total_tokens += outputs.shape[1] - p_tokens
        subject_stats[subject]["total"] += 1
        level_stats[level]["total"] += 1
        if is_correct:
            subject_stats[subject]["correct"] += 1
            level_stats[level]["correct"] += 1

        results.append({
            "index": idx, "subject": subject, "level": level,
            "problem": problem_text[:200], "model_output": generated[:500],
            "extracted_answer": pred, "exact_match": is_correct,
            "elapsed_sec": round(elapsed, 1),
            "output_tokens": outputs.shape[1] - p_tokens,
        })

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1:3d}/{args.num_samples}] acc={correct/(idx+1)*100:.1f}%")
        del inputs, outputs

    acc = correct / args.num_samples * 100
    print(f"\n  Correct: {correct}/{args.num_samples} ({acc:.1f}%)")
    print(f"  Failures: {failures}  Avg time: {total_time/args.num_samples:.1f}s  "
          f"Avg tokens: {total_tokens/args.num_samples:.0f}")
    print(f"  By Subject:")
    for subj in sorted(subject_stats):
        s = subject_stats[subj]
        if s["total"] > 0:
            print(f"    {subj:25s} {s['correct']:2d}/{s['total']:2d} "
                  f"({s['correct']/s['total']*100:.1f}%)")
    print(f"  By Level:")
    for lvl in sorted(level_stats, key=lambda x: int(x) if x.isdigit() else 999):
        s = level_stats[lvl]
        if s["total"] > 0:
            print(f"    Level {lvl}: {s['correct']:2d}/{s['total']:2d} "
                  f"({s['correct']/s['total']*100:.1f}%)")

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
