"""
Qwen3-1.7B-Base AIME25 批量推理评测。
支持 base model 直接推理，也支持加载 LoRA adapter 评测 SFT/OPD/GRPO checkpoint。

Prompt 格式与 MATH500 评测一致（与 GRPO train prompt 相同）:
  Question:
  {problem}

  Please solve the problem step by step and put the final answer in \\boxed{}.

  Answer:

AIME25 数据集: math-ai/aime25 (30 题，答案均为整数)
答案评测逻辑复用 src/eval/answer_extraction.py（boxed 提取 + math-verify + string match）

用法
  # 1. 评估 base model
  CUDA_VISIBLE_DEVICES=5 python scripts/eval/eval_qwen3_1.7b_aime25.py \
    --batch-size 8 --max-new-tokens 2048 \
    --output-dir outputs/eval/aime25/base_model

  # 2. 评估 LoRA checkpoint
  CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_qwen3_1.7b_aime25.py \
    --adapter outputs/sft/qwen3_1.7b_lora_stage1_v3/final_model \
    --batch-size 8 --max-new-tokens 2048

  # 3. 评估 GRPO checkpoint
  CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_qwen3_1.7b_aime25.py \
    --adapter outputs/grpo/qwen3_1.7b_openr1_v2/final_model \
    --batch-size 8 --max-new-tokens 2048

  # 4. 评估 OPD checkpoint
  CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_qwen3_1.7b_aime25.py \
    --adapter outputs/opd/qwen3_1.7b_opd/final_model \
    --batch-size 8 --max-new-tokens 2048

  # 5. 评估中间 checkpoint
  CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_qwen3_1.7b_aime25.py \
    --adapter outputs/grpo/qwen3_1.7b_openr1_v2/checkpoint-900 \
    --batch-size 8 --max-new-tokens 2048
"""

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.eval.answer_extraction import extract_and_match

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/home/zcy/OPD/models/hf_home"
os.environ["HF_HUB_CACHE"] = "/home/zcy/OPD/models/hub"
os.environ["TRANSFORMERS_CACHE"] = "/home/zcy/OPD/models/transformers"
os.environ["HF_DATASETS_CACHE"] = "/home/zcy/OPD/data/hf_datasets"
MODEL_PATH = "/home/zcy/OPD/models/Qwen3-1.7B-Base"
# GRPO train prompt format (must match data/processed/openr1_math_grpo_train.jsonl)
PROMPT_PREFIX = "Question:\n"
PROMPT_SUFFIX = "\n\nPlease solve the problem step by step and put the final answer in \\boxed{}.\n\nAnswer:"

# AIME25 answer validation: all answers should be integers
# AIME answer range: 000-999 (3-digit integer)
_AIME_ANSWER_RE = None  # cached compiled regex


def is_valid_aime_answer(answer_str: str) -> bool:
    """Check if extracted answer looks like a valid AIME answer (integer 0-999)."""
    global _AIME_ANSWER_RE
    if _AIME_ANSWER_RE is None:
        import re
        _AIME_ANSWER_RE = re.compile(r'^\s*\d{1,3}\s*$')
    return bool(_AIME_ANSWER_RE.match(answer_str))


def main():
    parser = argparse.ArgumentParser(description="Qwen3-1.7B-Base AIME25 Evaluation")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--adapter", default=None, help="LoRA adapter path (optional)")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--output-name", default=None,
                        help="Custom output filename prefix. Default: auto-detect "
                             "from --adapter or use 'qwen3_1.7b_aime25' for base model.")
    args = parser.parse_args()

    use_shard = args.shard_id is not None and args.num_shards is not None

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ---- output paths ----
    if args.output_name:
        output_name = args.output_name
    elif args.adapter:
        adapter_tag = args.adapter.rstrip('/').split('/')[-1]
        output_name = f"qwen3_1.7b_adapter_{adapter_tag}_aime25"
    else:
        output_name = "qwen3_1.7b_aime25"

    suffix = f"_s{args.shard_id}of{args.num_shards}" if use_shard else ""
    raw_path = os.path.join(out_dir, f"{output_name}_raw{suffix}.jsonl")
    scored_path = os.path.join(out_dir, f"{output_name}_scored{suffix}.jsonl")

    print("=" * 60)
    print("Qwen3-1.7B-Base AIME25 Evaluation (batched)")
    print(f"  Model:       {args.model}")
    print(f"  Adapter:     {args.adapter or 'none'}")
    print(f"  Max tokens:  {args.max_new_tokens}")
    print(f"  Batch size:  {args.batch_size}")
    if use_shard:
        print(f"  Shard:       {args.shard_id}/{args.num_shards}")
    print(f"  Output:      {scored_path}")
    print("=" * 60)

    # ---- Load dataset ----
    print("\n[1/4] Loading AIME25 dataset ...")
    from datasets import load_dataset
    ds = load_dataset("math-ai/aime25", split="test")
    problems = list(ds)
    if use_shard:
        problems = [problems[i] for i in range(args.shard_id, len(problems), args.num_shards)]
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
    gold_answers = [str(p["answer"]) for p in problems]
    problem_ids = [p.get("id", i) for i, p in enumerate(problems)]

    # ---- Inference loop ----
    correct = 0
    extraction_failures = 0
    has_boxed_count = 0  # outputs containing \boxed{}
    invalid_answer_count = 0  # extracted answers not matching AIME format (000-999)
    total_time = 0.0
    total_output_tokens = 0
    all_results = []  # collect for per-problem summary table

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
                # AIME answers should be integers (0-999); check format of extracted answer
                if method != "none" and not is_valid_aime_answer(pred_answer):
                    invalid_answer_count += 1

                total_output_tokens += gen_tokens

                result = {
                    "id": problem_ids[idx],
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

                all_results.append(result)

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
            print(f"  [{n_done:2d}/{n_total}] "
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
    print(f"  Invalid AIME format:   {invalid_answer_count} (extracted answer not 0-999 integer)")
    print(f"  Avg output tokens:     {avg_tokens:.0f}")
    print(f"  Avg time/sample:       {avg_elapsed:.1f}s")
    print(f"  Total inference time:  {total_time:.0f}s")

    # Per-problem summary table
    print(f"\n  Per-problem results:")
    print(f"  {'ID':>3s}  {'Gold':>6s}  {'Pred':>10s}  {'Correct':>7s}  {'Tok':>5s}  Method")
    for r in all_results:
        pred_display = r["extracted_answer"] if r["extraction_method"] != "none" else "(none)"
        if len(pred_display) > 10:
            pred_display = pred_display[:9] + "…"
        print(f"  {r['id']:3d}  {r['gold_answer']:>6s}  {pred_display:>10s}  "
              f"{'✓' if r['exact_match'] else '✗':>7s}  {r['output_tokens']:5d}  {r['extraction_method']}")

    print(f"\n  Raw outputs:    {raw_path}")
    print(f"  Scored outputs: {scored_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
