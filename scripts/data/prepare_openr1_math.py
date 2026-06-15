"""
Prepare OpenR1-Math-220k for GRPO training.

Downloads (or loads cached) open-r1/OpenR1-Math-220k, normalises the schema, and
writes train/eval JSONL files ready for GRPOTrainer.

Usage:
    # Full download and process (default: 20k train / 500 eval)
    python scripts/data/prepare_openr1_math.py

    # Custom sizes
    python scripts/data/prepare_openr1_math.py --max-train-samples 8000 --max-eval-samples 300

    # Offline
    python scripts/data/prepare_openr1_math.py --local-files-only

Output:
    data/processed/openr1_math_grpo_train.jsonl
    data/processed/openr1_math_grpo_eval.jsonl

Each line has: prompt, answer, source, original_id (if available)
"""

import argparse
import json
import os
import random
import sys
from typing import Optional

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/home/zcy/OPD/models/hf_home"
os.environ["HF_HUB_CACHE"] = "/home/zcy/OPD/models/hub"
os.environ["HF_DATASETS_CACHE"] = "/home/zcy/OPD/data/hf_datasets"


# ---- Prompt template ----
PROMPT_TEMPLATE = """Question:
{problem}

Please solve the problem step by step and put the final answer in \\boxed{{}}.

Answer:"""


# ---- Schema mapping ----

# Candidate column names for problem and answer fields (tried in order)
PROBLEM_CANDIDATES = ["problem", "question", "instruction", "prompt", "input", "query"]
ANSWER_CANDIDATES = ["answer", "solution", "response", "output", "completion", "target", "ground_truth"]
SOURCE_CANDIDATES = ["source", "dataset", "domain"]
ID_CANDIDATES = ["original_id", "id", "unique_id", "problem_id", "uuid"]


def _first_match(record: dict, candidates: list[str]) -> Optional[str]:
    """Return the first non-empty value for any key in *candidates*."""
    for key in candidates:
        val = record.get(key)
        if val is not None and str(val).strip():
            if isinstance(val, list):
                # Unwrap single-element lists
                val = val[0] if len(val) == 1 else val
            return str(val)
    return None


def _normalise_record(record: dict) -> dict:
    """Extract prompt, answer, source, original_id from a raw record.

    Raises ValueError if problem or answer cannot be found.
    """
    problem = _first_match(record, PROBLEM_CANDIDATES)
    answer = _first_match(record, ANSWER_CANDIDATES)

    if problem is None:
        raise ValueError(
            f"Cannot find problem field. Available keys: {list(record.keys())}. "
            f"Tried: {PROBLEM_CANDIDATES}"
        )
    if answer is None:
        raise ValueError(
            f"Cannot find answer field. Available keys: {list(record.keys())}. "
            f"Tried: {ANSWER_CANDIDATES}"
        )

    source = _first_match(record, SOURCE_CANDIDATES) or "openr1"
    original_id = _first_match(record, ID_CANDIDATES)

    prompt = PROMPT_TEMPLATE.format(problem=problem.strip())

    out = {"prompt": prompt, "answer": answer.strip(), "source": source}
    if original_id is not None:
        out["original_id"] = original_id
    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare OpenR1-Math-220k for GRPO")
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-eval-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--config", default="default",
                        help="Dataset config: 'default' (220k), 'all', or 'extended'")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "openr1_math_grpo_train.jsonl")
    eval_path = os.path.join(out_dir, "openr1_math_grpo_eval.jsonl")

    print("=" * 60)
    print("OpenR1-Math-220k GRPO Data Preparation")
    print(f"  Config:            {args.config}")
    print(f"  Max train:         {args.max_train_samples}")
    print(f"  Max eval:          {args.max_eval_samples}")
    print(f"  Seed:              {args.seed}")
    print(f"  Local files only:  {args.local_files_only}")
    print(f"  Output dir:        {out_dir}")
    print("=" * 60)

    # ---- Load ----
    print("\n[1/4] Loading dataset ...")
    from datasets import load_dataset

    ds = load_dataset(
        "open-r1/OpenR1-Math-220k",
        args.config,
        split="train",
    )

    print(f"  Split:  train")
    print(f"  Fields: {ds.column_names}")
    print(f"  Rows:   {len(ds)}")

    # ---- Print first 2 samples ----
    print("\n[2/4] First 2 raw samples:")
    for i in range(min(2, len(ds))):
        print(f"\n  --- Sample {i + 1} ---")
        for k, v in ds[i].items():
            s = str(v)
            if len(s) > 200:
                s = s[:200] + "..."
            print(f"  {k}: {s}")

    # ---- Normalise ----
    print("\n[3/4] Normalising records ...")
    all_records = []
    failed = 0
    for row in ds:
        try:
            all_records.append(_normalise_record(row))
        except ValueError as e:
            failed += 1
            if failed <= 5:
                print(f"  WARNING: {e}")

    print(f"  Normalised: {len(all_records)}")
    if failed:
        print(f"  Failed:     {failed}")

    # ---- Shuffle and split ----
    rng = random.Random(args.seed)
    rng.shuffle(all_records)

    n_train = min(args.max_train_samples, len(all_records))
    n_eval = min(args.max_eval_samples, max(0, len(all_records) - n_train))

    train_records = all_records[:n_train]
    eval_records = all_records[n_train:n_train + n_eval]

    # ---- Write ----
    print("\n[4/4] Writing output ...")
    for path, records in [(train_path, train_records), (eval_path, eval_records)]:
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {path}: {len(records)} records")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"  Train: {train_path} ({len(train_records)} samples)")
    print(f"  Eval:  {eval_path} ({len(eval_records)} samples)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
