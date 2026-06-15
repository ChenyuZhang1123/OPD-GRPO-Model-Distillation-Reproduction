"""Stratified train/eval split for SFT data.

Splits by (source, subject, level) so eval covers all domains evenly.
Groups with < 5 samples go entirely to train to avoid degenerate eval cells.

Usage:
    python scripts/data/split_train_eval.py \
        --input data/processed/sft_stage1_math_12.5k.jsonl \
        --train-out data/processed/sft_train.jsonl \
        --eval-out data/processed/sft_eval.jsonl \
        --eval-ratio 0.05
"""

import argparse
import json
import os
import sys
from collections import defaultdict


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_stratified(records: list[dict], eval_ratio: float, seed: int = 42):
    """Stratified split by (source, subject, level)."""
    import random
    rng = random.Random(seed)

    # Group records by (source, subject, level)
    groups = defaultdict(list)
    for r in records:
        key = (r.get("source", ""), r.get("subject", ""), r.get("level", ""))
        groups[key].append(r)

    train, eval_ = [], []
    min_group_size = max(2, int(1.0 / eval_ratio))  # groups smaller than this → all train

    for key, group in sorted(groups.items()):
        n_eval = max(0, int(len(group) * eval_ratio))
        if len(group) < 5 or n_eval == 0:
            # Too small to split meaningfully
            train.extend(group)
        else:
            shuffled = list(group)
            rng.shuffle(shuffled)
            eval_.extend(shuffled[:n_eval])
            train.extend(shuffled[n_eval:])

    return train, eval_


def print_summary(train: list[dict], eval_: list[dict]):
    """Print per-group split stats."""
    from collections import Counter

    total = len(train) + len(eval_)
    print(f"Total: {total}  |  Train: {len(train)}  |  Eval: {len(eval_)}  "
          f"({100*len(eval_)/total:.1f}%)")

    # By source
    print("\nBy source:")
    for src in sorted(set(r["source"] for r in train + eval_)):
        t = sum(1 for r in train if r["source"] == src)
        e = sum(1 for r in eval_ if r["source"] == src)
        print(f"  {src:<25s}  train={t:>5d}  eval={e:>4d}  ({100*e/(t+e):.1f}%)")

    # MATH: by subject x level
    math_all = [r for r in train + eval_ if r["source"] == "competition_math"]
    subjects = sorted(set(r.get("subject", "") for r in math_all))
    levels = sorted(set(r.get("level", "") for r in math_all))

    print("\nMATH train subject × level eval distribution:")
    header = f"{'Subject':<25s}"
    for lv in levels:
        header += f"{lv:>10s}"
    print(header)
    for subj in subjects:
        row = f"{subj:<25s}"
        for lv in levels:
            t = sum(1 for r in train if r.get("source") == "competition_math"
                    and r.get("subject") == subj and r.get("level") == lv)
            e = sum(1 for r in eval_ if r.get("source") == "competition_math"
                    and r.get("subject") == subj and r.get("level") == lv)
            if t + e > 0:
                row += f"{e:>4d}/{t+e:<4d} "
            else:
                row += f"{'':>10s}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Stratified train/eval split")
    parser.add_argument("--input", required=True, help="Full training data (.jsonl)")
    parser.add_argument("--train-out", required=True, help="Output train file")
    parser.add_argument("--eval-out", required=True, help="Output eval file")
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    train, eval_ = split_stratified(records, args.eval_ratio, args.seed)
    print_summary(train, eval_)

    save_jsonl(train, args.train_out)
    save_jsonl(eval_, args.eval_out)
    print(f"\nSaved: {args.train_out} ({len(train)} records)")
    print(f"Saved: {args.eval_out} ({len(eval_)} records)")


if __name__ == "__main__":
    main()
