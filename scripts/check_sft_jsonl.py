"""SFT JSONL 数据质量检查。

用法:
    python scripts/check_sft_jsonl.py data/processed/sft_stage1_math_12.5k.jsonl
    python scripts/check_sft_jsonl.py data/processed/sft_numinamath_5k.jsonl data/processed/sft_math_train_7.5k.jsonl
"""

import argparse
import json
import os
from collections import Counter


def check(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    n = len(records)
    empty_prompt = sum(1 for r in records if not r.get("prompt", "").strip())
    empty_response = sum(1 for r in records if not r.get("response", "").strip())

    prompts = [r.get("prompt", "") for r in records]
    dup_count = sum(v - 1 for v in Counter(prompts).values() if v > 1)

    resp_lens = [len(r.get("response", "")) for r in records]
    boxed = sum(1 for r in records if "\\boxed" in r.get("response", ""))
    sources = Counter(r.get("source", "?") for r in records)
    subjects = Counter(r.get("subject", "") or "(empty)" for r in records)
    levels = Counter(r.get("level", "") or "(empty)" for r in records)

    print(f"  {os.path.basename(path)}: {n} samples, "
          f"empty_prompt={empty_prompt}, empty_response={empty_response}, "
          f"dup_prompts={dup_count}")
    print(f"  response_len: min={min(resp_lens)}, max={max(resp_lens)}, "
          f"avg={sum(resp_lens)/len(resp_lens):.0f}")
    print(f"  \\boxed: {boxed}/{n} ({boxed/n*100:.1f}%)")
    print(f"  sources: {dict(sources)}")
    print(f"  subjects: {dict(subjects)}")
    print(f"  levels: {dict(levels)}")
    return n


def main():
    parser = argparse.ArgumentParser(description="Check SFT JSONL quality")
    parser.add_argument("files", nargs="+")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    for f in args.files:
        check(f)
        print()

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        json.dump({"checked": args.files}, open(args.output, "w"), indent=2)


if __name__ == "__main__":
    main()
