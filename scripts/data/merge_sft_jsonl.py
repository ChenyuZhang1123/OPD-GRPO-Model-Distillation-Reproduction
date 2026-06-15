"""合并多个 SFT JSONL 文件，prompt 精确去重后打乱输出。"""

import argparse
import json
import os
import random


def main():
    parser = argparse.ArgumentParser(description="Merge and dedup SFT JSONL files")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = []
    for fpath in args.inputs:
        with open(fpath, encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
        records.extend(recs)
        print(f"  {os.path.basename(fpath)}: {len(recs)}")

    before = len(records)
    seen = set()
    deduped = []
    for r in records:
        p = r.get("prompt", "")
        if p not in seen:
            seen.add(p)
            deduped.append(r)

    random.Random(args.seed).shuffle(deduped)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  before={before}  removed={before - len(deduped)}  after={len(deduped)}")
    print(f"  wrote: {args.output}")


if __name__ == "__main__":
    main()
