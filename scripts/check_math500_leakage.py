"""MATH500 防泄漏检查 — exact match + hash match + fast prefix match。"""

import argparse
import hashlib
import json
import os
import re
from datetime import datetime

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_DATASETS_CACHE"] = "/home/zcy/OPD/data/hf_datasets"


def _normalize(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _hash(text):
    return hashlib.md5(text.encode()).hexdigest()


def _extract_problem(prompt):
    for marker in ["Problem: ", "Problem:"]:
        if marker in prompt:
            p = prompt[prompt.index(marker) + len(marker):]
            suffix = "\n\nSolution:"
            if suffix in p:
                p = p[:p.index(suffix)]
            return p.strip()
    return prompt


def load_sft(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_math500():
    from datasets import load_dataset
    return [dict(s) for s in load_dataset("HuggingFaceH4/MATH-500", split="test")]


def main():
    parser = argparse.ArgumentParser(description="MATH500 leakage check")
    parser.add_argument("--train", required=True)
    parser.add_argument("--output-json", default="outputs/math500_leakage_check.json")
    parser.add_argument("--output-md", default="outputs/math500_leakage_check.md")
    args = parser.parse_args()

    sft = load_sft(args.train)
    m500 = load_math500()

    sft_problems = []
    for r in sft:
        prob = _extract_problem(r.get("prompt", ""))
        norm = _normalize(prob)
        sft_problems.append({
            "norm": norm, "hash": _hash(norm), "problem": prob,
            "source": r.get("source", ""), "dataset": r.get("dataset", ""),
            "subject": r.get("subject", ""), "raw_id": r.get("raw_id", ""),
        })

    m500_problems = []
    for r in m500:
        norm = _normalize(r.get("problem", ""))
        m500_problems.append({
            "norm": norm, "hash": _hash(norm), "problem": r.get("problem", ""),
            "unique_id": r.get("unique_id", ""),
            "subject": r.get("subject", ""), "level": r.get("level", ""),
        })

    m500_norm_set = {p["norm"] for p in m500_problems}
    m500_hash_set = {p["hash"] for p in m500_problems}

    exact = []
    for sp in sft_problems:
        if sp["norm"] in m500_norm_set:
            mp = next(p for p in m500_problems if p["norm"] == sp["norm"])
            exact.append({"sft": sp["raw_id"], "math500": mp["unique_id"],
                          "sft_problem": sp["problem"][:200],
                          "math500_problem": mp["problem"][:200]})

    hash_only = []
    for sp in sft_problems:
        if sp["hash"] in m500_hash_set and sp["norm"] not in m500_norm_set:
            mp = next(p for p in m500_problems if p["hash"] == sp["hash"])
            hash_only.append({"sft": sp["raw_id"], "math500": mp["unique_id"]})

    # Build prefix index for fast lookup
    prefix_idx = {}
    for mp in m500_problems:
        for plen in [100, 80, 60]:
            key = mp["norm"][:plen]
            prefix_idx.setdefault(key, []).append(mp)

    prefix_matches = []
    seen = set()
    for sp in sft_problems:
        if sp["norm"] in m500_norm_set:
            continue
        for plen in [100, 80, 60]:
            key = sp["norm"][:plen]
            if key in prefix_idx and sp["raw_id"] not in seen:
                for mp in prefix_idx[key]:
                    prefix_matches.append({
                        "sft": sp["raw_id"], "math500": mp["unique_id"],
                        "sft_source": sp["source"], "sft_dataset": sp["dataset"],
                        "math500_subject": mp["subject"], "prefix_len": plen,
                        "sft_problem": sp["problem"][:200],
                        "math500_problem": mp["problem"][:200],
                    })
                    seen.add(sp["raw_id"])
                break

    total = len(exact) + len(hash_only) + len(prefix_matches)

    print(f"SFT: {len(sft)}  MATH500: {len(m500)}")
    print(f"  Exact matches:  {len(exact)}")
    print(f"  Hash matches:   {len(hash_only)}")
    print(f"  Prefix matches: {len(prefix_matches)}")
    print(f"  Total flagged:  {total}")
    print(f"  {'✅ No true leakage' if len(exact) == 0 else '⚠️  LEAKAGE DETECTED'}")

    # JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    json.dump({
        "generated": datetime.now().isoformat(),
        "exact_matches": len(exact), "hash_matches": len(hash_only),
        "prefix_matches": len(prefix_matches), "total": total,
        "exact_items": exact, "prefix_items": prefix_matches[:30],
    }, open(args.output_json, "w"), indent=2, ensure_ascii=False)

    # MD
    lines = [f"# MATH500 Leakage Check\n",
             f"| Check | Count |", f"|---|---|",
             f"| Exact | {len(exact)} |", f"| Hash | {len(hash_only)} |",
             f"| Prefix (≥60 char) | {len(prefix_matches)} |",
             f"| **Total** | **{total}** |\n"]
    if len(exact) == 0:
        lines.append("✅ No exact leakage.")
    else:
        lines.append("⚠️ Exact matches found — remove before training.")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_md)), exist_ok=True)
    open(args.output_md, "w").write("\n".join(lines))
    print(f"  Reports: {args.output_json}, {args.output_md}")


if __name__ == "__main__":
    main()
