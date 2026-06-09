"""
从 HF 数学推理数据集构建统一 JSONL 格式的 SFT 数据。

用法:
    python scripts/build_sft_preview.py --dataset AI-MO/NuminaMath-CoT --max-samples 5000 --output data/processed/sft_numinamath_5k.jsonl
    python scripts/build_sft_preview.py --dataset EleutherAI/hendrycks_math --split train --max-samples 20000 --output data/processed/sft_math_train_7.5k.jsonl
"""

import argparse
import hashlib
import json
import os
import random
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/home/zcy/OPD/models/hf_home"
os.environ["HF_HUB_CACHE"] = "/home/zcy/OPD/models/hub"
os.environ["HF_DATASETS_CACHE"] = "/home/zcy/OPD/data/hf_datasets"

from datasets import load_dataset, get_dataset_config_names

_PROMPT_PREFIX = (
    "Solve the following math problem. Show your reasoning step by step, "
    "and put the final answer in \\boxed{}.\n\nProblem: ")
_PROMPT_SUFFIX = "\n\nSolution:"


def build_prompt(problem: str) -> str:
    return _PROMPT_PREFIX + problem + _PROMPT_SUFFIX


def _hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _make_record(problem, solution, source, dataset_name, split_name,
                 raw_id, subject="", level=""):
    return {
        "prompt": build_prompt(problem),
        "response": solution,
        "source": source,
        "raw_id": raw_id,
        "dataset": dataset_name,
        "split": split_name,
        "subject": subject,
        "level": level,
    }


# ---- Dataset adapters ----

def _adapt_numinamath(sample, dataset_name, split_name):
    problem = sample.get("problem") or sample.get("question")
    solution = sample.get("solution") or sample.get("answer")

    if not problem and "messages" in sample:
        for m in sample["messages"]:
            if m.get("role") == "user":
                problem = m.get("content")
            elif m.get("role") == "assistant":
                solution = m.get("content")

    if not problem or not solution:
        return None
    src = sample.get("source", "unknown")
    return _make_record(problem, solution, "numinamath_cot", dataset_name, split_name,
                        f"numinamath_{src}_{_hash(problem[:200])}")


def _adapt_hendrycks_math(sample, dataset_name, split_name):
    problem = sample.get("problem", "")
    solution = sample.get("solution", "")
    if not problem or not solution:
        return None
    subject = sample.get("type", "")
    level = sample.get("level", "")
    return _make_record(problem, solution, "competition_math", dataset_name, split_name,
                        f"math_{subject}_{level}_{_hash(problem[:200])}",
                        subject=subject, level=level)


def _adapt_mathinstruct(sample, dataset_name, split_name):
    instruction = sample.get("instruction", "")
    output = sample.get("output", "")
    if "```python" in output or "```python" in instruction:
        return None
    if not instruction or not output:
        return None
    return _make_record(instruction, output, "mathinstruct", dataset_name, split_name,
                        f"mathinstruct_{_hash(instruction[:200])}")


def _adapt_openmathinstruct(sample, dataset_name, split_name):
    problem = sample.get("problem") or sample.get("question")
    solution = sample.get("generated_solution") or sample.get("solution")
    if not problem or not solution:
        return None
    p_src = sample.get("problem_source", sample.get("dataset", ""))
    return _make_record(problem, solution, "openmathinstruct", dataset_name, split_name,
                        f"omi_{_hash(problem[:200])}", subject=p_src)


ADAPTERS = {
    "AI-MO/NuminaMath-CoT": _adapt_numinamath,
    "EleutherAI/hendrycks_math": _adapt_hendrycks_math,
    "TIGER-Lab/MathInstruct": _adapt_mathinstruct,
    "nvidia/OpenMathInstruct-1": _adapt_openmathinstruct,
    "nvidia/OpenMathInstruct-2": _adapt_openmathinstruct,
}

MULTI_CONFIG_DATASETS = {"EleutherAI/hendrycks_math"}


# ---- Main ----

def build(dataset_name, split="train", config=None, max_samples=100,
          output_path=None, seed=42):
    adapter = ADAPTERS.get(dataset_name)
    if adapter is None:
        print(f"Unknown dataset: {dataset_name}. Available: {list(ADAPTERS.keys())}")
        sys.exit(1)

    try:
        available = get_dataset_config_names(dataset_name)
    except Exception:
        available = []

    if config:
        configs = [config]
    elif dataset_name in MULTI_CONFIG_DATASETS and available:
        configs = available
    elif available:
        configs = [available[0]]
    else:
        configs = [None]

    print(f"Dataset: {dataset_name}  configs={configs}  split={split}  "
          f"max={max_samples}  seed={seed}")

    converted = []
    for cfg in configs:
        if len(converted) >= max_samples:
            break
        cfg_label = cfg or "default"
        try:
            load_kwargs = {"path": dataset_name, "split": split, "streaming": True}
            if cfg:
                load_kwargs["name"] = cfg
            ds = load_dataset(**load_kwargs).shuffle(seed=seed, buffer_size=10000)
        except Exception as e:
            print(f"  config={cfg_label}: ERROR {e}")
            continue

        limit = max_samples if len(configs) == 1 else max_samples // len(configs)
        n = 0
        for sample in ds:
            if n >= limit:
                break
            rec = adapter(dict(sample), dataset_name, split)
            if rec and rec["prompt"].strip() and rec["response"].strip():
                converted.append(rec)
                n += 1
        print(f"  config={cfg_label}: {n} samples")

    if len(configs) > 1 and converted:
        random.Random(seed).shuffle(converted)

    print(f"  total: {len(converted)}")

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in converted:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  wrote: {output_path} ({os.path.getsize(output_path)/1024:.0f} KB)")
    return converted


def main():
    parser = argparse.ArgumentParser(description="Build SFT JSONL from HF math datasets")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.dataset, args.split, args.config, args.max_samples, args.output, args.seed)


if __name__ == "__main__":
    main()
