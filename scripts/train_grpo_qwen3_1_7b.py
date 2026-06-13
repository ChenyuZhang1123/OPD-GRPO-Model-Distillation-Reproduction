"""Qwen3-1.7B-Base LoRA GRPO training on OpenR1-Math-220k.

============================================================================
FORMAL TRAINING — tmux workflow (copy-paste into terminal)
============================================================================

Create tmux session:
    cd ~/OPD
    tmux new -s grpo_vllm

Window 0 — vLLM server:
    conda activate opd-vllm-trl
    CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
      --model /home/zcy/OPD/models/Qwen3-1.7B-Base \
      --host 127.0.0.1 \
      --port 8000

Window 1 — GRPO training (Ctrl+B, C to create new window):
    cd ~/OPD
    conda activate opd-train-vllm
    deepspeed --include localhost:1,2,3,4,5,6,7 \
      scripts/train_grpo_qwen3_1_7b.py \
      --config configs/grpo/qwen3_1.7b_openr1.yaml --yes

tmux keys:
    detach:     Ctrl+B, D
    reattach:   tmux attach -t grpo_vllm
    switch win: Ctrl+B, 0/1
    scroll:     Ctrl+B, [  (then PgUp/PgDn, q to exit)

GPU check (separate terminal):
    watch -n 1 nvidia-smi

VERIFY BEFORE TRAINING:
    GPU 0:  vLLM server ONLY (no training processes)
    GPU 1-7: DeepSpeed training ONLY
    DeepSpeed log MUST show:
      WORLD INFO DICT: {'localhost': [1, 2, 3, 4, 5, 6, 7]}
    If it shows [0, 1, 2, ...], STOP immediately (GPU 0 conflict).

============================================================================

Uses trl.GRPOTrainer with LoRA adapters.  Reward = format_reward (has \boxed{}?) +
correctness_reward (matches reference answer?).

Usage:
    # Dry-run
    python scripts/train_grpo_qwen3_1_7b.py --config configs/grpo/qwen3_1.7b_openr1.yaml --dry-run

    # Single-GPU debug
    CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo_qwen3_1_7b.py \
        --config configs/grpo/qwen3_1.7b_openr1.yaml --yes --max-steps 5

    # === vLLM server mode (1 GPU server + 7 GPUs training) ===
    #
    # IMPORTANT — GPU allocation:
    #   GPU 0:   vLLM server (trl vllm-serve, opd-vllm-trl env)
    #   GPU 1-7: GRPO training (opd-train-vllm env)
    #   Use --include to pin training to GPUs 1-7.
    #   Do NOT use --num_gpus (it ignores CUDA_VISIBLE_DEVICES).
    #   Verify: DeepSpeed log should show WORLD INFO DICT: {'localhost': [1,2,3,4,5,6,7]}
    #
    # Step 0 — Verify no training processes on GPU 0:
    #   nvidia-smi
    #   (Only vLLM server should appear on GPU 0)
    #
    # Step 1 — Start vLLM server (GPU 0, separate terminal):
    #   conda activate opd-vllm-trl
    #   CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
    #     --model /home/zcy/OPD/models/Qwen3-1.7B-Base \
    #     --host 127.0.0.1 --port 8000
    #
    # Step 2 — Speed test (GPUs 1-7, separate terminal):
    #   conda activate opd-train-vllm
    #   deepspeed --include localhost:1,2,3,4,5,6,7 \
    #     scripts/train_grpo_qwen3_1_7b.py \
    #     --config configs/grpo/qwen3_1.7b_openr1.yaml \
    #     --yes --max-steps 5 \
    #     --output-dir outputs/grpo/vllm_server_speed_test
    #
    # Step 3 — Full training (after speed test passes):
    #   deepspeed --include localhost:1,2,3,4,5,6,7 \
    #     scripts/train_grpo_qwen3_1_7b.py \
    #     --config configs/grpo/qwen3_1.7b_openr1.yaml --yes
    #
    # Troubleshooting:
    #   - 127.0.0.1:51216 timeout: collect vLLM server full logs,
    #     trl vllm-serve --help, and training console.log.
    #     May need TRL/vLLM version alignment or port config.

    # === Without vLLM (standard model.generate(), use opd-train) ===
    # Set use_vllm: false in config, then:
    #   conda activate opd-train
    #   deepspeed --include localhost:1,2,3,4,5,6,7 \
    #     scripts/train_grpo_qwen3_1_7b.py \
    #     --config configs/grpo/qwen3_1.7b_openr1.yaml --yes

    # Custom output directory
    deepspeed --include localhost:1,2,3,4,5,6,7 \
      scripts/train_grpo_qwen3_1_7b.py \
      --config configs/grpo/qwen3_1.7b_openr1.yaml --yes \
      --output-dir outputs/grpo/qwen3_1.7b_openr1_v2

    # Single-GPU debug (no vLLM)
    CUDA_VISIBLE_DEVICES=5 python scripts/train_grpo_qwen3_1_7b.py \
        --config configs/grpo/qwen3_1.7b_openr1.yaml --yes \
        --output-dir outputs/grpo/debug_test --max-steps 5

    # Resume from checkpoint
    deepspeed --include localhost:1,2,3,4,5,6,7 \
      scripts/train_grpo_qwen3_1_7b.py \
      --config configs/grpo/qwen3_1.7b_openr1.yaml --yes \
      --resume-from-checkpoint outputs/grpo/qwen3_1.7b_openr1/checkpoint-200
"""

import argparse
import json
import os
import sys
import yaml

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/home/zcy/OPD/models/hf_home"
os.environ["HF_HUB_CACHE"] = "/home/zcy/OPD/models/hub"
os.environ["TRANSFORMERS_CACHE"] = "/home/zcy/OPD/models/transformers"

# Make project src available
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval.answer_extraction import extract_and_match


# ============================================================================
# Reward helpers
# ============================================================================

_reward_debug_done = False


def _get_rank():
    """Return LOCAL_RANK or RANK env var, default 0."""
    for key in ("LOCAL_RANK", "RANK"):
        val = os.environ.get(key)
        if val is not None:
            return int(val)
    return 0


def _completion_to_text(completion):
    """Normalise a completion to str regardless of TRL's internal format.

    TRL may pass completions as str, list[dict] (chat messages), or dict.
    This always returns a plain str.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # List of message dicts — concatenate "content" fields
        parts = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content", "")
                if content:
                    parts.append(str(content))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(completion, dict):
        # Single message dict
        for key in ("content", "text"):
            if key in completion:
                return str(completion[key])
        return str(completion)
    return str(completion)


def _align_answers_to_completions(answers, prompts, completions):
    """Return an answer list whose length matches len(completions).

    Handles:
      A) len(answers) == len(completions) → pass through
      B) len(answers) == len(prompts) and len(completions) is a multiple of
         len(prompts) → repeat each answer num_generations times
      C) Other mismatches → truncate or pad with "", warn
      D) answers is None → return [""] * len(completions)
    """
    n_comp = len(completions)
    if answers is None:
        return [""] * n_comp

    n_ans = len(answers)
    n_prompt = len(prompts) if prompts is not None else 0

    if n_ans == n_comp:
        return list(answers)

    if n_prompt > 0 and n_ans == n_prompt and n_comp % n_prompt == 0:
        repeat = n_comp // n_prompt
        aligned = []
        for a in answers:
            aligned.extend([a] * repeat)
        return aligned

    # Mismatch — warn and fix
    if _get_rank() == 0:
        print(f"  [WARN] reward answer mismatch: "
              f"len(answers)={n_ans}, len(prompts)={n_prompt}, "
              f"len(completions)={n_comp}")
    if n_ans > n_comp:
        return list(answers[:n_comp])
    else:
        return list(answers) + [""] * (n_comp - n_ans)


def _debug_reward_once(prompts, completions, **kwargs):
    """Print reward input shapes once (rank 0 only)."""
    global _reward_debug_done
    if _reward_debug_done or _get_rank() != 0:
        return
    _reward_debug_done = True

    print("=" * 60)
    print("[DEBUG] reward function input shapes (first call, rank 0)")
    print(f"  len(prompts):           {len(prompts) if prompts is not None else 'None'}")
    print(f"  len(completions):       {len(completions)}")
    print(f"  type(completions):      {type(completions).__name__}")
    if len(completions) > 0:
        c0 = completions[0]
        print(f"  type(completions[0]):   {type(c0).__name__}")
        text = _completion_to_text(c0)
        preview = text[:300].replace("\n", "\\n")
        print(f"  completions[0] preview: {preview}")
    print(f"  kwargs keys:            {list(kwargs.keys())}")
    for k, v in kwargs.items():
        vtype = type(v).__name__
        try:
            vlen = len(v)
            print(f"    {k}: {vtype} (len={vlen})")
        except TypeError:
            print(f"    {k}: {vtype} (no len)")
    answers = kwargs.get("answer")
    if answers is not None:
        print(f"  len(answer):            {len(answers)}")
        for i in range(min(3, len(answers))):
            print(f"  answer[{i}]:             {str(answers[i])[:120]}")
    else:
        print(f"  len(answer):            None (missing)")
    print("=" * 60)


# ============================================================================
# Reward functions
# ============================================================================

def format_reward(prompts, completions, completion_ids=None, **kwargs):
    """Reward 1.0 if the completion contains \\boxed{...}, else 0.0."""
    _debug_reward_once(prompts, completions, **kwargs)

    rewards = []
    for completion in completions:
        text = _completion_to_text(completion)
        if r'\boxed{' in text or r'\boxed ' in text:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    # Safety: ensure output length matches input length
    if len(rewards) != len(completions):
        if _get_rank() == 0:
            print(f"  [ERROR] format_reward: len(rewards)={len(rewards)} != "
                  f"len(completions)={len(completions)}. Returning zeros.")
        return [0.0] * len(completions)
    return rewards


def correctness_reward(prompts, completions, completion_ids=None, **kwargs):
    """Reward 1.0 if extracted answer matches reference answer, else 0.0.

    The reference answer is read from kwargs["answer"] and aligned to
    completions via _align_answers_to_completions, which handles both
    per-prompt and per-completion answer lists.
    """
    _debug_reward_once(prompts, completions, **kwargs)

    answers = kwargs.get("answer")
    if answers is None:
        return [0.0] * len(completions)

    # Align answers to completions length
    answers = _align_answers_to_completions(answers, prompts, completions)

    # Now len(answers) == len(completions) — safe to zip
    rewards = []
    for completion, ref_ans in zip(completions, answers):
        text = _completion_to_text(completion)
        _, is_correct, _ = extract_and_match(text, str(ref_ans))
        rewards.append(1.0 if is_correct else 0.0)

    if len(rewards) != len(completions):
        if _get_rank() == 0:
            print(f"  [ERROR] correctness_reward: len(rewards)={len(rewards)} != "
                  f"len(completions)={len(completions)}. Returning zeros.")
        return [0.0] * len(completions)
    return rewards


# ============================================================================
# Helpers
# ============================================================================

def load_jsonl(path: str, limit: int = None) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def dry_run(config: dict):
    """Print config summary and dataset stats without starting training."""
    import torch
    from transformers import AutoTokenizer

    model_cfg = config["model"]
    data_cfg = config["data"]
    lora_cfg = config["lora"]
    grpo_cfg = config["grpo"]
    training_cfg = config["training"]
    output_cfg = config["output"]

    print("=" * 60)
    print("GRPO DRY RUN")
    print(f"  Model:       {model_cfg['name_or_path']}")
    print(f"  Train data:  {data_cfg['train_file']}")
    print(f"  Eval data:   {data_cfg['eval_file']}")
    print(f"  Output dir:  {output_cfg['dir']}")
    print("=" * 60)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"], trust_remote_code=True)

    # ---- Data ----
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = data_cfg["train_file"]
    if not os.path.isabs(train_path):
        train_path = os.path.join(project_root, train_path)

    records = load_jsonl(train_path, limit=training_cfg.get("max_train_samples"))
    print(f"\n  Train records loaded: {len(records)}")
    if records:
        print(f"  Fields: {list(records[0].keys())}")
        print(f"\n  Sample 0 prompt (first 200 chars):")
        print(f"    {records[0]['prompt'][:200]}...")
        print(f"  Sample 0 answer:")
        print(f"    {records[0]['answer']}")

    # ---- Token stats ----
    if records:
        lengths = []
        for r in records:
            full_text = r["prompt"]
            lengths.append(len(tokenizer.encode(full_text)))
        sorted_lens = sorted(lengths)
        pct = lambda arr, p: arr[min(int(len(arr) * p / 100), len(arr) - 1)]
        print(f"\n  Prompt token stats (n={len(records)}): "
              f"min={min(lengths)} p50={pct(sorted_lens, 50)} "
              f"p95={pct(sorted_lens, 95)} max={max(lengths)}")

    # ---- Training budget ----
    per_device = training_cfg["per_device_train_batch_size"]
    grad_acc = training_cfg.get("gradient_accumulation_steps", 1)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_bs = per_device * grad_acc * world_size
    num_gen = grpo_cfg["num_generations"]
    max_steps_val = training_cfg.get("max_steps", -1)

    print(f"\n  GPU count (WORLD_SIZE): {world_size}")
    print(f"  per_device_train_batch_size: {per_device}")
    print(f"  gradient_accumulation_steps: {grad_acc}")
    print(f"  Global batch (distinct prompts/step): {global_bs}")
    print(f"  num_generations per prompt: {num_gen}")
    print(f"  Completions per step: {global_bs * num_gen}")
    print(f"  max_completion_length: {grpo_cfg['max_completion_length']}")
    print(f"  max_prompt_length: {grpo_cfg.get('max_prompt_length', 'N/A')}")
    # ---- vLLM ----
    print(f"  use_vllm: {grpo_cfg.get('use_vllm', False)}")
    if grpo_cfg.get("use_vllm", False):
        vllm_mode = grpo_cfg.get("vllm_mode", "colocate")
        print(f"  vllm_mode: {vllm_mode}")
        if vllm_mode == "server":
            print(f"  vllm_server_host: {grpo_cfg.get('vllm_server_host', '127.0.0.1')}")
            print(f"  vllm_server_port: {grpo_cfg.get('vllm_server_port', 8000)}")
            print(f"  [HINT] Use --include to avoid GPU 0 conflict:")
            print(f"    deepspeed --include localhost:1,2,3,4,5,6,7 \\")
            print(f"      scripts/train_grpo_qwen3_1_7b.py ...")
        print(f"  vllm_gpu_memory_utilization: {grpo_cfg.get('vllm_gpu_memory_utilization', 0.3)}")
        print(f"  vllm_tensor_parallel_size: {grpo_cfg.get('vllm_tensor_parallel_size', 1)}")
    if max_steps_val > 0:
        total_prompts = global_bs * max_steps_val
        print(f"  max_steps: {max_steps_val}")
        print(f"  Total distinct prompts needed: {total_prompts}")
        print(f"  Samples available: {len(records) if records else '?'}")

    print("=" * 60)


class _Tee:
    """Write to a file and the original stdout/stderr simultaneously."""

    def __init__(self, file_path):
        self.file = open(file_path, "w", buffering=1)  # line-buffered
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()


def _build_grpo_config(**kwargs):
    """Build GRPOConfig, filtering out unsupported kwargs with clear messages.

    vLLM-related parameters that GRPOConfig doesn't accept cause a hard error
    with upgrade instructions.  Other unknown parameters produce a warning and
    are silently dropped.
    """
    from dataclasses import fields as dc_fields  # noqa: re-imported for clarity
    from trl import GRPOConfig as _GRPOConfig
    import trl

    valid_fields = {f.name for f in dc_fields(_GRPOConfig)}

    # Parameters that are vLLM-related — if missing, the TRL version is too old.
    _VLLM_PARAMS = {
        "use_vllm", "vllm_mode", "vllm_model_impl", "vllm_enable_sleep_mode",
        "vllm_structured_outputs_regex", "vllm_server_base_url", "vllm_server_host",
        "vllm_server_port", "vllm_server_timeout", "vllm_group_port",
        "vllm_gpu_memory_utilization", "vllm_max_model_length",
        "vllm_tensor_parallel_size", "vllm_importance_sampling_correction",
        "vllm_importance_sampling_mode", "vllm_importance_sampling_cap",
    }

    filtered = {}
    for key, value in kwargs.items():
        if key in valid_fields:
            filtered[key] = value
        elif key in _VLLM_PARAMS:
            print("=" * 60)
            print(f"[ERROR] GRPOConfig in trl=={trl.__version__} does not support "
                  f"'{key}'.")
            print("  This parameter requires a newer TRL version with vLLM support.")
            print("  Upgrade:  pip install trl[vllm] --upgrade")
            print("=" * 60)
            sys.exit(1)
        else:
            print(f"  [WARN] GRPOConfig does not accept parameter '{key}'. Skipping.")

    return _GRPOConfig(**filtered)


def check_env(config: dict):
    """Check environment and GRPOConfig support without loading model or data."""
    import torch
    import transformers
    import trl
    from dataclasses import fields as dc_fields
    from trl import GRPOConfig

    grpo_cfg = config.get("grpo", {})

    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print(f"  Python:        {sys.executable}")
    print(f"  torch:         {torch.__version__}  (CUDA {torch.version.cuda})")
    print(f"  transformers:  {transformers.__version__}")
    print(f"  trl:           {trl.__version__}")

    try:
        import vllm
        print(f"  vllm:          {vllm.__version__}  [installed]")
    except ImportError:
        print(f"  vllm:          NOT INSTALLED")

    print()
    print("GRPOConfig parameter support (trl==" + trl.__version__ + "):")
    valid = {f.name for f in dc_fields(GRPOConfig)}
    check_params = [
        "use_vllm",
        "vllm_mode",
        "vllm_server_host",
        "vllm_server_port",
        "vllm_gpu_memory_utilization",
        "vllm_tensor_parallel_size",
        "max_prompt_length",
    ]
    for p in check_params:
        status = "SUPPORTED" if p in valid else "MISSING"
        print(f"    {p:40s} {status}")

    print()
    print("Config vLLM settings:")
    print(f"  use_vllm:                     {grpo_cfg.get('use_vllm', False)}")
    print(f"  vllm_mode:                    {grpo_cfg.get('vllm_mode', 'colocate')}")
    if grpo_cfg.get('vllm_mode') == 'server':
        print(f"  vllm_server_host:             {grpo_cfg.get('vllm_server_host', '127.0.0.1')}")
        print(f"  vllm_server_port:             {grpo_cfg.get('vllm_server_port', 8000)}")

    if grpo_cfg.get("use_vllm", False):
        try:
            import vllm  # noqa: F401,F811
            print()
            print("  [OK] use_vllm=True and vllm is importable.")
        except ImportError:
            print()
            print("  [WARNING] use_vllm=True but vllm is NOT installed in this env!")
            print("    Create a vLLM-capable training env (does not pollute current env):")
            print("      conda create --name opd-train-vllm --clone opd-train")
            print("      conda activate opd-train-vllm")
            print('      pip install "trl[vllm]"')
            print()
            print("    Verify after install:")
            print("      python -c 'import trl, vllm, torch; print(trl.__version__, vllm.__version__)'")
        if grpo_cfg.get('vllm_mode') == 'server':
            print()
            print("  [HINT] vllm_mode=server: use --include to pin training to GPUs 1-7:")
            print("    deepspeed --include localhost:1,2,3,4,5,6,7 \\")
            print("      scripts/train_grpo_qwen3_1_7b.py ...")
            print("    Do NOT use --num_gpus (it ignores CUDA_VISIBLE_DEVICES).")
            print("    Verify DeepSpeed log shows: {'localhost': [1,2,3,4,5,6,7]}")
    else:
        print()
        print("  use_vllm=False — standard model.generate() will be used.")

    print("=" * 60)


def train(config: dict, max_steps_override: int = None, resume_from_checkpoint: str = None):
    import torch
    import tempfile
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, TaskType
    from trl import GRPOTrainer, GRPOConfig
    import trl

    model_cfg = config["model"]
    data_cfg = config["data"]
    lora_cfg = config["lora"]
    grpo_cfg = config["grpo"]
    training_cfg = config["training"]
    output_cfg = config["output"]
    reward_cfg = config.get("reward", {})
    ds_cfg = config.get("deepspeed", {})

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_ds = bool(ds_cfg and "zero_optimization" in ds_cfg and world_size > 1)

    # ---- dtype ----
    want_bf16 = training_cfg.get("bf16", True)
    bf16_ok = want_bf16 and torch.cuda.is_bf16_supported()
    use_bf16 = bf16_ok
    use_fp16 = not bf16_ok
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # ---- vLLM availability ----
    # (The hard import check happens in main() before train() is called.
    #  Here we just log the mode for user awareness.)
    if grpo_cfg.get("use_vllm", False):
        vllm_mode = grpo_cfg.get("vllm_mode", "colocate")
        if vllm_mode == "server":
            server_host = grpo_cfg.get("vllm_server_host", "127.0.0.1")
            server_port = grpo_cfg.get("vllm_server_port", 8000)
            if local_rank == 0:
                print(f"  [vLLM] server mode — expecting vLLM at {server_host}:{server_port}")
        elif vllm_mode == "colocate":
            if local_rank == 0:
                print(f"  [vLLM] colocate mode — vLLM will be launched in-process")
        else:
            if local_rank == 0:
                print(f"  [WARN] Unknown vllm_mode='{vllm_mode}'. Expected 'colocate' or 'server'.")

    # ---- stdout log (rank 0 only) ----
    log_dir = output_cfg["dir"]
    os.makedirs(log_dir, exist_ok=True)
    tee = None
    if local_rank == 0:
        console_log_path = os.path.join(log_dir, "console.log")
        tee = _Tee(console_log_path)
        sys.stdout = tee
        sys.stderr = tee

    print("=" * 60)
    print(f"LoRA GRPO Training  |  bf16={use_bf16}  fp16={use_fp16}  "
          f"world_size={world_size}  deepspeed={use_ds}")
    print(f"Console log: {os.path.join(log_dir, 'console.log')}")
    print("=" * 60)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # GRPOTrainer expects the tokenizer to have a padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # ---- Model ----
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=None,
    )

    if model_cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # ---- Data ----
    train_path = data_cfg["train_file"]
    if not os.path.isabs(train_path):
        train_path = os.path.join(project_root, train_path)

    max_train = training_cfg.get("max_train_samples")
    records = load_jsonl(train_path, limit=max_train)
    from datasets import Dataset
    train_dataset = Dataset.from_list(records)
    print(f"  Train dataset: {len(train_dataset)} samples")
    print(f"  Columns: {train_dataset.column_names}")

    eval_path = data_cfg["eval_file"]
    if eval_path:
        if not os.path.isabs(eval_path):
            eval_path = os.path.join(project_root, eval_path)
        max_eval = training_cfg.get("max_eval_samples")
        eval_records = load_jsonl(eval_path, limit=max_eval)
        eval_dataset = Dataset.from_list(eval_records)
        print(f"  Eval dataset:  {len(eval_dataset)} samples")
    else:
        eval_dataset = None

    # ---- LoRA ----
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
    )

    # ---- Reward functions ----
    reward_funcs = [format_reward, correctness_reward]
    reward_weights = reward_cfg.get("reward_weights", None)

    # ---- DeepSpeed config ----
    ds_config_path = None
    if use_ds:
        ds_cfg = json.loads(json.dumps(ds_cfg))  # deep copy
        ds_cfg["bf16"] = ds_cfg.get("bf16", {})
        ds_cfg["bf16"]["enabled"] = use_bf16
        ds_cfg["fp16"] = ds_cfg.get("fp16", {})
        ds_cfg["fp16"]["enabled"] = use_fp16
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(ds_cfg, f, indent=2)
            ds_config_path = f.name

    # ---- max_steps override ----
    max_steps = max_steps_override or training_cfg.get("max_steps", -1)

    # ---- GRPOConfig ----
    training_args = _build_grpo_config(
        output_dir=output_cfg["dir"],
        # Generation
        num_generations=grpo_cfg["num_generations"],
        max_completion_length=grpo_cfg["max_completion_length"],
        temperature=grpo_cfg.get("temperature", 1.0),
        top_p=grpo_cfg.get("top_p", 1.0),
        # GRPO-specific
        beta=grpo_cfg.get("beta", 0.04),
        # Training
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
        learning_rate=training_cfg["learning_rate"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        warmup_steps=training_cfg.get("warmup_steps", 0),
        weight_decay=training_cfg["weight_decay"],
        optim=training_cfg["optim"],
        max_grad_norm=training_cfg["max_grad_norm"],
        bf16=use_bf16,
        fp16=use_fp16,
        # Steps / epochs
        max_steps=max_steps if max_steps > 0 else -1,
        num_train_epochs=training_cfg.get("num_train_epochs", 1) if max_steps <= 0 else -1,
        # Logging
        logging_steps=training_cfg["logging_steps"],
        report_to="tensorboard",
        # Saving
        save_strategy=training_cfg.get("save_strategy", "steps"),
        save_steps=training_cfg.get("save_steps", 200),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        # Evaluation
        eval_strategy=training_cfg.get("eval_strategy", "steps") if eval_dataset else "no",
        eval_steps=training_cfg.get("eval_steps", 200) if eval_dataset else None,
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 4),
        num_generations_eval=grpo_cfg.get("num_generations_eval", grpo_cfg["num_generations"]),
        # Misc
        seed=training_cfg["seed"],
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 0),
        deepspeed=ds_config_path,
        remove_unused_columns=False,
        # Reward
        reward_weights=reward_weights,
        scale_rewards=reward_cfg.get("scale_rewards", "group"),
        # Generation batches
        generation_batch_size=grpo_cfg.get("generation_batch_size", 8),
        log_completions=grpo_cfg.get("log_completions", True),
        num_completions_to_print=grpo_cfg.get("num_completions_to_print", 2),
        # vLLM generation backend
        use_vllm=grpo_cfg.get("use_vllm", False),
        vllm_mode=grpo_cfg.get("vllm_mode", "colocate"),
        vllm_server_host=grpo_cfg.get("vllm_server_host", "127.0.0.1"),
        vllm_server_port=grpo_cfg.get("vllm_server_port", 8000),
        vllm_gpu_memory_utilization=grpo_cfg.get("vllm_gpu_memory_utilization", 0.3),
        vllm_tensor_parallel_size=grpo_cfg.get("vllm_tensor_parallel_size", 1),
    )

    # ---- Trainer ----
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # ---- Param count ----
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"  LoRA r={lora_cfg['r']} alpha={lora_cfg['alpha']} "
          f"trainable={trainable/1e6:.2f}M / total={total_params/1e9:.2f}B "
          f"({100*trainable/total_params:.2f}%)")
    print(f"  Rewards: {[f.__name__ for f in reward_funcs]}")
    if reward_weights:
        print(f"  Reward weights: {reward_weights}")

    # ---- DeepSpeed + PyTorch >=2.12 scheduler compatibility ----
    # PyTorch >=2.12 made LRScheduler._update_lr use strict=True in zip(),
    # which crashes when DeepSpeed replaces the optimizer after scheduler
    # creation (param_groups count changes).  Patch to:
    #   1) use strict=False in the zip
    #   2) maintain self._last_lr (required by Trainer._get_learning_rate)
    if use_ds:
        import torch.optim.lr_scheduler as _lrsched_module

        _orig_update_lr = _lrsched_module.LRScheduler._update_lr

        def _patched_update_lr(self, values):
            """Like the original but DeepSpeed-compatible."""
            if not getattr(self, "optimizer", None):
                return
            if values is not None:
                for param_group, lr in zip(self.optimizer.param_groups, values):
                    param_group["lr"] = lr
            # Maintain _last_lr (required by Trainer._get_learning_rate)
            self._last_lr = [
                group.get("lr", 0.0) for group in self.optimizer.param_groups
            ]

        _lrsched_module.LRScheduler._update_lr = _patched_update_lr

        if local_rank == 0:
            print(f"  [DeepSpeed] patched LRScheduler._update_lr: strict=False, "
                  f"_last_lr maintained (PyTorch >=2.12 + DeepSpeed compat)")

    # ---- Diagnostic: optimizer / scheduler alignment (rank 0) ----
    if local_rank == 0 and hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
        opt = trainer.optimizer
        sch = getattr(trainer, 'lr_scheduler', None)
        print(f"  [DIAG] optimizer type:         {type(opt).__name__}")
        print(f"  [DIAG] optimizer param_groups: {len(opt.param_groups)}")
        if sch is not None:
            print(f"  [DIAG] scheduler type:         {type(sch).__name__}")
            base_lrs = getattr(sch, 'base_lrs', [])
            print(f"  [DIAG] scheduler base_lrs:     {len(base_lrs)}")
            last_lr = getattr(sch, '_last_lr', None)
            print(f"  [DIAG] scheduler _last_lr:     {len(last_lr) if last_lr else 'MISSING'}")
            print(f"  [DIAG] sch.optimizer is opt:   {getattr(sch, 'optimizer', None) is opt}")
            # Ensure _last_lr exists (belt-and-suspenders)
            if not hasattr(sch, '_last_lr') or sch._last_lr is None:
                sch._last_lr = [pg.get("lr", 0.0) for pg in opt.param_groups]
                print(f"  [DIAG] initialized _last_lr:   {sch._last_lr}")
        else:
            print(f"  [DIAG] scheduler: None (will be created by Trainer)")
        for i, pg in enumerate(opt.param_groups):
            n_params = sum(p.numel() for p in pg.get('params', []))
            print(f"  [DIAG]   pg[{i}]: lr={pg.get('lr', 'N/A')}, "
                  f"wd={pg.get('weight_decay', 'N/A')}, params={n_params}")

    # ---- Train ----
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # ---- Save final ----
    final_path = os.path.join(output_cfg["dir"], "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"  Model saved to: {final_path}")

    # ---- Save training log (rank 0 only) ----
    log_history = trainer.state.log_history
    if log_history and local_rank == 0:
        import csv

        with open(os.path.join(log_dir, "train_log.json"), "w") as f:
            json.dump(log_history, f, indent=2)

        fieldnames = list(dict.fromkeys(k for row in log_history for k in row))
        with open(os.path.join(log_dir, "train_log.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(log_history)

        print(f"  Logs saved: {log_dir}/train_log.json, train_log.csv")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3-1.7B-Base LoRA GRPO Training")
    parser.add_argument("--config", required=True, help="YAML config file path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config summary and exit")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max_steps from config (for debugging)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory from config. "
                             "All logs, checkpoints, and final model go here.")
    parser.add_argument("--resume-from-checkpoint", default=None,
                        help="Resume training from a checkpoint directory, e.g. "
                             "outputs/grpo/.../checkpoint-200")
    parser.add_argument("--yes", action="store_true",
                        help="Skip confirmation prompt")
    parser.add_argument("--check-env", action="store_true",
                        help="Check environment (python, torch, trl, vllm, GRPOConfig "
                             "param support) and exit without training")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--local-rank", type=int, default=-1, dest="local_rank")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(project_root, args.config)
    config = yaml.safe_load(open(config_path))

    # Override output dir from CLI
    if args.output_dir:
        config["output"]["dir"] = args.output_dir

    if args.check_env:
        check_env(config)
        return

    if args.dry_run:
        dry_run(config)
        return

    # ---- Early vLLM check (fail fast before model loading / DeepSpeed init) ----
    grpo_cfg = config.get("grpo", {})
    if grpo_cfg.get("use_vllm", False):
        try:
            import vllm  # noqa: F401
        except ImportError:
            print("=" * 60)
            print("[ERROR] use_vllm=True but vllm is not importable in this environment.")
            print(f"  Python:  {sys.executable}")
            try:
                import trl as _trl
                print(f"  TRL:     {_trl.__version__}")
            except Exception:
                pass
            print()
            print("  TRL requires vllm to be importable even in vllm_mode='server'.")
            print("  The trainer uses TRL's VLLMClient, which imports vllm internals.")
            print()
            print("  Recommended fix (safe — does NOT pollute current opd-train):")
            print("    conda create --name opd-train-vllm --clone opd-train")
            print("    conda activate opd-train-vllm")
            print('    pip install "trl[vllm]"')
            print()
            print("  Alternative: disable vLLM by setting use_vllm: false in config.")
            print("=" * 60)
            sys.exit(1)

    if not args.yes:
        yn = input("Start GRPO training? (yes/no): ")
        if yn.lower() != "yes":
            sys.exit(0)
    train(config, max_steps_override=args.max_steps,
          resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
