# OPD/GRPO Model Distillation Reproduction

复现 **Thinking Machines On-Policy Distillation (OPD)** 思路，
在 8× RTX 3090 上对比 **OPD** 与 **GRPO** 两种方法。

## 技术路线

| 组件 | 方案 |
|---|---|
| 训练框架 | transformers + peft + accelerate + deepspeed |
| 教师推理 | vLLM (teacher logprob / rollout) |
| GRPO baseline | trl.GRPOTrainer |
| 评估 | src/eval/answer_extraction.py (boxed exact match) |

## 服务器环境

- **OS**: Ubuntu 24.04.2 LTS, kernel 6.8.0-124
- **GPU**: 8× NVIDIA GeForce RTX 3090 (24 GB each, 192 GB total)
- **CPU**: 64 cores, RAM: 247 GB
- **Disk**: / partition 967 GB available

| 环境 | 用途 | 关键包 |
|---|---|---|
| `opd-train` | SFT / OPD / GRPO 训练 | PyTorch 2.12+cu126, trl 1.5.1, deepspeed 0.19.1 |
| `opd-vllm` | 教师模型推理 | vLLM 0.22.1 |

详见 `configs/env.yaml`。

## HuggingFace 访问

- `huggingface.co` DNS 不可达，使用 `HF_ENDPOINT=https://hf-mirror.com`
- 推荐缓存配置见 `configs/env.yaml`

## 已下载资源

| 资源 | 路径 | 大小 |
|---|---|---|
| Qwen3-8B-Base (学生模型) | `models/Qwen3-8B-Base/` | 16 GB |
| MATH-500 数据集 | `data/hf_datasets/` | ~400 KB |

## 已完成实验

### Qwen3-8B-Base Zero-shot MATH500 Baseline

| 实验 | 正确数 | 准确率 |
|---|---|---|
| 前 50 题 | 27/50 | **54.0%** |
| 完整 500 题 | 250/500 | **50.0%** |

- 平均每题耗时: 10.3s, 平均输出长度: 264 tokens
- 答案抽取失败: 1/500

### 按 Subject 准确率

| Subject | Accuracy |
|---|---|
| Algebra | 67.7% |
| Prealgebra | 62.2% |
| Number Theory | 56.5% |
| Counting & Probability | 44.7% |
| Precalculus | 39.3% |
| Geometry | 36.6% |
| **Intermediate Algebra** | **26.8%** |

### 按 Level 准确率

| Level | Accuracy |
|---|---|
| 1 (easiest) | 81.4% |
| 2 | 75.6% |
| 3 | 61.9% |
| 4 | 40.6% |
| 5 (hardest) | 22.4% |

### 关键发现

- Qwen3-8B-Base 在 Level 1/2 表现出色 (75%+)，具备基本数学能力
- Level 4/5 急剧下降，缺乏复杂多步推理能力
- Intermediate Algebra 是最弱 subject (26.8%)
- 答案抽取 (`\boxed{}`) 基本可用，500 题仅 1 次失败

## 当前进度

| 阶段 | 状态 | 说明 |
|---|---|---|
| 环境配置 | ✅ | opd-train / opd-vllm conda envs |
| 模型下载 | ✅ | Qwen3-8B-Base (16 GB) |
| MATH500 baseline | ✅ | 250/500 = 50.0% |
| SFT 数据准备 | ✅ | 12,455 samples, 0 泄漏 |
| SFT 训练 | ⏳ **待启动** | 单 GPU ~2h |
| OPD 蒸馏 | ⏳ | 待 SFT 完成 |
| GRPO 对比 | ⏳ | 待 SFT 完成 |

### 关键文档

| 文件 | 内容 |
|---|---|
| `docs/experiment_plan.md` | 整体实验计划 |
| `docs/sft_data_source_survey.md` | SFT 数据源调研报告 |
| `docs/sft_lora_stage1_plan.md` | LoRA SFT Stage1 设计文档 |

## Stage1 LoRA SFT — 启动训练

```bash
# 激活环境
conda activate opd-train

# 启动单 GPU LoRA SFT
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -u scripts/train_sft_lora.py \
    --config configs/sft/qwen3_8b_lora_stage1.yaml \
    --yes \
    2>&1 | tee logs/train_sft_lora_stage1.log

# 多 GPU DeepSpeed
# deepspeed --num_gpus=8 scripts/train_sft_lora.py \
#   --config configs/sft/qwen3_8b_lora_stage1.yaml --yes \
#   2>&1 | tee logs/train_sft_lora_stage1.log
```

### 训练后评估

```bash
# MATH500 first 50
python -u scripts/eval_sft_checkpoint.py \
  --adapter outputs/sft/qwen3_8b_lora_stage1/final_model \
  --num-samples 50 --max-new-tokens 1024 \
  2>&1 | tee logs/eval_sft_stage1_first50.log

# MATH500 full 500
python -u scripts/eval_qwen3_8b_math500.py \
  --num_samples 500 --max_new_tokens 1024 \
  --output_path outputs/eval/sft_stage1_math500.jsonl \
  2>&1 | tee logs/eval_sft_stage1_full500.log
```

### 配置摘要

| 参数 | 值 |
|---|---|
| 模型 | Qwen3-8B-Base |
| 数据 | sft_stage1_math_12.5k.jsonl (12,455) |
| LoRA r/alpha/dropout | 16 / 32 / 0.05 |
| Precision | bf16 |
| lr / scheduler / warmup | 5e-5 / cosine / 12 steps |
| Batch size (effective) | 1 × 4 grad_accum = 4 |
| Max seq length | 2048 |
| Steps/epoch | ~3,114 |
| Epochs | 1 |
| Output | `outputs/sft/qwen3_8b_lora_stage1/` |

## 项目结构

```
OPD/
├── README.md
├── CLAUDE.md
├── configs/
│   ├── env.yaml
│   └── sft/
│       ├── qwen3_8b_lora_stage1.yaml      # 正式训练配置
│       └── qwen3_8b_lora_stage1_smoke.yaml # Smoke test 配置
├── docs/
│   ├── experiment_plan.md
│   ├── sft_data_source_survey.md           # 数据源调研
│   └── sft_lora_stage1_plan.md             # Stage1 设计
├── data/
│   ├── raw/ / hf_datasets/
│   └── processed/
│       ├── sft_numinamath_5k.jsonl
│       ├── sft_math_train_7.5k.jsonl
│       └── sft_stage1_math_12.5k.jsonl     # 训练数据 (12,455)
├── models/
│   └── Qwen3-8B-Base/
├── outputs/
│   ├── eval/                                # 评估结果
│   ├── sft/
│   │   ├── qwen3_8b_lora_stage1_smoke/     # Smoke checkpoint
│   │   └── qwen3_8b_lora_stage1/           # 正式训练输出
│   ├── math500_leakage_check.{json,md}
│   └── sft_data_quality_summary.json
├── scripts/
│   ├── train_sft_lora.py                   # SFT 训练脚本
│   ├── eval_qwen3_8b_math500.py            # Baseline 评估
│   ├── eval_sft_checkpoint.py              # LoRA checkpoint 评估
│   ├── build_sft_preview.py                # SFT 数据构建
│   ├── check_sft_jsonl.py                  # 数据质量检查
│   ├── check_math500_leakage.py            # 泄漏检查
│   ├── merge_sft_jsonl.py                  # 数据合并
│   └── inspect_sft_samples.py              # 样本抽查
├── logs/
│   ├── train_sft_lora_smoke.log            # Smoke 训练日志
│   ├── eval_sft_checkpoint_first50.log     # Checkpoint 评估日志
│   └── check_math500_leakage.log           # 泄漏检查日志
└── src/
    └── eval/
        └── answer_extraction.py
```
