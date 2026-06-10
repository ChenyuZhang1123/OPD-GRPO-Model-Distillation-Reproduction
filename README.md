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
| Qwen3-8B-Base | `models/Qwen3-8B-Base/` | 16 GB |
| **Qwen3-1.7B-Base (当前学生模型)** | `models/Qwen3-1.7B-Base/` | 3.4 GB |
| MATH-500 数据集 | `data/hf_datasets/` | ~400 KB |
| SFT 训练数据 | `data/processed/sft_stage1_math_12.5k.jsonl` | 12,455 samples |

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

### 关键发现 (8B)

- Qwen3-8B-Base 在 Level 1/2 表现出色 (75%+)，具备基本数学能力
- Level 4/5 急剧下降，缺乏复杂多步推理能力
- Intermediate Algebra 是最弱 subject (26.8%)
- 答案抽取 (`\boxed{}`) 基本可用，500 题仅 1 次失败

### Qwen3-1.7B-Base Zero-shot MATH500 Baseline (2026-06-10)

| 实验 | 正确数 | 准确率 |
|---|---|---|
| 完整 500 题 | 225/500 | **45.0%** |

- 平均每题耗时: 15.4s, 平均输出长度: 529 tokens
- 答案抽取失败: 0/500

### 按 Subject 准确率 (1.7B vs 8B)

| Subject | 1.7B | 8B | 差距 |
|---|---|---|---|
| Algebra | **66.1%** | 67.7% | −1.6 |
| Number Theory | 58.1% | 56.5% | +1.6 |
| Prealgebra | 50.0% | 62.2% | −12.2 |
| Counting & Probability | 39.5% | 44.7% | −5.2 |
| Geometry | 34.1% | 36.6% | −2.5 |
| Intermediate Algebra | 25.8% | 26.8% | −1.0 |
| **Precalculus** | **21.4%** | 39.3% | −17.9 |

### 按 Level 准确率 (1.7B vs 8B)

| Level | 1.7B | 8B | 差距 |
|---|---|---|---|
| 1 (easiest) | 76.7% | 81.4% | −4.7 |
| 2 | 68.9% | 75.6% | −6.7 |
| 3 | 54.3% | 61.9% | −7.6 |
| 4 | 40.6% | 40.6% | 0 |
| **5 (hardest)** | **15.7%** | 22.4% | −6.7 |

### 关键发现 (1.7B)

- 1.7B 整体比 8B 低 5 个百分点 (45.0% vs 50.0%)，差距在预期范围内
- 输出更长 (529 vs 264 tokens)，1.7B 模型更啰嗦，推理效率更低
- Precalculus 下降最严重 (−17.9%)，是 1.7B 的最弱项
- Level 4 反而持平 (40.6%)，Level 5 差距大 (−6.7%)
- 答案抽取 0 失败，比 8B (1 次) 更好

## 当前进度

| 阶段 | 状态 | 说明 |
|---|---|---|
| 环境配置 | ✅ | opd-train / opd-vllm conda envs |
| 模型下载 | ✅ | Qwen3-8B-Base (16 GB) + Qwen3-1.7B-Base (3.4 GB) |
| MATH500 baseline (8B) | ✅ | 250/500 = 50.0% |
| MATH500 baseline (1.7B) | ✅ | 225/500 = 45.0% |
| SFT 数据准备 | ✅ | 12,455 samples, 0 泄漏 |
| SFT 训练 | ⏳ | 待启动 |
| OPD 蒸馏 | ⏳ | 待 SFT 完成 |
| GRPO 对比 | ⏳ | 待 SFT 完成 |

## SFT 训练数据

### 数据组成

| 子集 | 样本数 | avg response | \boxed{} 覆盖 | subject/level |
|---|---|---|---|---|
| NuminaMath-CoT | 5,000 | 1,164 chars | 97.3% | — 无标注 |
| MATH train | 7,500 | 552 chars | 100% | 7 subjects × 5 levels |
| **合并去重后** | **12,455** | **798 chars** | **98.9%** | — |

- 135 条不含 `\boxed{}` 的样本全部来自 NuminaMath olympiads（证明题），保留
- MATH500 泄漏检查: 0 exact / 0 hash / 58 prefix（均为同模板不同参数的题目变体，非泄漏）

### 数据文件

| 文件 | 用途 |
|---|---|
| `data/processed/sft_numinamath_5k.jsonl` | NuminaMath 子集 |
| `data/processed/sft_math_train_7.5k.jsonl` | MATH train 子集 |
| `data/processed/sft_stage1_math_12.5k.jsonl` | 合并去重后的训练数据 |

### 数据构建脚本

```bash
python scripts/build_sft_preview.py --dataset AI-MO/NuminaMath-CoT --max-samples 5000 --output data/processed/sft_numinamath_5k.jsonl
python scripts/build_sft_preview.py --dataset EleutherAI/hendrycks_math --split train --max-samples 20000 --output data/processed/sft_math_train_7.5k.jsonl
python scripts/check_sft_jsonl.py data/processed/sft_numinamath_5k.jsonl data/processed/sft_math_train_7.5k.jsonl
```

### 关键文档

| 文件 | 内容 |
|---|---|
| `docs/experiment_plan.md` | 整体实验计划 |
| `docs/sft_data_source_survey.md` | SFT 数据源调研报告 |
| `docs/sft_lora_stage1_plan.md` | LoRA SFT Stage1 设计文档 |

## Stage1 LoRA SFT — 训练

```bash
conda activate opd-train

# Dry-run (验证数据/tokenizer/配置)
python scripts/train_sft_lora.py --config configs/sft/qwen3_1.7b_lora_stage1.yaml --dry-run

# 单 GPU 训练
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_sft_lora.py \
  --config configs/sft/qwen3_1.7b_lora_stage1.yaml --yes \
  2>&1 | tee logs/train_sft_lora_stage1.log

# 多 GPU DeepSpeed
deepspeed --num_gpus=8 scripts/train_sft_lora.py \
  --config configs/sft/qwen3_1.7b_lora_stage1.yaml --yes \
  2>&1 | tee logs/train_sft_lora_stage1.log
```

### 训练后评估

```bash
# LoRA checkpoint MATH500 评估
python scripts/eval_sft_checkpoint.py \
  --adapter outputs/sft/qwen3_1.7b_lora_stage1/final_model \
  --num-samples 50 --max-new-tokens 1024

# Baseline 评估脚本
python scripts/eval_qwen3_1.7b_math500.py
```

### 配置摘要

| 参数 | 值 |
|---|---|
| 模型 | **Qwen3-1.7B-Base** |
| 数据 | sft_stage1_math_12.5k.jsonl (12,455) |
| LoRA r/alpha/dropout | 16 / 32 / 0.05 |
| Precision | bf16 |
| lr / scheduler / warmup | 5e-5 / cosine / 47 steps |
| Batch size (8 GPU) | 1 × 1 grad_accum = 8 effective |
| Max seq length | 2048 |
| Epochs | 1 |
| Output | `outputs/sft/qwen3_1.7b_lora_stage1/` |

## 项目结构

```
OPD/
├── README.md
├── configs/
│   ├── env.yaml
│   └── sft/
│       └── qwen3_1.7b_lora_stage1.yaml      # 正式训练配置
├── data/
│   ├── raw/ / hf_datasets/
│   └── processed/
│       ├── sft_numinamath_5k.jsonl
│       ├── sft_math_train_7.5k.jsonl
│       └── sft_stage1_math_12.5k.jsonl     # 训练数据 (12,455)
├── docs/
├── logs/
├── models/
│   ├── Qwen3-1.7B-Base/
│   └── Qwen3-8B-Base/
├── outputs/
│   ├── eval/                                # 评估结果
│   ├── sft/
│   │   └── qwen3_1.7b_lora_stage1/         # 正式训练输出
│   └── math500_leakage_check.{json,md}
├── scripts/
│   ├── train_sft_lora.py                   # SFT 训练脚本
│   ├── eval_qwen3_1.7b_math500.py          # Baseline 评估
│   ├── eval_sft_checkpoint.py              # LoRA checkpoint 评估
│   ├── build_sft_preview.py                # SFT 数据构建
│   ├── check_sft_jsonl.py                  # 数据质量检查
│   ├── check_math500_leakage.py            # 泄漏检查
│   └── merge_sft_jsonl.py                  # 数据合并
└── src/
    └── eval/
        └── answer_extraction.py
```
