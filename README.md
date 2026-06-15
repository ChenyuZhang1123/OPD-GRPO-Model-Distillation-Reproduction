# OPD/GRPO Model Distillation Reproduction

复现 **Thinking Machines On-Policy Distillation (OPD)**，
在 8× RTX 3090 上对比 **SFT → GRPO → OPD** 三条路径对数学推理能力的提升效果。

学生模型: **Qwen3-1.7B-Base**，教师模型: **Qwen3-32B-Base**。

## 环境 & 资源

| 环境 | 用途 | 关键包 |
|---|---|---|
| `opd-train` | SFT / OPD 训练 | PyTorch 2.12+cu126, trl 1.5.1, deepspeed 0.19.1 |
| `opd-train-vllm` | GRPO 训练 | PyTorch 2.10, vLLM 0.19.0, trl 1.6.0, deepspeed 0.19.1 |
| `opd-vllm-trl` | GRPO vLLM 推理 | vLLM 0.19.0, trl 1.6.0 |

- **GPU**: 8× RTX 3090 (24 GB each), CPU: 64 cores, RAM: 247 GB
- **HF 镜像**: `HF_ENDPOINT=https://hf-mirror.com`
- 详见 `configs/env.yaml`

## SFT — Supervised Fine-Tuning

使用 12,455 条数学推理样本（NuminaMath-CoT + MATH train）对 Qwen3-1.7B-Base 做 LoRA SFT，建立有监督基线。

### 数据

| 子集 | 样本数 | \boxed{} 覆盖 |
|---|---|---|
| NuminaMath-CoT | 5,000 | 97.3% |
| MATH train | 7,500 | 100% |
| **合并去重** | **12,455** | **98.9%** |

MATH500 泄漏检查: 0 exact / 0 hash — 安全。

### 训练

```bash
conda activate opd-train
deepspeed --num_gpus=8 scripts/train_sft_lora.py \
  --config configs/sft/qwen3_1.7b_lora_stage1_v3.yaml --yes \
```

| 参数 | 值 |
|---|---|
| LoRA | r=16, alpha=32, dropout=0.05 |
| lr / scheduler / warmup | 1e-4 / cosine / 50 steps |
| Batch size (8 GPU) | 2 × 1 × 8 = 16 effective |
| Max seq length / epochs | 4096 / 3 |
| Output | `outputs/sft/qwen3_1.7b_lora_stage1_v3/` |

## GRPO — Group Relative Policy Optimization

### 原理

GRPO 源自 DeepSeek 系列工作，核心思想是用**组内相对优势**替代传统的 critic 网络：

1. **DeepSeekMath** (Shao et al., 2024) 提出 GRPO 算法 — 对每个 prompt 采样 N 个 completion，用组内 reward 的均值/标准差做归一化得到 advantage，省去价值函数训练
2. **DeepSeekR1** (Guo et al., 2025) 将 GRPO 用于推理增强 — 通过 rule-based reward（format + correctness）驱动模型习得长链推理能力，无需过程标注

本项目使用 `trl.GRPOTrainer` 实现，reward 由两部分加权组合：
- **format reward** (0.1): 是否包含 `\boxed{}`
- **correctness reward** (1.0): 答案 exact match
- loss_type = `dapo`, scale_rewards = `group`

### 训练

```bash
# 1. 数据准备
python scripts/data/prepare_openr1_math.py --max-train-samples 20000 --max-eval-samples 500

# 2. 启动训练（1 GPU vLLM + 7 GPU DeepSpeed）
# 在 tmux 中分别运行以下两个面板:

# Panel A — vLLM reward model server (GPU 0)
conda activate opd-vllm-trl
CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
  --model /home/zcy/OPD/models/Qwen3-1.7B-Base \
  --host 127.0.0.1 --port 8000

# Panel B — GRPO 训练 (GPU 1-7)
conda activate opd-train-vllm
deepspeed --include localhost:1,2,3,4,5,6,7 \
  src/grpo/grpo.py \
  --config configs/grpo/qwen3_1.7b_openr1_flat.yaml --skip_confirmation \
  --output_dir outputs/grpo/qwen3_1.7b_openr1_v2
```

| 参数 | 值 |
|---|---|
| 数据 | openr1_math_grpo_train.jsonl (20,000) / eval 500 |
| LoRA | r=16, alpha=32, dropout=0.05 |
| lr / scheduler / warmup | 2e-5 / cosine / 50 steps |
| Batch size (8 GPU) | 2 × 4 × 8 = 64 effective |
| num_generations | 8 (train) / 4 (eval) |
| max_completion_length | 2048 |
| temperature / top_p | 1.0 / 1.0 |
| beta | 0.01 |
| loss_type | dapo |
| reward_weights | [0.1 (format), 1.0 (correctness)] |
| vLLM | server mode, gpu_memory_util=0.3 |
| max_steps / DeepSpeed | 1000 / ZeRO-2 |
| Output | `outputs/grpo/qwen3_1.7b_openr1_v2/` |

### MATH500 结果

评测配置: greedy decoding, max_new_tokens=2048, boxed exact match.

| 实验 | 正确数 | 准确率 |
|---|---|---|
| Base (Zero-shot) | 204/500 | **40.8%** |
| **GRPO** | **304/500** | **60.8%** |

GRPO 相对 Base 提升 **+20.0 pp**，输出更简洁（852 → 629 avg tokens），boxed 率 96.0% → 100.0%。

#### 按 Subject

| Subject | Base | GRPO | Δ |
|---|---|---|---|
| Algebra | 62.9% | **81.5%** | +18.6 |
| Counting & Probability | 28.9% | **50.0%** | +21.1 |
| Geometry | 39.0% | **51.2%** | +12.2 |
| Intermediate Algebra | 18.6% | 33.0% | +14.4 |
| Number Theory | 35.5% | **71.0%** | +35.5 |
| Prealgebra | 52.4% | 74.4% | +22.0 |
| Precalculus | 28.6% | 46.4% | +17.8 |

#### 按 Level

| Level | Base | GRPO | Δ |
|---|---|---|---|
| 1 | 76.7% | **86.0%** | +9.3 |
| 2 | 52.2% | **83.3%** | +31.1 |
| 3 | 48.6% | **74.3%** | +25.7 |
| 4 | 40.6% | 58.6% | +18.0 |
| 5 | 15.7% | 29.1% | +13.4 |

**关键发现**: GRPO 在所有维度均有提升，Number Theory 提升最大 (+35.5 pp)，Level 2/3 中等问题改善最显著。Intermediate Algebra 和 Level 5 仍是瓶颈。

## OPD — On-Policy Distillation

### 原理

OPD (Agarwal et al., 2025) 核心思路：用**大教师模型**的逐 token logprob 作为 soft signal 蒸馏小模型，替代 RL 中的 reward 信号。

- Student 生成 rollout → Teacher 逐 token 计算 logprob
- 以 reverse KL (teacher_logprob − student_logprob) 作为 token-level advantage
- Clipped importance sampling loss，类似 PPO 但不依赖 value network
- 相比 GRPO 的优势：稠密 token-level 监督，无需显式 reward 设计

### 训练

```bash
conda activate opd-train
deepspeed --num_gpus=8 src/opd/opd.py \
  --config configs/opd/qwen3_1.7b_opd.yaml --skip_confirmation \
  2>&1 | tee logs/train_opd_qwen3_1.7b.log
```

> 结果将在训练完成后更新。

## 评估

```bash
# 单 checkpoint 评估
CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_qwen3_1.7b_math500.py \
  --adapter outputs/grpo/qwen3_1.7b_openr1_v2/final_model --batch-size 32

# 一键并行评估所有 checkpoint
bash scripts/eval/eval_all_checkpoints.sh outputs/grpo/qwen3_1.7b_openr1_v2
bash scripts/eval/eval_all_checkpoints.sh outputs/grpo/qwen3_1.7b_openr1_v2 --dataset aime25 --num-gpus 4
```

## 项目结构

```
OPD/
├── configs/
│   ├── env.yaml
│   ├── sft/qwen3_1.7b_lora_stage1_v3.yaml
│   ├── grpo/qwen3_1.7b_openr1_flat.yaml
│   └── opd/qwen3_1.7b_opd.yaml
├── data/processed/
│   ├── sft_stage1_math_12.5k.jsonl         # SFT 训练数据
│   ├── openr1_math_grpo_train.jsonl        # GRPO/OPD 训练数据
│   └── openr1_math_grpo_eval.jsonl         
├── models/
│   ├── Qwen3-1.7B-Base/                    # 学生模型
│   └── Qwen3-32B-Base/                     # OPD 教师模型
├── outputs/
│   ├── sft/qwen3_1.7b_lora_stage1_v3/
│   ├── grpo/qwen3_1.7b_openr1_v2/
│   └── opd/qwen3_1.7b_opd/                 # 待运行
├── scripts/
│   ├── train_sft_lora.py
│   ├── eval/
│   │   ├── eval_qwen3_1.7b_math500.py
│   │   ├── eval_qwen3_1.7b_aime25.py
│   │   └── eval_all_checkpoints.sh
│   └── data/
│       ├── build_sft_preview.py
│       ├── prepare_openr1_math.py
│       └── ...
└── src/
    ├── eval/answer_extraction.py
    ├── grpo/                                # GRPO 训练
    └── opd/                                 # OPD 蒸馏训练
```
