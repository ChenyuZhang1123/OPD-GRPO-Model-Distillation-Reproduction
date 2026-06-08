# OPD/GRPO Model Distillation Reproduction

复现 **Thinking Machines On-Policy Distillation (OPD)** 思路，聚焦数学推理能力的蒸馏。
在 8× RTX 3090 上对比 **OPD 蒸馏** 与 **GRPO 强化学习** 两条路径。

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

## 下一步计划

1. 调研 SFT 数学训练数据（NuminaMath-CoT / OpenMathInstruct 等）
2. Qwen3-8B-Base SFT baseline on math reasoning
3. OPD 蒸馏（Qwen3-32B teacher）
4. GRPO 对比训练
5. 关注弱项切片：Intermediate Algebra、Geometry、Level 4/5

## 项目结构

```
OPD/
├── README.md                           # 本文件
├── CLAUDE.md                           # Claude Code 项目记忆
├── configs/env.yaml                    # 环境配置
├── docs/experiment_plan.md            # 实验计划
├── data/                               # 数据集
├── logs/env_check/                     # 环境冒烟测试
├── models/                             # 模型权重
│   └── Qwen3-8B-Base/                  # 学生模型 (16 GB)
├── outputs/eval/                       # 评估输出
├── scripts/                            # 脚本
│   └── eval_qwen3_8b_math500.py       # MATH500 评估脚本
└── src/
    └── eval/
        └── answer_extraction.py        # 答案抽取 + exact match
```

## 状态

✅ 环境配置完成  ✅ 学生模型下载  ✅ MATH500 baseline 完成
⏳ SFT 训练  ⏳ OPD 蒸馏  ⏳ GRPO 对比
