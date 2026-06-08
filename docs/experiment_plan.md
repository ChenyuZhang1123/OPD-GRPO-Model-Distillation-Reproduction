# OPD/GRPO 数学推理蒸馏复现 — 实验计划

## 项目目标

复现 **Thinking Machines On-Policy Distillation (OPD)** 思路，聚焦于**数学推理能力**的蒸馏。
以 SFT 基线为起点，对比 OPD 蒸馏与 GRPO 强化学习两条路径的效果。

核心问题：能否通过 On-Policy 蒸馏，让学生模型（~8B）在数学推理上逼近教师模型（~32B）的表现，
同时保持比 RL（GRPO）更稳定的训练过程和更高的数据效率。

## 模型候选

| 角色 | 模型 | 理由 |
|---|---|---|
| 学生模型 | **Qwen3-8B-Base** / Qwen2.5-Math-7B | 单张 3090 可容纳 LoRA 微调；8B 规模推理成本低 |
| 教师模型 | **Qwen3-32B** / Qwen2.5-Math-72B | 32B 可在 2-4 张 3090 上运行 vLLM 推理 |

初始选择：学生 Qwen3-8B-Base + 教师 Qwen3-32B（如果 32B logprob 太慢，考虑换成 14B 或 distil-32B）。

## 训练路线

```
阶段 1: SFT Baseline
  ├── 在数学推理数据上对学生模型做监督微调（LoRA）
  ├── 评估 MATH500 / AIME'24
  └── 作为后续对比的基线

阶段 2: OPD 蒸馏
  ├── 教师模型（vLLM）在线生成 logprob / rollout
  ├── 学生模型 on-policy 采样
  ├── 蒸馏目标：min KL(student || teacher) over on-policy trajectories
  └── 对比 SFT baseline 的数学推理提升

阶段 3: GRPO 对比
  ├── 使用 trl.GRPOTrainer 在学生模型上做 RL 微调
  ├── Reward: 数学答案正确性（基于规则 / 教师打分）
  └── 对比 OPD vs GRPO 的效果和训练稳定性
```

## Context Length 计划

| 阶段 | Context Length | 说明 |
|---|---|---|
| 初始 | **4096** | 单卡 3090 友善；大部分 MATH/AIME 题目在此范围 |
| 扩展 | 8192 | 更长推理链及部分长题 |
| 探索 | 16384 / 更长 | 复杂证明类题目；需配合 Flash Attention + ZeRO-2 |

## 评估集建议

| 数据集 | 规模 | 用途 |
|---|---|---|
| **MATH500** | 500 题 | 小规模快速评估，覆盖多难度（Level 1-5） |
| **AIME 2024** | 30 题 | 高难度竞赛题，验证推理上限 |
| AIME 2025 | 30 题 | 扩展评估（确认无数据泄漏后使用） |
| OlympiadBench / Minerva | TBD | 后期大规模评估 |

优先级：先跑通 MATH500 + AIME'24 → 确认 pipeline 正确 → 再扩展评估集。

## 训练数据建议

- 初始数据：从 HuggingFace 下载 open-source 数学推理数据集（如 OpenMathInstruct, MetaMathQA, NuminaMath-CoT）
- 数据预处理：清洗、统一格式（chain-of-thought prompting）、按难度分层采样
- 若教师模型需要：可用教师模型对数据集做 re-label / filtering

## 主要风险及缓解

| 风险 | 严重程度 | 缓解策略 |
|---|---|---|
| **教师 logprob 生成速度** | 高 | vLLM 批量推理；考虑减少教师推理频率（如每 N 步蒸馏一次） |
| **学生训练显存（8B + LoRA）** | 中 | LoRA r=16, alpha=32；gradient checkpointing；bf16；单卡 3090 24GB 可容纳 |
| **长上下文 OOM** | 中 | 初始用 4096，逐步扩展；必要时 Flash Attention 2 |
| **数学答案解析精度** | 中 | 先用规则匹配（答案提取+符号比较），后续可以 LLM-as-judge |
| **Benchmark 区分度** | 低 | MATH500 和 AIME'24 均有较好区分度；8B SFT 预期有提升空间 |
| **OPD 蒸馏效果不明确** | 高 | 严格保存 SFT baseline checkpoint；先小规模 OPD 验证后再全量 |

## 成功标准

- MATH500: SFT baseline → OPD 蒸馏有 +3% 以上绝对提升
- AIME 2024: OPD 蒸馏后 pass@1 比 SFT baseline 有所提升
- 训练稳定性：OPD 蒸馏 loss 平稳收敛，无剧烈震荡
- 与 GRPO 对比：OPD 训练更稳定，最终指标不低于 GRPO

## Zero-shot Baseline 结果 (2026-06-08)

**Qwen3-8B-Base 在 MATH500 上零训练评估：**

| 指标 | 值 |
|---|---|
| Overall exact match | **250/500 = 50.0%** |
| 平均每题耗时 | 10.3s |
| 平均输出长度 | 264 tokens |
| 答案抽取失败 | 1/500 |

### Subject 弱项

| Subject | Accuracy | 错误率 |
|---|---|---|
| Intermediate Algebra | 26.8% | **73.2%** |
| Geometry | 36.6% | **63.4%** |
| Precalculus | 39.3% | **60.7%** |
| Counting & Probability | 44.7% | 55.3% |
| Number Theory | 56.5% | 43.5% |
| Prealgebra | 62.2% | 37.8% |
| Algebra | 67.7% | 32.3% |

### Level 弱项

| Level | Accuracy | 错误率 |
|---|---|---|
| Level 1 | 81.4% | 18.6% |
| Level 2 | 75.6% | 24.4% |
| Level 3 | 61.9% | 38.1% |
| Level 4 | 40.6% | **59.4%** |
| Level 5 | 22.4% | **77.6%** |

### 基线结论

- Base 模型在 Level 1-2 已有较好基础 (75%+)，Level 4-5 是主要提升空间
- **后续训练评估必须关注 subject/level 切片，不能只看 overall accuracy**
- SFT baseline 预期可将 overall 提升至 60-70%；OPD/GRPO 重点攻克 Level 4/5

## 时间线（初步）

| 阶段 | 预计工作量 |
|---|---|
| 环境搭建 + 数据准备 | 1–2 天 |
| SFT Baseline | 1–2 天 |
| OPD 蒸馏实现 + 调试 | 3–5 天 |
| GRPO 对比 + 评估 | 2–3 天 |
| 汇总分析 | 1 天 |
