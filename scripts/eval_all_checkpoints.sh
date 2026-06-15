#!/bin/bash
# ---------------------------------------------------------------------------
# 一键并行评估所有 SFT checkpoint
#
# 用法:
#   bash scripts/eval_all_checkpoints.sh <checkpoint_dir> [gpu_list]
#
# 示例:
#   # 评估 v3 所有 checkpoint
#   bash scripts/eval_all_checkpoints.sh outputs/sft/qwen3_1.7b_lora_stage1_v3
#   bash scripts/eval_all_checkpoints.sh outputs/grpo/qwen3_1.7b_openr1_v2
#   # 指定 GPU
#   bash scripts/eval_all_checkpoints.sh outputs/sft/qwen3_1.7b_lora_stage1_v3 "0,1,2,3,4,5,6"
# ---------------------------------------------------------------------------

set -euo pipefail

# ---- 参数 ----
CKPT_DIR="${1:?Usage: $0 <checkpoint_dir> [gpu_list]}"
GPU_LIST="${2:-0,1,2,3,4,5,6,7}"

# ---- 路径 ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/eval_qwen3_1.7b_math500.py"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "ERROR: eval script not found: $EVAL_SCRIPT"
    exit 1
fi

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: checkpoint dir not found: $CKPT_DIR"
    exit 1
fi

# ---- 发现所有待评估目标 ----
# 收集 checkpoint-N 目录 + final_model
TARGETS=()
for d in "$CKPT_DIR"/checkpoint-*/; do
    name="$(basename "$d")"
    TARGETS+=("$name:$d")
done
if [ -d "$CKPT_DIR/final_model" ]; then
    TARGETS+=("final_model:$CKPT_DIR/final_model")
fi

N_TARGETS=${#TARGETS[@]}
if [ "$N_TARGETS" -eq 0 ]; then
    echo "ERROR: no checkpoints or final_model found in $CKPT_DIR"
    exit 1
fi

# ---- 解析 GPU 列表 ----
IFS=',' read -ra GPUS <<< "$GPU_LIST"
N_GPUS=${#GPUS[@]}

# ---- 打印信息 ----
echo "============================================================"
echo "  Parallel Eval Launcher"
echo "============================================================"
echo "  Checkpoint dir: $CKPT_DIR"
echo "  Targets found:  $N_TARGETS"
for t in "${TARGETS[@]}"; do
    echo "    - ${t%%:*}"
done
EVAL_OUT_BASE="outputs/eval/$(basename "$CKPT_DIR")"
echo "  GPUs:           ${GPUS[*]} ($N_GPUS available)"
echo "  Output:         $EVAL_OUT_BASE/<checkpoint>/"
echo "============================================================"

# ---- 启动 ----
PIDS=()
GPU_IDX=0

for target in "${TARGETS[@]}"; do
    NAME="${target%%:*}"
    ADAPTER_PATH="${target##*:}"
    GPU="${GPUS[$GPU_IDX]}"
    OUT_DIR="outputs/eval/$(basename "$CKPT_DIR")/$NAME"
    LOG_FILE="$OUT_DIR/eval.log"

    echo ""
    echo "Launching: $NAME -> GPU $GPU"
    echo "  output: $OUT_DIR/"

    mkdir -p "$OUT_DIR"

    CUDA_VISIBLE_DEVICES="$GPU" python "$EVAL_SCRIPT" \
        --adapter "$ADAPTER_PATH" \
        --batch-size 32 \
        --output-dir "$OUT_DIR" \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
    GPU_IDX=$(( (GPU_IDX + 1) % N_GPUS ))
done

echo ""
echo "All $N_TARGETS jobs launched. Waiting..."
echo ""

# ---- 等待全部完成 ----
FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${TARGETS[$i]%%:*}"
    if wait "$pid"; then
        echo "  [OK]    $name (pid $pid)"
    else
        echo "  [FAIL]  $name (pid $pid) — check $EVAL_OUT_BASE/$name/eval.log"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
echo "  Done. $((N_TARGETS - FAILED))/$N_TARGETS succeeded."
if [ "$FAILED" -gt 0 ]; then
    echo "  $FAILED failed. Check logs in $EVAL_OUT_BASE/<checkpoint>/eval.log"
fi
echo "============================================================"
