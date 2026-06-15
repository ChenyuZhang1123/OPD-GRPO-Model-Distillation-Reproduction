#!/bin/bash
# ---------------------------------------------------------------------------
# 一键并行评估所有 checkpoint
#
# 用法:
#   bash scripts/eval/eval_all_checkpoints.sh <checkpoint_dir> [options]
#
# 选项:
#   --dataset math500|aime25     数据集 (default: math500)
#   --num-gpus N                 GPU 数量，从 GPU 0 开始连续分配 (default: 8)
#   --gpus "0,2,4"               显式指定 GPU 列表（覆盖 --num-gpus）
#   --batch-size N               batch size (default: 32)
#   --max-new-tokens N           max new tokens (default: 2048)
#
# 行为:
#   从 checkpoint_dir 下倒序选取 checkpoint：
#     final_model (如有) > checkpoint-{最大步数} > checkpoint-{次大步数} > ...
#   取前 N 个（N = GPU 数量），一卡一个并行评估。
#
# 示例:
#   # MATH500: 8 卡评估最新的 8 个 checkpoint
#   bash scripts/eval/eval_all_checkpoints.sh outputs/grpo/qwen3_1.7b_openr1_v2
#
#   # AIME25: 4 卡
#   bash scripts/eval/eval_all_checkpoints.sh outputs/grpo/qwen3_1.7b_openr1_v2 --dataset aime25 --num-gpus 4
#
#   # 指定 GPU
#   bash scripts/eval/eval_all_checkpoints.sh outputs/sft/qwen3_1.7b_lora_stage1_v3 --dataset math500 --gpus "0,3,5"
# ---------------------------------------------------------------------------

set -euo pipefail

# ---- 默认值 ----
DATASET="math500"
NUM_GPUS=8
GPU_LIST=""
BATCH_SIZE=32
MAX_NEW_TOKENS=2048

# ---- 解析 positional arg ----
CKPT_DIR="${1:?Usage: $0 <checkpoint_dir> [--dataset math500|aime25] [--num-gpus N] [--gpus 0,1,...] [--batch-size N] [--max-new-tokens N]}"
shift

# ---- 解析选项 ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"; shift 2 ;;
        --num-gpus)
            NUM_GPUS="$2"; shift 2 ;;
        --gpus)
            GPU_LIST="$2"; shift 2 ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"; shift 2 ;;
        *)
            echo "ERROR: unknown option: $1"
            exit 1 ;;
    esac
done

# ---- 校验 dataset ----
case "$DATASET" in
    math500) EVAL_SCRIPT_NAME="eval_qwen3_1.7b_math500.py" ;;
    aime25)  EVAL_SCRIPT_NAME="eval_qwen3_1.7b_aime25.py" ;;
    *)
        echo "ERROR: unknown dataset '$DATASET'. Valid: math500, aime25"
        exit 1 ;;
esac

# ---- 路径 ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/$EVAL_SCRIPT_NAME"

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "ERROR: eval script not found: $EVAL_SCRIPT"
    exit 1
fi

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: checkpoint dir not found: $CKPT_DIR"
    exit 1
fi

# ---- 发现所有 checkpoint，倒序排列 ----
# checkpoint-N → 按 N 降序
CHECKPOINTS=()
for d in "$CKPT_DIR"/checkpoint-*/; do
    [ -d "$d" ] || continue
    name="$(basename "$d")"
    # 提取步数，支持 checkpoint-123 或 checkpoint-123-xxx 格式
    step="$(echo "$name" | sed -n 's/^checkpoint-\([0-9]\+\).*/\1/p')"
    CHECKPOINTS+=("$step:$name:$d")
done

# 按 step 降序
IFS=$'\n' CHECKPOINTS_SORTED=($(sort -t: -k1,1nr <<<"${CHECKPOINTS[*]}"))
unset IFS

# 构建有序 target 列表：final_model 在最前（如有），然后 checkpoint 降序
TARGETS=()
if [ -d "$CKPT_DIR/final_model" ]; then
    TARGETS+=("final_model:$CKPT_DIR/final_model")
fi
for entry in "${CHECKPOINTS_SORTED[@]}"; do
    name="${entry#*:}"; name="${name%%:*}"
    path="${entry##*:}"
    TARGETS+=("$name:$path")
done

N_TOTAL=${#TARGETS[@]}
if [ "$N_TOTAL" -eq 0 ]; then
    echo "ERROR: no checkpoints or final_model found in $CKPT_DIR"
    exit 1
fi

# ---- 解析 GPU 列表 ----
if [ -n "$GPU_LIST" ]; then
    IFS=',' read -ra GPUS <<< "$GPU_LIST"
else
    GPUS=()
    for ((i = 0; i < NUM_GPUS; i++)); do
        GPUS+=("$i")
    done
fi
N_GPUS=${#GPUS[@]}

# ---- 按 GPU 数量截断 target 列表（倒序取前 N_GPUS 个） ----
if [ "$N_TOTAL" -gt "$N_GPUS" ]; then
    echo "NOTE: $N_TOTAL checkpoints found, but only $N_GPUS GPU(s) available."
    echo "      Will evaluate the $N_GPUS most recent checkpoint(s):"
    SELECTED=("${TARGETS[@]:0:$N_GPUS}")
    SKIPPED=$((N_TOTAL - N_GPUS))
else
    SELECTED=("${TARGETS[@]}")
    SKIPPED=0
fi

# ---- 打印信息 ----
EVAL_OUT_BASE="outputs/eval/$(basename "$CKPT_DIR")"
echo "============================================================"
echo "  Parallel Eval Launcher"
echo "============================================================"
echo "  Checkpoint dir: $CKPT_DIR"
echo "  Dataset:        $DATASET ($EVAL_SCRIPT_NAME)"
echo "  GPUs:           ${GPUS[*]} ($N_GPUS available)"
echo "  Batch size:     $BATCH_SIZE"
echo "  Max new tokens: $MAX_NEW_TOKENS"
echo "  Total targets:  $N_TOTAL"
if [ "$SKIPPED" -gt 0 ]; then
    echo "  Selected:       ${#SELECTED[@]} (skipped $SKIPPED older)"
fi
echo "  Output base:    $EVAL_OUT_BASE/$DATASET/<checkpoint>/"
echo "------------------------------------------------------------"
echo "  Evaluation order (newest → oldest):"
_i=0
for t in "${SELECTED[@]}"; do
    printf "    %-30s → GPU %s\n" "${t%%:*}" "${GPUS[$_i]}"
    _i=$((_i + 1))
done
echo "============================================================"

# ---- 启动 ----
PIDS=()
GPU_IDX=0

for target in "${SELECTED[@]}"; do
    NAME="${target%%:*}"
    ADAPTER_PATH="${target##*:}"
    GPU="${GPUS[$GPU_IDX]}"
    OUT_DIR="$EVAL_OUT_BASE/$DATASET/$NAME"
    LOG_FILE="$OUT_DIR/eval.log"

    echo ""
    echo "Launching: $NAME → GPU $GPU"
    echo "  output: $LOG_FILE"

    mkdir -p "$OUT_DIR"

    CUDA_VISIBLE_DEVICES="$GPU" python "$EVAL_SCRIPT" \
        --adapter "$ADAPTER_PATH" \
        --batch-size "$BATCH_SIZE" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --output-dir "$OUT_DIR" \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
    GPU_IDX=$((GPU_IDX + 1))
done

echo ""
echo "All ${#SELECTED[@]} jobs launched. Waiting..."
echo ""

# ---- 等待全部完成 ----
FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${SELECTED[$i]%%:*}"
    if wait "$pid"; then
        echo "  [OK]    $name (pid $pid)"
    else
        echo "  [FAIL]  $name (pid $pid) — check $EVAL_OUT_BASE/$DATASET/$name/eval.log"
        FAILED=$((FAILED + 1))
    fi
done

N_DONE=${#SELECTED[@]}
echo ""
echo "============================================================"
echo "  Done. $((N_DONE - FAILED))/$N_DONE succeeded."
if [ "$FAILED" -gt 0 ]; then
    echo "  $FAILED failed. Check logs in $EVAL_OUT_BASE/$DATASET/<checkpoint>/eval.log"
fi
if [ "$SKIPPED" -gt 0 ]; then
    echo "  $SKIPPED older checkpoint(s) skipped (not enough GPUs)."
fi
echo "============================================================"
