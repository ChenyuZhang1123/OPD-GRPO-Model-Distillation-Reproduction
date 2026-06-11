"""Plot training curves from Trainer log_history (JSON or CSV).

Usage:
    python scripts/plot_train_log.py \
      --log outputs/sft/qwen3_8b_lora_stage1/train_log.json \
      --out outputs/sft/qwen3_8b_lora_stage1/training_curves.png \
      --title "Qwen3-1.7B LoRA SFT Stage1 - 12,455 samples, 1 epoch"

    python scripts/plot_train_log.py \
      --log outputs/sft/qwen3_1.7b_lora_stage1_v1/train_log.json \
      --out outputs/sft/qwen3_1.7b_lora_stage1_v1/training_curves.png \
      --title "Qwen3-1.7B LoRA SFT Stage1 - 12,455 samples, 3 epoch"
"""

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NUMERIC_KEYS = {"step", "loss", "learning_rate", "grad_norm", "mean_token_accuracy", "epoch", "num_tokens"}


def load_logs(path: str) -> list[dict]:
    if path.endswith(".csv"):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for k in NUMERIC_KEYS & r.keys():
                try:
                    r[k] = float(r[k])
                except (ValueError, TypeError):
                    pass
        return rows
    with open(path) as f:
        return json.load(f)


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(x, np.ones(window) / window, mode="valid")


def _smooth_x(xs: np.ndarray, window: int, smoothed_len: int) -> np.ndarray:
    hw = window // 2
    return xs[hw:hw + smoothed_len]


def main():
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument("--log", required=True, help="train_log.json or train_log.csv")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--title", default=None, help="Figure title")
    parser.add_argument("--smooth-window", type=int, default=50, help="Smoothing window size")
    args = parser.parse_args()

    logs = load_logs(args.log)
    # Separate step logs from train summary row (CSV values are strings)
    steps = [e for e in logs if "loss" in e and not e.get("train_runtime")]
    has_acc = any("mean_token_accuracy" in e for e in steps)

    xs = np.array([s["step"] for s in steps])
    loss = np.array([s["loss"] for s in steps])
    lr = np.array([s["learning_rate"] for s in steps])
    gn = np.array([s["grad_norm"] for s in steps])
    acc = np.array([s["mean_token_accuracy"] for s in steps]) if has_acc else None

    w = args.smooth_window
    loss_s = _smooth(loss, w)
    gn_s = _smooth(gn, w)
    acc_s = _smooth(acc, w) if has_acc else None
    xs_s = _smooth_x(xs, w, len(loss_s))

    n_plots = 4 if has_acc else 3
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    if args.title:
        fig.suptitle(args.title, fontsize=14, fontweight="bold")

    # --- Loss ---
    ax = axes[0, 0]
    ax.plot(xs, loss, alpha=0.25, color="steelblue", linewidth=0.5)
    ax.plot(xs_s, loss_s, color="steelblue", linewidth=1.5, label=f"Smoothed (w={w})")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.95, f"Final: {loss[-1]:.4f}\nAvg: {loss.mean():.4f}\nMin: {loss.min():.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # --- LR ---
    ax = axes[0, 1]
    ax.plot(xs, lr * 1e6, color="darkorange", linewidth=1)
    ax.set_xlabel("Step"); ax.set_ylabel("LR (×10⁻⁶)")
    ax.set_title("Learning Rate")
    ax.grid(True, alpha=0.3)

    # --- Gradient Norm ---
    ax = axes[1, 0]
    ax.plot(xs, gn, alpha=0.25, color="seagreen", linewidth=0.5)
    ax.plot(xs_s, gn_s, color="seagreen", linewidth=1.5)
    ax.set_xlabel("Step"); ax.set_ylabel("Grad Norm")
    ax.set_title("Gradient Norm")
    ax.grid(True, alpha=0.3)

    # --- Token Accuracy (or placeholder) ---
    ax = axes[1, 1]
    if has_acc:
        ax.plot(xs, acc * 100, alpha=0.25, color="indianred", linewidth=0.5)
        ax.plot(xs_s, acc_s * 100, color="indianred", linewidth=1.5)
        ax.set_xlabel("Step"); ax.set_ylabel("Token Accuracy (%)")
        ax.set_title("Mean Token Accuracy")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No mean_token_accuracy in log", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_axis_off()

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
