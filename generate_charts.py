"""
Generate publication-quality charts for TrOCR handwriting recognition benchmarks.
Outputs high-res PNGs to charts/:
  - cer_by_epoch.png
  - loss_by_epoch.png
  - scaling_law.png
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Ensure charts dir exists
out_dir = Path("charts")
out_dir.mkdir(parents=True, exist_ok=True)

# Colors and styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SF Pro Display", "Helvetica Neue", "Arial", "sans-serif"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.titleweight": "bold",
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

# Series data
runs = {
    "Run 3 (334 lines)": {
        "cer": [26.99, 29.46, 34.22, 20.36, 14.88, 18.07, 10.18, 8.67, 6.33, 6.63],
        "loss": [1.677, 1.662, 2.059, 1.269, 1.052, 1.049, 0.816, 0.924, 0.528, 0.544],
        "color": "#f59e0b",
        "ls": "-",
        "marker": "o",
    },
    "Run 4 (507 lines)": {
        "cer": [21.09, 12.49, 17.57, 19.47, 12.69, 8.68, 7.06, 5.19, 5.67, 5.63],
        "loss": [1.254, 1.077, 0.924, 1.175, 0.718, 0.651, 0.538, 0.384, 0.394, 0.394],
        "color": "#3b82f6",
        "ls": "-",
        "marker": "s",
    },
    "Run 5 (593 lines)": {
        "cer": [22.20, 19.42, 10.92, 15.93, 8.91, 10.82, 6.52, 5.71, 5.21, 4.54],
        "loss": [1.461, 1.370, 0.849, 1.046, 0.747, 0.781, 0.611, 0.508, 0.488, 0.431],
        "color": "#10b981",
        "ls": "-",
        "marker": "^",
    },
    "Run 6 (762 lines)": {
        "cer": [19.88, 21.04, 14.82, 8.85, 10.92, 6.83, 5.16, 4.31, 3.82, 3.44],
        "loss": [1.201, 1.488, 0.946, 0.631, 0.631, 0.534, 0.414, 0.382, 0.344, 0.313],
        "color": "#a855f7",
        "ls": "-",
        "marker": "D",
    },
    "Run 7 (1,017 lines)": {
        "cer": [19.10, 26.09, 16.75, 17.52, 10.43, 8.39, 6.81, 5.31, 5.10, 5.27],
        "loss": [1.315, 1.613, 1.059, 1.086, 0.801, 0.664, 0.525, 0.476, 0.474, 0.504],
        "color": "#ec4899",
        "ls": "--",
        "marker": "v",
    },
    "Run 8 (1,277 lines)": {
        "cer": [7.72, 6.63, 6.04, 6.82, 6.87, 6.00, 6.20, 3.58, 2.73, 2.50, 2.72, 2.68],
        "loss": [0.517, 0.473, 0.500, 0.527, 0.440, 0.436, 0.441, 0.319, 0.265, 0.260, 0.255, 0.254],
        "color": "#0ea5e9",
        "ls": "-",
        "marker": "p",
    },
    "Run 9 (2,721 lines)": {
        "cer": [11.86, 10.14, 5.99, 5.42, 4.02, 3.70, 2.93, 2.91, 2.30, 2.27],
        "loss": [0.9365, 0.7486, 0.4651, 0.4414, 0.3396, 0.3290, 0.2688, 0.2718, 0.2536, 0.2400],
        "color": "#6366f1",
        "ls": "-",
        "marker": "*",
        "linewidth": 2.8,
    },
}

# -------------------------------------------------------------
# 1. CER BY EPOCH
# -------------------------------------------------------------
plt.figure(figsize=(10, 6), dpi=300)
for name, meta in runs.items():
    epochs = list(range(1, len(meta["cer"]) + 1))
    lw = meta.get("linewidth", 2.0)
    plt.plot(
        epochs,
        meta["cer"],
        label=name,
        color=meta["color"],
        linestyle=meta["ls"],
        marker=meta["marker"],
        linewidth=lw,
        markersize=6,
        alpha=0.9,
    )

plt.title("TrOCR Fine-Tuning: Validation Character Error Rate (CER) by Epoch", pad=15)
plt.xlabel("Epoch", labelpad=10)
plt.ylabel("Validation CER (%)", labelpad=10)
plt.xticks(range(1, 13))
plt.ylim(0, 36)
plt.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", loc="upper right")
plt.tight_layout()
plt.savefig(out_dir / "cer_by_epoch.png", dpi=300)
plt.close()
print("Saved charts/cer_by_epoch.png")

# -------------------------------------------------------------
# 2. LOSS BY EPOCH
# -------------------------------------------------------------
plt.figure(figsize=(10, 6), dpi=300)
for name, meta in runs.items():
    epochs = list(range(1, len(meta["loss"]) + 1))
    lw = meta.get("linewidth", 2.0)
    plt.plot(
        epochs,
        meta["loss"],
        label=name,
        color=meta["color"],
        linestyle=meta["ls"],
        marker=meta["marker"],
        linewidth=lw,
        markersize=6,
        alpha=0.9,
    )

plt.title("TrOCR Fine-Tuning: Validation Cross-Entropy Loss by Epoch", pad=15)
plt.xlabel("Epoch", labelpad=10)
plt.ylabel("Validation Loss", labelpad=10)
plt.xticks(range(1, 13))
plt.ylim(0.15, 2.2)
plt.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", loc="upper right")
plt.tight_layout()
plt.savefig(out_dir / "loss_by_epoch.png", dpi=300)
plt.close()
print("Saved charts/loss_by_epoch.png")

# -------------------------------------------------------------
# 3. SCALING LAW (CORPUS SIZE VS CER)
# -------------------------------------------------------------
scaling_points = [
    (84, 16.00, "Run 1 (84 lines)"),
    (169, 6.44, "Run 2 (169 lines)"),
    (334, 6.33, "Run 3 (334 lines)"),
    (507, 5.63, "Run 4 (507 lines)"),
    (593, 4.54, "Run 5 (593 lines)"),
    (762, 3.44, "Run 6 (762 lines)"),
    (1017, 5.10, "Run 7 (1,017 lines)*"),
    (1277, 2.50, "Run 8 (1,277 lines)"),
    (2721, 2.27, "Run 9 (2,721 lines)"),
]

sizes = [p[0] for p in scaling_points]
cers = [p[1] for p in scaling_points]

plt.figure(figsize=(10, 6), dpi=300)
plt.plot(sizes, cers, color="#0ea5e9", linewidth=2.5, linestyle="-", marker="o", markersize=7, zorder=3)

# Highlight Run 9
plt.scatter([2721], [2.27], color="#6366f1", s=130, zorder=4, label="Run 9 (2,721 lines: 2.27% CER)")

for size, cer, label in scaling_points:
    y_offset = 0.6 if size != 1017 else 0.7
    plt.annotate(
        f"{cer:.2f}%",
        (size, cer),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontweight="bold",
        fontsize=10,
        color="#1e293b",
    )

plt.title("Scaling Trajectory: Validation CER vs Verified Dataset Size (84 to 2,721 Lines)", pad=15)
plt.xlabel("Verified Training & Validation Lines", labelpad=10)
plt.ylabel("Best Validation CER (%)", labelpad=10)
plt.xlim(0, 2900)
plt.ylim(0, 18)
plt.axhline(2.5, color="#10b981", linestyle=":", alpha=0.7, label="2.5% Character Error Rate Threshold (>97.5% Acc)")
plt.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", loc="upper right")
plt.tight_layout()
plt.savefig(out_dir / "scaling_law.png", dpi=300)
plt.close()
print("Saved charts/scaling_law.png")
