import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

SAVED = Path("/home/jupyter-st126222/Project/rdd2022/saved")
PLOTS = Path("/home/jupyter-st126222/Project/rdd2022/plots")
PLOTS.mkdir(exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        12,
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "axes.labelsize":   12,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
})

COLORS  = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
CLASSES = ["D00", "D10", "D20", "D40"]
FRACS   = [10, 25, 50, 100]

per_class = {
    10:  [0.3407, 0.2683, 0.4562, 0.5021],
    25:  [0.4089, 0.3346, 0.5283, 0.6263],
    50:  [0.4652, 0.4073, 0.6108, 0.6941],
    100: [0.5121, 0.4833, 0.6355, 0.7416],
}
imbalance = {
    "Default":  [0.5121, 0.4833, 0.6355, 0.7416],
    "Weighted": [0.5162, 0.4847, 0.6448, 0.7536],
}

# ── Plot 1: Learning Curve ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
map50s = []
for f in FRACS:
    name = f"fraction_{f}pct"
    df = pd.read_csv(SAVED / name / "results.csv")
    df.columns = df.columns.str.strip()
    map50s.append(df["metrics/mAP50(B)"].max())

ax.plot(FRACS, map50s, marker="o", linewidth=2.5, markersize=9,
        color="#2980B9", markerfacecolor="white", markeredgewidth=2.5)
for x, y in zip(FRACS, map50s):
    ax.annotate(f"{y:.3f}", (x, y),
                textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=11, fontweight="bold", color="#2980B9")

ax.set_xlabel("Training Data Fraction (%)")
ax.set_ylabel("Best mAP@50")
ax.set_title("Label Fraction Ablation — Learning Curve")
ax.set_xticks(FRACS)
ax.set_xticklabels(["10%", "25%", "50%", "100%"])
ax.set_ylim(0.2, 0.75)
plt.tight_layout()
plt.savefig(PLOTS / "ablation_1_learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ablation_1_learning_curve.png")

# ── Plot 2: Per-class AP across fractions ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(FRACS))
width = 0.18

for i, (cls, col) in enumerate(zip(CLASSES, COLORS)):
    vals = [per_class[f][i] for f in FRACS]
    bars = ax.bar(x + i*width - width*1.5, vals, width,
                  label=cls, color=col, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)

ax.set_xlabel("Training Data Fraction")
ax.set_ylabel("AP@50")
ax.set_title("Per-Class AP@50 Across Training Data Fractions")
ax.set_xticks(x)
ax.set_xticklabels(["10%", "25%", "50%", "100%"])
ax.legend(title="Damage Type", framealpha=0.9)
ax.set_ylim(0, 0.9)
plt.tight_layout()
plt.savefig(PLOTS / "ablation_2_perclass_fraction.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ablation_2_perclass_fraction.png")

# ── Plot 3: Imbalance comparison ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Overall mAP
df_def = pd.read_csv(SAVED / "imbalance_default" / "results.csv")
df_wgt = pd.read_csv(SAVED / "imbalance_weighted" / "results.csv")
df_def.columns = df_def.columns.str.strip()
df_wgt.columns = df_wgt.columns.str.strip()
overall = [df_def["metrics/mAP50(B)"].max(), df_wgt["metrics/mAP50(B)"].max()]
bars = axes[0].bar(["Default", "Weighted"], overall,
                   color=["#3498DB", "#E74C3C"], width=0.4, alpha=0.85, edgecolor="white")
for bar, v in zip(bars, overall):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")
axes[0].set_ylabel("mAP@50")
axes[0].set_title("Overall mAP@50")
axes[0].set_ylim(0, 0.75)

# Per-class
x = np.arange(len(CLASSES))
width = 0.35
for i, (label, col) in enumerate(zip(["Default", "Weighted"], ["#3498DB", "#E74C3C"])):
    vals = imbalance[label]
    bars = axes[1].bar(x + (i - 0.5)*width, vals, width,
                       label=label, color=col, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{v:.3f}", ha="center", fontsize=9)
axes[1].set_ylabel("AP@50")
axes[1].set_title("Per-Class AP@50")
axes[1].set_xticks(x)
axes[1].set_xticklabels(CLASSES)
axes[1].legend(framealpha=0.9)
axes[1].set_ylim(0, 0.9)
fig.suptitle("Label Imbalance Correction — Default vs Weighted Sampling", fontweight="bold", fontsize=14)
plt.tight_layout()
plt.savefig(PLOTS / "ablation_3_imbalance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ablation_3_imbalance.png")

# ── Plot 4: BBox area vs AP50 scatter ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
bbox_areas = [0.025, 0.018, 0.123, 0.061]
ap50s      = per_class[100]

for i, cls in enumerate(CLASSES):
    ax.scatter(bbox_areas[i], ap50s[i], s=250, color=COLORS[i],
               zorder=5, edgecolors="white", linewidth=1.5)
    ax.annotate(cls, (bbox_areas[i], ap50s[i]),
                textcoords="offset points", xytext=(10, 5),
                fontsize=13, fontweight="bold", color=COLORS[i])

ax.set_xlabel("Mean Bounding Box Area (normalized)")
ax.set_ylabel("AP@50 (100% data)")
ax.set_title("Damage Type Difficulty — BBox Area vs AP@50")
plt.tight_layout()
plt.savefig(PLOTS / "ablation_4_difficulty_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ablation_4_difficulty_scatter.png")

# ── Plot 5: Training curves per fraction ───────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
line_styles = ["-", "--", "-.", ":"]
for f, ls in zip(FRACS, line_styles):
    name = f"fraction_{f}pct"
    df = pd.read_csv(SAVED / name / "results.csv")
    df.columns = df.columns.str.strip()
    ax.plot(df["epoch"], df["metrics/mAP50(B)"],
            label=f"{f}% data", linewidth=2, linestyle=ls)

ax.set_xlabel("Epoch")
ax.set_ylabel("mAP@50")
ax.set_title("Training Progression — mAP@50 per Epoch by Data Fraction")
ax.legend(title="Data Fraction", framealpha=0.9)
plt.tight_layout()
plt.savefig(PLOTS / "ablation_5_training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ablation_5_training_curves.png")

print("\nAll 5 plots saved to:", PLOTS)

# ── Plot 6: Anchor Sensitivity ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

anchor_labels = ["COCO\nAnchors", "Custom\nAnchors"]
overall_anchor = [0.5931, 0.5898]
bars = axes[0].bar(anchor_labels, overall_anchor,
                   color=["#3498DB", "#E74C3C"], width=0.4, alpha=0.85, edgecolor="white")
for bar, v in zip(bars, overall_anchor):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")
axes[0].set_ylabel("mAP@50")
axes[0].set_title("Overall mAP@50")
axes[0].set_ylim(0.55, 0.65)
axes[0].grid(True, alpha=0.3, axis='y')

anchor_perclass = {
    "COCO":   [0.5121, 0.4833, 0.6355, 0.7416],
    "Custom": [0.5065, 0.4831, 0.6367, 0.7328],
}
x = np.arange(len(CLASSES))
width = 0.35
for i, (label, col) in enumerate(zip(["COCO", "Custom"], ["#3498DB", "#E74C3C"])):
    vals = anchor_perclass[label]
    bars = axes[1].bar(x + (i - 0.5)*width, vals, width,
                       label=f"{label} Anchors", color=col, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{v:.3f}", ha="center", fontsize=9)
axes[1].set_ylabel("AP@50")
axes[1].set_title("Per-Class AP@50")
axes[1].set_xticks(x)
axes[1].set_xticklabels(CLASSES)
axes[1].legend(framealpha=0.9)
axes[1].set_ylim(0, 0.9)
axes[1].grid(True, alpha=0.3, axis='y')
fig.suptitle("Anchor Sensitivity — COCO Default vs RDD2022 Custom Anchors",
             fontweight="bold", fontsize=14)
plt.tight_layout()
plt.savefig(PLOTS / "ablation_6_anchor.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ablation_6_anchor.png")

# ── Plot 7: NMS Threshold Sweep ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
iou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
map50_nms      = [0.5958, 0.6088, 0.6132, 0.6078, 0.5931]
map5095_nms    = [0.3064, 0.3128, 0.3186, 0.3218, 0.3208]

ax.plot(iou_thresholds, map50_nms, marker="o", linewidth=2.5, markersize=9,
        color="#2980B9", markerfacecolor="white", markeredgewidth=2.5, label="mAP@50")
ax.plot(iou_thresholds, map5095_nms, marker="s", linewidth=2.5, markersize=9,
        color="#E74C3C", markerfacecolor="white", markeredgewidth=2.5, label="mAP@50-95",
        linestyle="--")

# annotate peak
ax.annotate("Peak\nmAP@50=0.613", xy=(0.5, 0.6132),
            xytext=(0.55, 0.605), fontsize=10, color="#2980B9",
            arrowprops=dict(arrowstyle="->", color="#2980B9"))

for x, y in zip(iou_thresholds, map50_nms):
    ax.annotate(f"{y:.3f}", (x, y),
                textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=9, color="#2980B9")

ax.set_xlabel("NMS IoU Threshold")
ax.set_ylabel("mAP")
ax.set_title("NMS Threshold Sensitivity Sweep")
ax.set_xticks(iou_thresholds)
ax.legend(framealpha=0.9)
ax.set_ylim(0.28, 0.65)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS / "ablation_7_nms_sweep.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ablation_7_nms_sweep.png")

print("\nAll 7 plots saved to:", PLOTS)