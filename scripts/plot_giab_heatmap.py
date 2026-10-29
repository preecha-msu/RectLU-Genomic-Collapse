import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"E:\TeachPree\Research_Academic_Services\GENOME"
OUTPUT_DIR = os.path.join(BASE_DIR, "tuning_results")

df_1d = pd.read_csv(os.path.join(OUTPUT_DIR, "tuning_results_1d.csv"))
df_2d = pd.read_csv(os.path.join(OUTPUT_DIR, "tuning_results_2d.csv"))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

# 1. Heatmap 1D-CNN (Val_MCC_Tuned)
piv_1d = df_1d.pivot(index="Init_L", columns="Init_W", values="Val_MCC_Tuned")
piv_1d = piv_1d.sort_index(ascending=False)
im1 = axes[0].imshow(piv_1d.values, cmap="viridis", aspect="auto", vmin=0.0, vmax=0.55)
axes[0].set_xticks(range(len(piv_1d.columns)))
axes[0].set_xticklabels([f"{w:.2f}" for w in piv_1d.columns])
axes[0].set_yticks(range(len(piv_1d.index)))
axes[0].set_yticklabels([f"{l:.1f}" for l in piv_1d.index])
axes[0].set_xlabel("Initial Width $W$", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Initial Threshold $L$", fontsize=11, fontweight="bold")
axes[0].set_title("(a) 1D-CNN GIAB $k$-mer Tuning (Calibrated MCC)", fontsize=11, fontweight="bold")

for i in range(piv_1d.shape[0]):
    for j in range(piv_1d.shape[1]):
        val = piv_1d.values[i, j]
        color = "white" if val < 0.30 else "black"
        axes[0].text(j, i, f"{val:.4f}", ha="center", va="center", color=color, fontsize=8.5, fontweight="bold")
fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label="Validation MCC")

# 2. Heatmap 2D-CNN (Val_MCC_05)
piv_2d = df_2d.pivot(index="Init_L", columns="Init_W", values="Val_MCC_05")
piv_2d = piv_2d.sort_index(ascending=False)
im2 = axes[1].imshow(piv_2d.values, cmap="plasma", aspect="auto", vmin=0.0, vmax=1.0)
axes[1].set_xticks(range(len(piv_2d.columns)))
axes[1].set_xticklabels([f"{w:.2f}" for w in piv_2d.columns])
axes[1].set_yticks(range(len(piv_2d.index)))
axes[1].set_yticklabels([f"{l:.1f}" for l in piv_2d.index])
axes[1].set_xlabel("Initial Width $W$", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Initial Threshold $L$", fontsize=11, fontweight="bold")
axes[1].set_title("(b) 2D-CNN GIAB Pileup Tuning (MCC at $\\tau=0.5$)", fontsize=11, fontweight="bold")

for i in range(piv_2d.shape[0]):
    for j in range(piv_2d.shape[1]):
        val = piv_2d.values[i, j]
        color = "white" if val < 0.50 else "black"
        axes[1].text(j, i, f"{val:.4f}", ha="center", va="center", color=color, fontsize=8.5, fontweight="bold")
fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="Validation MCC")

plt.tight_layout()
pdf_out = os.path.join(OUTPUT_DIR, "Fig_GIAB_WL_Tuning_Heatmap_300dpi.pdf")
png_out = os.path.join(OUTPUT_DIR, "Fig_GIAB_WL_Tuning_Heatmap_300dpi.png")
plt.savefig(pdf_out, dpi=300, bbox_inches="tight")
plt.savefig(png_out, dpi=300, bbox_inches="tight")
print("Saved 300 DPI Heatmap successfully:")
print(" ->", pdf_out)
print(" ->", png_out)
