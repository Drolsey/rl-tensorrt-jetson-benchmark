"""
Heatmap of effect sizes from stat_tests.py.

Reads experiments/stats/welch_ttest_results.csv (CartPole, TensorRT
engines). Rows are metric x engine pair, columns are batch sizes, cell
color is |Cohen's d|, and '*' marks cells with p < 0.05.

Output: experiments/stats/figures/effect_size_heatmap.{png,pdf}
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IN_PATH = "experiments/stats/welch_ttest_results.csv"
OUT_DIR = "experiments/stats/figures"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN_PATH)

df["comparison"] = df["engine_a"] + " vs " + df["engine_b"]
df["row_label"] = df["metric"] + " | " + df["comparison"]

batches = sorted(df["batch"].unique())
row_labels = sorted(df["row_label"].unique())

# build matrix: rows = metric/comparison, cols = batch
d_matrix = np.full((len(row_labels), len(batches)), np.nan)
sig_matrix = np.zeros_like(d_matrix, dtype=bool)

row_idx = {r: i for i, r in enumerate(row_labels)}
col_idx = {b: i for i, b in enumerate(batches)}

for _, row in df.iterrows():
    r = row_idx[row["row_label"]]
    c = col_idx[row["batch"]]
    d_matrix[r, c] = row["cohens_d"]
    sig_matrix[r, c] = bool(row["significant_at_0.05"])

fig, ax = plt.subplots(figsize=(8, max(6, 0.32 * len(row_labels))))

im = ax.imshow(np.abs(d_matrix), aspect="auto", cmap="viridis")
ax.set_xticks(range(len(batches)))
ax.set_xticklabels(batches)
ax.set_xlabel("Batch size")
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=7)
ax.set_title("Effect Size |Cohen's d| (n=3 seeds/group - interpret with caution)")

# overlay significance markers
for r in range(len(row_labels)):
    for c in range(len(batches)):
        if sig_matrix[r, c]:
            ax.text(c, r, "*", ha="center", va="center", color="white", fontsize=10)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("|Cohen's d|")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "effect_size_heatmap.png"), dpi=300)
fig.savefig(os.path.join(OUT_DIR, "effect_size_heatmap.pdf"))
plt.close(fig)

print(f"Saved -> {OUT_DIR}/effect_size_heatmap.png (+ .pdf)")
print("'*' marks cells significant at alpha=0.05 (Welch's t-test). "
      "Caveat: n=3 seeds/group gives low statistical power.")
