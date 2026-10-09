"""
Bar charts from the earlier single-sample evaluation (results_clean.csv):
mean latency per model with 95% CI error bars, and speedup relative to
the first row (FP32). Saved to figures/.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("results_clean.csv")


labels = df["model"]
means = df["latency_mean_ms"]
ci95 = df["latency_ci95_ms"]


# =========================
# CREATE OUTPUT FOLDER
# =========================
import os
os.makedirs("figures", exist_ok=True)


# =========================
# BAR PLOT
# =========================
plt.figure(figsize=(7,4))
plt.bar(labels, means, yerr=ci95, capsize=6)
plt.ylabel("Latency (ms)")
plt.title("Inference Latency Comparison (Jetson Orin)")
plt.tight_layout()
plt.savefig("figures/latency_bar.png", dpi=300)


# =========================
# SPEEDUP
# =========================
baseline = means.iloc[0]  # first row is the FP32 reference
speedup = baseline / means

plt.figure(figsize=(7,4))
plt.bar(labels, speedup)
plt.ylabel("Speedup (× FP32)")
plt.title("TensorRT Acceleration")
plt.tight_layout()
plt.savefig("figures/speedup.png", dpi=300)


print("Figures saved in /figures")
