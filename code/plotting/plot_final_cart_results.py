"""
CartPole figures comparing the PyTorch baseline with the FP32 / FP16 /
INT8 TensorRT engines across batch sizes: latency (mean, P95, spread),
throughput, power, energy per sample and FPS/W.

Reads final_results.csv (from merge_results.py); writes PDF and PNG
files to figures_ieee/.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =========================
# CONFIG
# =========================
CSV_PATH = "final_results.csv"
OUT_DIR = "figures_ieee"

os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)

models = df["model"].unique()
batches = sorted(df["batch"].unique())


# =========================
# HELPERS
# =========================
def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{name}.pdf")
    fig.savefig(f"{OUT_DIR}/{name}.png")
    plt.close(fig)


# =========================
# 1. LATENCY ANALYSIS
# =========================

# --- (A) Mean latency vs batch ---
fig, ax = plt.subplots()

for m in models:
    d = df[df["model"] == m].groupby("batch").mean(numeric_only=True).reset_index()
    ax.plot(d["batch"], d["latency_mean_ms"], marker="o", label=m)

ax.set_xscale("log", base=2)
ax.set_xlabel("Batch size")
ax.set_ylabel("Latency (ms)")
ax.set_title("Mean Latency vs Batch Size")
ax.legend()

save(fig, "latency_mean")


# --- (B) P95 (tail) latency vs batch ---
fig, ax = plt.subplots()

for m in models:
    d = df[df["model"] == m].groupby("batch").mean(numeric_only=True).reset_index()
    ax.plot(d["batch"], d["latency_p95_ms"], marker="s", linestyle="--", label=m)

ax.set_xscale("log", base=2)
ax.set_xlabel("Batch size")
ax.set_ylabel("P95 Latency (ms)")
ax.set_title("Tail Latency (P95) vs Batch Size")
ax.legend()

save(fig, "latency_p95")


# --- (C) Spread of mean latency across rows of final_results.csv ---
fig, ax = plt.subplots()

sns.boxplot(
    data=df,
    x="batch",
    y="latency_mean_ms",
    hue="model",
    ax=ax
)

ax.set_xscale("log", base=2)
ax.set_title("Latency Distribution Across All Runs")
ax.set_ylabel("Latency (ms)")

save(fig, "latency_distribution")


# =========================
# 2. THROUGHPUT ANALYSIS
# =========================

fig, ax = plt.subplots()

for m in models:
    d = df[df["model"] == m].groupby("batch").mean(numeric_only=True).reset_index()
    ax.plot(d["batch"], d["throughput"], marker="o", label=m)

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Batch size")
ax.set_ylabel("Throughput (samples/s)")
ax.set_title("Throughput Scaling vs Batch Size")
ax.legend()

save(fig, "throughput")


# --- Throughput, one point per row ---
fig, ax = plt.subplots()

sns.stripplot(
    data=df,
    x="batch",
    y="throughput",
    hue="model",
    dodge=True,
    alpha=0.6
)

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_title("Throughput Variability Across Seeds")

save(fig, "throughput_scatter")


# =========================
# 3. POWER ANALYSIS
# =========================

fig, ax = plt.subplots()

for m in models:
    d = df[df["model"] == m].groupby("batch").mean(numeric_only=True).reset_index()
    ax.plot(d["batch"], d["avg_power_w"], marker="o", label=m)

ax.set_xscale("log", base=2)
ax.set_xlabel("Batch size")
ax.set_ylabel("Power (W)")
ax.set_title("Average Power Consumption")
ax.legend()

save(fig, "power")


# --- Spread of average power ---
fig, ax = plt.subplots()

sns.boxplot(
    data=df,
    x="batch",
    y="avg_power_w",
    hue="model",
    ax=ax
)

ax.set_xscale("log", base=2)
ax.set_title("Power Distribution Across Runs")

save(fig, "power_distribution")


# =========================
# 4. ENERGY ANALYSIS
# =========================

fig, ax = plt.subplots()

for m in models:
    d = df[df["model"] == m].groupby("batch").mean(numeric_only=True).reset_index()
    ax.plot(d["batch"], d["energy_per_sample_mj"], marker="o", label=m)

ax.set_xscale("log", base=2)
ax.set_xlabel("Batch size")
ax.set_ylabel("Energy per sample (mJ)")
ax.set_title("Energy Efficiency Scaling")
ax.legend()

save(fig, "energy")


# =========================
# 5. EFFICIENCY (FPS/WATT)
# =========================

fig, ax = plt.subplots()

for m in models:
    d = df[df["model"] == m].groupby("batch").mean(numeric_only=True).reset_index()
    ax.plot(d["batch"], d["fps_per_watt"], marker="o", label=m)

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Batch size")
ax.set_ylabel("FPS per Watt")
ax.set_title("Energy Efficiency Scaling (FPS/W)")
ax.legend()

save(fig, "efficiency")


# =========================
# DONE
# =========================
print(f"Saved IEEE figures to: {OUT_DIR}/")
