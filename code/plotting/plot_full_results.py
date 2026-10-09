"""
Quick-look plots of the TensorRT-only results in batch_results.csv
(already averaged over seeds): latency, throughput, power, energy per
sample, FPS/W and a latency-vs-throughput scatter. PNGs go to
results_plots/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# OUTPUT DIRECTORY
# =========================
OUT_DIR = "results_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("batch_results.csv")

precisions = df["precision"].unique()

# =========================
# STYLE
# =========================
plt.style.use("seaborn-v0_8-darkgrid")

# =========================
# 1. LATENCY vs BATCH
# =========================
plt.figure()

for p in precisions:
    d = df[df["precision"] == p]
    plt.plot(d["batch"], d["latency_mean_ms"], marker="o", label=p)

plt.xlabel("Batch Size")
plt.ylabel("Latency (ms)")
plt.title("Inference Latency vs Batch Size")
plt.legend()

plt.savefig(os.path.join(OUT_DIR, "latency_vs_batch.png"), dpi=300, bbox_inches="tight")
plt.close()


# =========================
# 2. THROUGHPUT vs BATCH
# =========================
plt.figure()

for p in precisions:
    d = df[df["precision"] == p]
    plt.plot(d["batch"], d["throughput"], marker="o", label=p)

plt.xlabel("Batch Size")
plt.ylabel("Throughput (samples/sec)")
plt.title("Throughput vs Batch Size")
plt.legend()

plt.savefig(os.path.join(OUT_DIR, "throughput_vs_batch.png"), dpi=300, bbox_inches="tight")
plt.close()


# =========================
# 3. POWER vs BATCH
# =========================
plt.figure()

for p in precisions:
    d = df[df["precision"] == p]
    plt.plot(d["batch"], d["avg_power_w"], marker="o", label=p)

plt.xlabel("Batch Size")
plt.ylabel("Power (W)")
plt.title("Power Consumption vs Batch Size")
plt.legend()

plt.savefig(os.path.join(OUT_DIR, "power_vs_batch.png"), dpi=300, bbox_inches="tight")
plt.close()


# =========================
# 4. ENERGY PER SAMPLE
# =========================
plt.figure()

for p in precisions:
    d = df[df["precision"] == p]
    # W / (samples/s) = J per sample; x1000 for mJ
    plt.plot(d["batch"], (d["avg_power_w"] / d["throughput"]) * 1000, marker="o", label=p)

plt.xlabel("Batch Size")
plt.ylabel("Energy per Sample (mJ)")
plt.title("Energy Efficiency vs Batch Size")
plt.legend()

plt.savefig(os.path.join(OUT_DIR, "energy_vs_batch.png"), dpi=300, bbox_inches="tight")
plt.close()


# =========================
# 5. FPS / WATT
# =========================
plt.figure()

for p in precisions:
    d = df[df["precision"] == p]
    plt.plot(d["batch"], d["fps_per_watt"], marker="o", label=p)

plt.xlabel("Batch Size")
plt.ylabel("FPS / Watt")
plt.title("Energy Efficiency (FPS/W)")
plt.legend()

plt.savefig(os.path.join(OUT_DIR, "fps_per_watt.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================
# 6. EFFICIENCY FRONTIER
# =========================
plt.figure()

colors = {
    "FP32": "blue",
    "FP16": "green",
    "INT8": "red"
}

for p in precisions:
    d = df[df["precision"] == p]

    plt.scatter(
        d["latency_mean_ms"],
        d["throughput"],
        s=d["batch"] * 40,   # marker area grows with batch size
        alpha=0.7,
        label=p,
        color=colors.get(p, "black")
    )

plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (samples/sec)")
plt.title("Efficiency Frontier: Latency vs Throughput")
plt.legend()

plt.savefig(os.path.join(OUT_DIR, "efficiency_frontier.png"),
            dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved all plots to: {OUT_DIR}/")

