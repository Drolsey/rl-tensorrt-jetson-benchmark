"""
CartPole vs LunarLander comparison of the aggregated batch-benchmark
results.

Shows whether the ordering of FP32 / FP16 / INT8 (latency and
efficiency) and the power/energy trends stay the same when the
observation size grows from 4 (CartPole) to 8 (LunarLander).

Reads:
    batch_results.csv               (CartPole, aggregated)
    experiments/lunarlander/ll_results.csv  (LunarLander, aggregated)

Output: experiments/cross_environment/figures/*.{png,pdf}
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

CARTPOLE_CSV = "batch_results.csv"
LUNARLANDER_CSV = "experiments/lunarlander/ll_results.csv"
OUT_DIR = "experiments/cross_environment/figures"
os.makedirs(OUT_DIR, exist_ok=True)

cp = pd.read_csv(CARTPOLE_CSV)
cp["environment"] = "CartPole-v1 (obs=4)"

ll = pd.read_csv(LUNARLANDER_CSV)
ll["environment"] = "LunarLander-v3 (obs=8)"

df = pd.concat([cp, ll], ignore_index=True)

precisions = ["FP32", "FP16", "INT8"]
environments = df["environment"].unique()


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=300)
    fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"))
    plt.close(fig)


# =========================
# 1. Latency vs batch, side-by-side panels per environment
# =========================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

for ax, env in zip(axes, environments):
    d_env = df[df["environment"] == env]
    for p in precisions:
        d = d_env[d_env["precision"] == p]
        if len(d) == 0:
            continue
        ax.plot(d["batch"], d["latency_mean_ms"], marker="o", label=p)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(env)
    ax.legend()

fig.suptitle("Cross-Environment Latency Comparison")
save(fig, "latency_cross_env")


# =========================
# 2. FPS/Watt vs batch, side-by-side panels
# =========================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

for ax, env in zip(axes, environments):
    d_env = df[df["environment"] == env]
    for p in precisions:
        d = d_env[d_env["precision"] == p]
        if len(d) == 0:
            continue
        ax.plot(d["batch"], d["fps_per_watt"], marker="o", label=p)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("FPS per Watt")
    ax.set_title(env)
    ax.legend()

fig.suptitle("Cross-Environment Energy Efficiency Comparison")
save(fig, "efficiency_cross_env")


# =========================
# 3. Latency per precision at batch=1 (one observation per control step)
# =========================
fig, ax = plt.subplots(figsize=(7, 4.5))

batch1 = df[df["batch"] == 1]
width = 0.25
import numpy as np
x = np.arange(len(environments))

for i, p in enumerate(precisions):
    vals = []
    for env in environments:
        row = batch1[(batch1["environment"] == env) & (batch1["precision"] == p)]
        vals.append(row["latency_mean_ms"].values[0] if len(row) else np.nan)
    ax.bar(x + i * width - width, vals, width, label=p)

ax.set_xticks(x)
ax.set_xticklabels(environments)
ax.set_ylabel("Latency at batch=1 (ms)")
ax.set_title("Precision Ranking at Batch=1: CartPole vs LunarLander")
ax.legend()
save(fig, "precision_ranking_batch1")

print(f"Saved cross-environment figures to: {OUT_DIR}/")
