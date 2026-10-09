"""
Policy fidelity comparison plot - CartPole vs LunarLander.

Reads:
    experiments/fidelity/fidelity_results.csv       (CartPole, aggregated)
    experiments/lunarlander/ll_fidelity_results.csv (LunarLander, aggregated)

Produces a 2-panel figure:
    (A) action_agreement vs FP32, grouped bars by engine x environment
    (B) reward_degradation_pct, grouped bars by engine x environment

Output: experiments/fidelity/figures/fidelity_comparison.{png,pdf}
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CARTPOLE_CSV = "experiments/fidelity/fidelity_results.csv"
LUNARLANDER_CSV = "experiments/lunarlander/ll_fidelity_results.csv"
OUT_DIR = "experiments/fidelity/figures"
os.makedirs(OUT_DIR, exist_ok=True)

cp = pd.read_csv(CARTPOLE_CSV)
cp["environment"] = "CartPole-v1"

ll = pd.read_csv(LUNARLANDER_CSV)
ll["environment"] = "LunarLander-v3"

df = pd.concat([cp, ll], ignore_index=True)

# FP32 is the reference (agreement 1, degradation 0), so only FP16/INT8 are plotted
df_compare = df[df["engine"] != "FP32"].copy()

engines = ["FP16", "INT8"]
environments = ["CartPole-v1", "LunarLander-v3"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# ---- Panel A: action agreement ----
ax = axes[0]
width = 0.35
x = np.arange(len(engines))

for i, env in enumerate(environments):
    vals = [
        df_compare[(df_compare["engine"] == e) & (df_compare["environment"] == env)]
        ["mean_action_agreement"].values[0]
        for e in engines
    ]
    ax.bar(x + i * width - width / 2, vals, width, label=env)

ax.set_xticks(x)
ax.set_xticklabels(engines)
ax.set_ylabel("Mean action agreement vs FP32")
ax.set_title("Action Agreement vs FP32")
ax.set_ylim(0, 1.05)
ax.legend()

# ---- Panel B: reward degradation ----
ax = axes[1]
for i, env in enumerate(environments):
    vals = [
        df_compare[(df_compare["engine"] == e) & (df_compare["environment"] == env)]
        ["reward_degradation_pct"].values[0]
        for e in engines
    ]
    ax.bar(x + i * width - width / 2, vals, width, label=env)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(engines)
ax.set_ylabel("Reward degradation vs FP32 (%)")
ax.set_title("Reward Degradation vs FP32")
ax.legend()

fig.suptitle("Policy Fidelity: CartPole vs LunarLander")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fidelity_comparison.png"), dpi=300)
fig.savefig(os.path.join(OUT_DIR, "fidelity_comparison.pdf"))
plt.close(fig)

print(f"Saved -> {OUT_DIR}/fidelity_comparison.png (+ .pdf)")
