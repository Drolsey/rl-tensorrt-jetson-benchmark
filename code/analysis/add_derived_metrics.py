"""
Adds scaling_efficiency and latency_growth_factor columns to
batch_results.csv (overwrites the file in place).

run_full_benchmark.py already computes both columns, so this is only
needed for results files produced before that was added.
"""

import pandas as pd

df = pd.read_csv("batch_results.csv")

# =========================
# SCALING EFFICIENCY
# measured throughput / (batch-1 throughput x batch size);
# 1.0 means throughput grows perfectly linearly with batch size
# =========================
base_tp = {}

for p in df["precision"].unique():
    tp1 = df[(df["precision"] == p) & (df["batch"] == 1)]["throughput"].values[0]
    base_tp[p] = tp1

efficiencies = []

for _, row in df.iterrows():
    ideal = base_tp[row["precision"]] * row["batch"]
    efficiency = row["throughput"] / ideal
    efficiencies.append(efficiency)

df["scaling_efficiency"] = efficiencies

# =========================
# LATENCY GROWTH FACTOR
# mean latency relative to batch 1 at the same precision
# =========================
base_lat = {}

for p in df["precision"].unique():
    lat1 = df[(df["precision"] == p) & (df["batch"] == 1)]["latency_mean_ms"].values[0]
    base_lat[p] = lat1

growth = []

for _, row in df.iterrows():
    growth.append(row["latency_mean_ms"] / base_lat[row["precision"]])

df["latency_growth_factor"] = growth

# =========================
# SAVE
# =========================
df.to_csv("batch_results.csv", index=False)

print("\nUpdated batch_results.csv with:")
print(" - scaling_efficiency")
print(" - latency_growth_factor")
