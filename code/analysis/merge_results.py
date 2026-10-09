"""
Combines the TensorRT batch results and the PyTorch CPU baseline into a
single table, final_results.csv, which the CartPole plotting script reads.
"""

import pandas as pd
import numpy as np

# =========================
# LOAD DATASETS
# =========================
# Note: batch_results.csv is already averaged over seeds, whereas
# baseline_results_raw.csv has one row per seed.
trt_df = pd.read_csv("batch_results.csv")
torch_df = pd.read_csv("baseline_results_raw.csv")

print("\nLoaded:")
print("TensorRT:", trt_df.shape)
print("PyTorch :", torch_df.shape)


# =========================
# STANDARDIZE COLUMN NAMES
# =========================
# The baseline already uses the same column names; this mapping is where
# any name that differs would be fixed.

torch_df = torch_df.rename(columns={
    "fps_per_watt": "fps_per_watt",
    "energy_per_sample_mj": "energy_per_sample_mj"
})

# Fill metrics the baseline doesn't record with NaN so both tables share
# the same columns
required_cols = [
    "latency_mean_ms",
    "latency_p95_ms",
    "latency_std_ms",
    "latency_ci95_ms",
    "throughput",
    "avg_power_w",
    "fps_per_watt",
    "energy_per_sample_mj"
]

for col in required_cols:
    if col not in torch_df.columns:
        torch_df[col] = np.nan


# =========================
# ADD IDENTIFIER
# =========================
trt_df["model"] = trt_df["precision"]
torch_df["model"] = "PYTORCH"


# =========================
# ALIGN COLUMNS
# =========================
common_cols = [
    "model",
    "batch",
    "latency_mean_ms",
    "latency_p95_ms",
    "latency_std_ms",
    "throughput",
    "avg_power_w",
    "fps_per_watt",
    "energy_per_sample_mj"
]

# Keep only relevant columns
trt_clean = trt_df[common_cols]
torch_clean = torch_df[common_cols]


# =========================
# COMBINE DATASETS
# =========================
final_df = pd.concat([trt_clean, torch_clean], ignore_index=True)


# =========================
# SORT FOR PLOTTING
# =========================
model_order = ["PYTORCH", "FP32", "FP16", "INT8"]

final_df["model"] = pd.Categorical(final_df["model"], categories=model_order, ordered=True)
final_df = final_df.sort_values(["model", "batch"])


# =========================
# SAVE OUTPUTS
# =========================
final_df.to_csv("final_results.csv", index=False)

print("\nSaved merged dataset -> final_results.csv")
print("Shape:", final_df.shape)


# =========================
# QUICK SANITY CHECK
# =========================
print("\nModels included:")
print(final_df["model"].unique())

print("\nBatch sizes:")
print(sorted(final_df["batch"].unique()))
