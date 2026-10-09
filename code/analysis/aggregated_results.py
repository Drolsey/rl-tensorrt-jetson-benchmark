"""
Summarises final_results.csv (from merge_results.py) as mean and std
per (model, batch size) -> final_results_agg.csv.
"""

import pandas as pd

df = pd.read_csv("final_results.csv")

grouped = df.groupby(["model", "batch"]).agg({
    "latency_mean_ms": ["mean", "std"],
    "latency_p95_ms": ["mean", "std"],
    "latency_std_ms": "mean",
    "throughput": ["mean", "std"],
    "avg_power_w": ["mean", "std"]
}).reset_index()

# flatten (metric, stat) column pairs into names like "throughput_mean"
grouped.columns = [
    "_".join(col).strip("_") if col[1] != "" else col[0]
    for col in grouped.columns
]

grouped.to_csv("final_results_agg.csv", index=False)
