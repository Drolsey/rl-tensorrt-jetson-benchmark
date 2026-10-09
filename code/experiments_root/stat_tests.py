"""
Welch's t-tests and Cohen's d between precisions, computed from
batch_results_raw.csv (3 precisions x 6 batch sizes x 3 seeds = 54
rows); no new benchmark runs are needed.

For every batch size, metric and engine pair:
  - Welch's t-test (scipy.stats.ttest_ind, equal_var=False)
  - Cohen's d with the pooled standard deviation
  - whether p < 0.05

With n=3 seeds per group the tests have low statistical power, so the
p-values and effect sizes are indicative only. The output CSV carries
a caveat column saying so.

Output: experiments/stats/welch_ttest_results.csv
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

IN_PATH = "batch_results_raw.csv"
OUT_DIR = "experiments/stats"
OUT_PATH = os.path.join(OUT_DIR, "welch_ttest_results.csv")

os.makedirs(OUT_DIR, exist_ok=True)

METRICS = [
    "latency_mean_ms",
    "latency_p95_ms",
    "throughput",
    "energy_per_sample_mj",
    "fps_per_watt",
]

COMPARISONS = [
    ("FP32", "FP16"),
    ("FP32", "INT8"),
    ("FP16", "INT8"),
]

ALPHA = 0.05


def cohens_d(a, b):
    """(mean(a) - mean(b)) / pooled SD; NaN if a group has < 2 values or SD is 0."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return np.nan
    return (np.mean(a) - np.mean(b)) / pooled_sd


def main():
    df = pd.read_csv(IN_PATH)

    rows = []

    for batch in sorted(df["batch"].unique()):
        d_batch = df[df["batch"] == batch]

        for metric in METRICS:
            if metric not in d_batch.columns:
                continue

            for eng_a, eng_b in COMPARISONS:
                a = d_batch[d_batch["precision"] == eng_a][metric].dropna().values
                b = d_batch[d_batch["precision"] == eng_b][metric].dropna().values

                if len(a) < 2 or len(b) < 2:
                    t_stat, p_val = np.nan, np.nan
                else:
                    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)

                d = cohens_d(a, b)

                rows.append({
                    "batch": batch,
                    "metric": metric,
                    "engine_a": eng_a,
                    "engine_b": eng_b,
                    "n_a": len(a),
                    "n_b": len(b),
                    "mean_a": np.mean(a) if len(a) else np.nan,
                    "mean_b": np.mean(b) if len(b) else np.nan,
                    "t_statistic": t_stat,
                    "p_value": p_val,
                    "cohens_d": d,
                    "significant_at_0.05": bool(p_val < ALPHA) if pd.notna(p_val) else False,
                    "caveat": "n=3 seeds/group: low statistical power, interpret cautiously",
                })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False)

    print(f"Saved {len(out_df)} comparisons -> {OUT_PATH}")
    print(out_df.head(15))


if __name__ == "__main__":
    main()
