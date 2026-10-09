"""
One 2x3 overview figure per environment (CartPole and LunarLander):
mean latency, P95 latency, throughput, power, energy per sample and
FPS/W against batch size.

Reads the per-seed *_raw.csv files and aggregates them here, so each
line is the mean across seeds with a shaded +/-1 std band.

Output:
    experiments/figures_clear/cartpole_overview.{png,pdf}
    experiments/figures_clear/lunarlander_overview.{png,pdf}
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from plot_style import apply_style, style_batch_axis, plot_mean_with_band, PRECISION_STYLE

OUT_DIR = "experiments/figures_clear"
os.makedirs(OUT_DIR, exist_ok=True)

apply_style()

# (column, std column, y-axis label). The std column is not read: the
# band is the std across seeds, computed in make_overview().
METRICS = [
    ("latency_mean_ms", "latency_std_ms", "Mean Latency (ms)"),
    ("latency_p95_ms", None, "P95 Latency (ms)"),
    ("throughput", "throughput_std", "Throughput (samples/s)"),
    ("avg_power_w", "avg_power_w_std", "Power (W)"),
    ("energy_per_sample_mj", "energy_per_sample_mj_std", "Energy / sample (mJ)"),
    ("fps_per_watt", "fps_per_watt_std", "FPS / Watt"),
]

LOG_Y_METRICS = {"throughput", "fps_per_watt"}


def make_overview(raw_csv_path, title, out_name, precisions=("FP32", "FP16", "INT8")):
    raw = pd.read_csv(raw_csv_path)

    # aggregate across seeds: mean + std per (precision, batch)
    agg_cols = [m for m, _, _ in METRICS if m in raw.columns]
    agg = (
        raw.groupby(["precision", "batch"])[agg_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = [
        "_".join(c).strip("_") if c[1] != "" else c[0] for c in agg.columns
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, (metric, std_metric, ylabel) in zip(axes, METRICS):
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in agg.columns:
            ax.set_visible(False)
            continue

        # plot_mean_with_band expects the mean under the metric's own name
        plot_df = agg.rename(columns={mean_col: metric, std_col: f"{metric}__std"})
        plot_mean_with_band(
            ax, plot_df, x_col="batch", y_col=metric,
            std_col=f"{metric}__std", precision_list=precisions,
        )

        style_batch_axis(ax)
        if metric in LOG_Y_METRICS:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", framealpha=0.9)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(os.path.join(OUT_DIR, f"{out_name}.png"))
    fig.savefig(os.path.join(OUT_DIR, f"{out_name}.pdf"))
    plt.close(fig)

    print(f"Saved -> {OUT_DIR}/{out_name}.png (+ .pdf)")


if __name__ == "__main__":
    make_overview(
        "batch_results_raw.csv",
        "CartPole-v1 (obs=4): TensorRT Inference Benchmark Overview",
        "cartpole_overview",
    )
    make_overview(
        "experiments/lunarlander/ll_results_raw.csv",
        "LunarLander-v3 (obs=8): TensorRT Inference Benchmark Overview",
        "lunarlander_overview",
    )
