"""
Shared plotting style for the figures.

Call apply_style() once at the top of a plotting script.
PRECISION_STYLE[precision] gives the color, marker and line style used
for each of FP32 / FP16 / INT8 / PYTORCH, so they look the same in
every figure.
"""

import matplotlib.pyplot as plt

PRECISION_STYLE = {
    "FP32": dict(color="#1f3a93", marker="o", linestyle="-"),   # navy
    "FP16": dict(color="#1a9988", marker="s", linestyle="-"),   # teal
    "INT8": dict(color="#c0392b", marker="^", linestyle="-"),   # crimson
    "PYTORCH": dict(color="#7f8c8d", marker="D", linestyle="--"),  # grey, dashed (baseline)
}

BATCH_TICKS = [1, 2, 4, 8, 16, 32]


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2,
        "lines.markersize": 7,
    })


def style_batch_axis(ax, log=True):
    """Batch-size x-axis (log2 by default) with ticks at 1, 2, 4, ..., 32."""
    if log:
        ax.set_xscale("log", base=2)
    ax.set_xticks(BATCH_TICKS)
    ax.set_xticklabels([str(b) for b in BATCH_TICKS])
    ax.set_xlabel("Batch size")


def plot_mean_with_band(ax, df, x_col, y_col, std_col, precision_list, label_suffix=""):
    """
    Draws the mean line and a shaded +/-1 std band for each precision.
    df must already be aggregated: one row per (precision, x_col), with
    the mean in y_col and the std across seeds in std_col.
    """
    for p in precision_list:
        d = df[df["precision"] == p].sort_values(x_col)
        if len(d) == 0:
            continue
        style = PRECISION_STYLE.get(p, {})
        ax.plot(d[x_col], d[y_col], label=f"{p}{label_suffix}", **style)
        if std_col in d.columns:
            ax.fill_between(
                d[x_col],
                d[y_col] - d[std_col].fillna(0),
                d[y_col] + d[std_col].fillna(0),
                color=style.get("color", "gray"),
                alpha=0.15,
            )
