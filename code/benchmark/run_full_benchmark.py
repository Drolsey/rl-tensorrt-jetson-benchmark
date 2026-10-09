"""
Main CartPole batch benchmark: FP32 / FP16 / INT8 TensorRT engines on
the Jetson.

For each seed (3), engine and batch size (1-32): 200 warm-up calls,
then 2000 timed calls on batches drawn from cartpole_states.npy, with
board power sampled by tegrastats during the timed loop.

Output:
    batch_results_raw.csv   one row per (seed, precision, batch)
    batch_results.csv       mean/std across seeds + derived metrics
"""

import time
import numpy as np
import pandas as pd
import subprocess
import os

from policies.trt_batch_policy import TRTBatchPolicy

# =========================
# CONFIG
# =========================
states = np.load("cartpole_states.npy")
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
N_RUNS = 2000
WARMUP_RUNS = 200

SEEDS = [0, 1, 2]

MODELS = {
    "FP32": "policy_fp32_dynamic.trt",
    "FP16": "policy_fp16_dynamic.trt",
    "INT8": "policy_int8_dynamic.trt"
}

POWER_LOG_FILE = "power_log.txt"
os.makedirs("logs", exist_ok=True)


# =========================
# UTIL
# =========================
def ci95(x):
    return 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))


def parse_power(file):
    """
    Mean board input power (W) from a tegrastats log. Each line reports
    VDD_IN as <current>mW/<average>mW; the current value is used.
    """
    powers = []

    with open(file, "r") as f:
        for line in f:
            if "VDD_IN" in line:
                try:
                    parts = line.split("VDD_IN")[-1]
                    mw = parts.split("mW")[0].split("/")[-1].strip()
                    powers.append(float(mw) / 1000.0)
                except:
                    continue

    return np.mean(powers) if len(powers) > 0 else 0.0


def start_power_logger():
    return subprocess.Popen(
        ["tegrastats", "--interval", "100"],
        stdout=open(POWER_LOG_FILE, "w"),
        stderr=subprocess.DEVNULL
    )


# =========================
# MAIN BENCHMARK
# =========================
all_results = []

print("\n=== STEP 3 BENCHMARK (FP32 vs FP16 vs INT8) ===\n")

# =========================
# SEED LOOP
# the seed controls which states are sampled; the *_std columns in
# batch_results.csv are the spread across these seeds
# =========================
for seed in SEEDS:
    np.random.seed(seed)

    print(f"\n===== SEED {seed} =====\n")

    for precision, engine_path in MODELS.items():

        print(f"\n================ {precision} ================\n")

        model = TRTBatchPolicy(engine_path)

        for batch in BATCH_SIZES:

            print(f"\n--- Batch {batch} ---")

            idxs = np.random.randint(0, len(states), size=batch)
            obs = states[idxs]

            # =========================
            # WARMUP (one fixed batch, not timed)
            # =========================
            for _ in range(WARMUP_RUNS):
                _ = model(obs)

            time.sleep(1)  # short idle gap before the measured run

            latencies_ms = []
            kernel_times_ms = []

            # =========================
            # POWER START
            # =========================
            power_proc = start_power_logger()
            time.sleep(1)  # give tegrastats time to start sampling

            start_total = time.perf_counter()

            # =========================
            # BENCHMARK LOOP
            # Each call takes a contiguous slice of the state pool at a
            # random offset.
            #   latency_ms = wall-clock time of the whole call
            #                (copies + execution + sync)
            #   kernel_ms  = engine execution only (CUDA events)
            # =========================
            for _ in range(N_RUNS):

                idx = np.random.randint(0, len(states) - batch)
                obs = states[idx:idx + batch]

                t0 = time.perf_counter()
                kernel_ms = model(obs, return_kernel_time=True)
                t1 = time.perf_counter()

                latencies_ms.append((t1 - t0) * 1000)
                kernel_times_ms.append(kernel_ms)

            end_total = time.perf_counter()

            # =========================
            # POWER STOP
            # =========================
            power_proc.terminate()
            power_proc.wait()

            avg_power = parse_power(POWER_LOG_FILE)

            # =========================
            # METRICS
            # =========================
            total_time = end_total - start_total
            throughput = (N_RUNS * batch) / total_time

            # energy = mean board power x wall-clock time of the timed loop
            energy_j = avg_power * total_time
            energy_per_sample_mj = (energy_j / (N_RUNS * batch)) * 1000

            latencies_ms = np.array(latencies_ms)
            kernel_times_ms = np.array(kernel_times_ms)

            # =========================
            # STORE
            # =========================
            all_results.append({
                "seed": seed,
                "precision": precision,
                "batch": batch,

                "latency_mean_ms": np.mean(latencies_ms),
                "latency_p95_ms": np.percentile(latencies_ms, 95),
                "latency_std_ms": np.std(latencies_ms, ddof=1),
                "latency_ci95_ms": ci95(latencies_ms),

                "kernel_mean_ms": np.mean(kernel_times_ms),

                "throughput": throughput,
                "avg_power_w": avg_power,

                "fps_per_watt": throughput / avg_power if avg_power > 0 else 0,
                "energy_per_sample_mj": energy_per_sample_mj
            })


# =========================
# SAVE RAW RESULTS
# =========================
df = pd.DataFrame(all_results)
df.to_csv("batch_results_raw.csv", index=False)

# =========================
# AGGREGATED RESULTS
# =========================
group_cols = ["precision", "batch"]

metrics = [
    "latency_mean_ms",
    "latency_p95_ms",
    "latency_std_ms",
    "latency_ci95_ms",
    "kernel_mean_ms",
    "throughput",
    "avg_power_w",
    "energy_per_sample_mj",
    "fps_per_watt"
]

df_agg = df.groupby(group_cols)[metrics].agg(["mean", "std"]).reset_index()

# =========================
# FLATTEN MULTIINDEX COLUMNS
# =========================
df_agg.columns = [
    "_".join(col).strip("_") if col[1] != "" else col[0]
    for col in df_agg.columns
]

# =========================
# RENAME: "<metric>_mean" -> "<metric>", std columns keep their suffix
# =========================
df_agg = df_agg.rename(columns={
    "precision_": "precision",
    "batch_": "batch",

    "latency_mean_ms_mean": "latency_mean_ms",
    "latency_mean_ms_std": "latency_mean_ms_std",

    "latency_p95_ms_mean": "latency_p95_ms",
    "latency_p95_ms_std": "latency_p95_ms_std",

    "latency_std_ms_mean": "latency_std_ms",
    "latency_std_ms_std": "latency_std_ms_std",

    "kernel_mean_ms_mean": "kernel_mean_ms",
    "kernel_mean_ms_std": "kernel_mean_ms_std",

    "throughput_mean": "throughput",
    "throughput_std": "throughput_std",

    "avg_power_w_mean": "avg_power_w",
    "avg_power_w_std": "avg_power_w_std",

    "energy_per_sample_mj_mean": "energy_per_sample_mj",
    "energy_per_sample_mj_std": "energy_per_sample_mj_std",

    "fps_per_watt_mean": "fps_per_watt",
    "fps_per_watt_std": "fps_per_watt_std"
})

# =========================
# DERIVED METRICS
# =========================
# samples/s divided by batch size = batched calls per second
df_agg["throughput_per_sample"] = df_agg["throughput"] / df_agg["batch"]

# W / (samples/s) = joules per sample
df_agg["energy_per_throughput"] = (
    df_agg["avg_power_w"] / df_agg["throughput"]
)

# =========================
# SCALING EFFICIENCY
# throughput / (batch-1 throughput x batch size);
# 1.0 = perfectly linear scaling, lower values show saturation
# =========================
baseline_tp = {}

for p in df_agg["precision"].unique():
    baseline_tp[p] = df_agg[
        (df_agg["precision"] == p) &
        (df_agg["batch"] == 1)
    ]["throughput"].values[0]

df_agg["scaling_efficiency"] = df_agg.apply(
    lambda row: row["throughput"] /
    (baseline_tp[row["precision"]] * row["batch"]),
    axis=1
)


# =========================
# LATENCY GROWTH FACTOR
# mean latency relative to batch 1 at the same precision
# =========================
baseline_lat = {}

for p in df_agg["precision"].unique():
    baseline_lat[p] = df_agg[
        (df_agg["precision"] == p) &
        (df_agg["batch"] == 1)
    ]["latency_mean_ms"].values[0]

df_agg["latency_growth_factor"] = df_agg.apply(
    lambda row: row["latency_mean_ms"] / baseline_lat[row["precision"]],
    axis=1
)

# =========================
# SAVE FINAL AGGREGATED CSV
# =========================
df_agg.to_csv("batch_results.csv", index=False)

print("\nSaved:")
print(" - batch_results_raw.csv")
print(" - batch_results.csv")
