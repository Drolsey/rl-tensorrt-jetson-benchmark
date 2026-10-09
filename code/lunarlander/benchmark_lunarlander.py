"""
Batch benchmark for the LunarLander-v3 engines (FP32 / FP16 / INT8).

Same procedure as run_full_benchmark.py (3 seeds x 6 batch sizes,
200 warm-up + 2000 timed calls, tegrastats power), with batches of
8-dimensional observations drawn from lunarlander_states.npy.

Output:
    experiments/lunarlander/ll_results_raw.csv
    experiments/lunarlander/ll_results.csv
"""

import os
import time
import numpy as np
import pandas as pd
import subprocess

from policies.trt_batch_policy import TRTBatchPolicy

# =========================
# CONFIG
# =========================
OUT_DIR = "experiments/lunarlander"
states = np.load(os.path.join(OUT_DIR, "lunarlander_states.npy"))

BATCH_SIZES = [1, 2, 4, 8, 16, 32]
N_RUNS = 2000
WARMUP_RUNS = 200
SEEDS = [0, 1, 2]

MODELS = {
    "FP32": os.path.join(OUT_DIR, "ll_fp32.trt"),
    "FP16": os.path.join(OUT_DIR, "ll_fp16.trt"),
    "INT8": os.path.join(OUT_DIR, "ll_int8.trt"),
}

POWER_LOG_FILE = os.path.join(OUT_DIR, "ll_power_log.txt")
os.makedirs("logs", exist_ok=True)


# =========================
# UTIL
# =========================
def ci95(x):
    return 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))


def parse_power(file):
    # mean VDD_IN (board input) power in W; uses the current reading
    # from each "<current>mW/<average>mW" pair
    powers = []
    with open(file, "r") as f:
        for line in f:
            if "VDD_IN" in line:
                try:
                    parts = line.split("VDD_IN")[-1]
                    mw = parts.split("mW")[0].split("/")[-1].strip()
                    powers.append(float(mw) / 1000.0)
                except Exception:
                    continue
    return np.mean(powers) if len(powers) > 0 else 0.0


def start_power_logger():
    return subprocess.Popen(
        ["tegrastats", "--interval", "100"],
        stdout=open(POWER_LOG_FILE, "w"),
        stderr=subprocess.DEVNULL,
    )


# =========================
# MAIN BENCHMARK
# =========================
all_results = []

print("\n=== LUNARLANDER BENCHMARK (FP32 vs FP16 vs INT8) ===\n")

for seed in SEEDS:
    np.random.seed(seed)
    print(f"\n===== SEED {seed} =====\n")

    for precision, engine_path in MODELS.items():
        print(f"\n================ {precision} ================\n")
        model = TRTBatchPolicy(engine_path)

        for batch in BATCH_SIZES:
            print(f"\n--- Batch {batch} ---")

            # warm-up on one fixed batch (not timed)
            idxs = np.random.randint(0, len(states), size=batch)
            obs = states[idxs]

            for _ in range(WARMUP_RUNS):
                _ = model(obs)

            time.sleep(1)

            latencies_ms = []
            kernel_times_ms = []

            power_proc = start_power_logger()
            time.sleep(1)  # give tegrastats time to start sampling

            start_total = time.perf_counter()

            # timed loop: wall-clock latency per call plus engine-only
            # kernel time, on a random contiguous slice of the state pool
            for _ in range(N_RUNS):
                idx = np.random.randint(0, len(states) - batch)
                obs = states[idx:idx + batch]

                t0 = time.perf_counter()
                kernel_ms = model(obs, return_kernel_time=True)
                t1 = time.perf_counter()

                latencies_ms.append((t1 - t0) * 1000)
                kernel_times_ms.append(kernel_ms)

            end_total = time.perf_counter()

            power_proc.terminate()
            power_proc.wait()

            avg_power = parse_power(POWER_LOG_FILE)

            total_time = end_total - start_total
            throughput = (N_RUNS * batch) / total_time

            energy_j = avg_power * total_time
            energy_per_sample_mj = (energy_j / (N_RUNS * batch)) * 1000

            latencies_ms = np.array(latencies_ms)
            kernel_times_ms = np.array(kernel_times_ms)

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
                "energy_per_sample_mj": energy_per_sample_mj,
            })

# =========================
# SAVE RAW RESULTS
# =========================
df = pd.DataFrame(all_results)
df.to_csv(os.path.join(OUT_DIR, "ll_results_raw.csv"), index=False)

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
    "fps_per_watt",
]

df_agg = df.groupby(group_cols)[metrics].agg(["mean", "std"]).reset_index()

df_agg.columns = [
    "_".join(col).strip("_") if col[1] != "" else col[0]
    for col in df_agg.columns
]

df_agg = df_agg.rename(columns={
    "precision_": "precision",
    "batch_": "batch",
    "latency_mean_ms_mean": "latency_mean_ms",
    "latency_p95_ms_mean": "latency_p95_ms",
    "latency_std_ms_mean": "latency_std_ms",
    "kernel_mean_ms_mean": "kernel_mean_ms",
    "throughput_mean": "throughput",
    "avg_power_w_mean": "avg_power_w",
    "energy_per_sample_mj_mean": "energy_per_sample_mj",
    "fps_per_watt_mean": "fps_per_watt",
})

# =========================
# DERIVED METRICS (same definitions as run_full_benchmark.py)
# =========================
df_agg["throughput_per_sample"] = df_agg["throughput"] / df_agg["batch"]
df_agg["energy_per_throughput"] = df_agg["avg_power_w"] / df_agg["throughput"]

baseline_tp = {}
for p in df_agg["precision"].unique():
    baseline_tp[p] = df_agg[
        (df_agg["precision"] == p) & (df_agg["batch"] == 1)
    ]["throughput"].values[0]

df_agg["scaling_efficiency"] = df_agg.apply(
    lambda row: row["throughput"] / (baseline_tp[row["precision"]] * row["batch"]),
    axis=1,
)

baseline_lat = {}
for p in df_agg["precision"].unique():
    baseline_lat[p] = df_agg[
        (df_agg["precision"] == p) & (df_agg["batch"] == 1)
    ]["latency_mean_ms"].values[0]

df_agg["latency_growth_factor"] = df_agg.apply(
    lambda row: row["latency_mean_ms"] / baseline_lat[row["precision"]],
    axis=1,
)

df_agg.to_csv(os.path.join(OUT_DIR, "ll_results.csv"), index=False)

print("\nSaved:")
print(f" - {os.path.join(OUT_DIR, 'll_results_raw.csv')}")
print(f" - {os.path.join(OUT_DIR, 'll_results.csv')}")
