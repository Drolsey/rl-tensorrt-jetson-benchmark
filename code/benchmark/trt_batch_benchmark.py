"""
Earlier single-engine (FP16) batch sweep on random inputs, batch 1-8.
Superseded by run_full_benchmark.py.

Note: it also writes batch_results.csv, so running it overwrites the
output of run_full_benchmark.py.
"""

import time
import numpy as np
import pandas as pd
import subprocess
from policies.trt_batch_policy import TRTBatchPolicy

# =========================
# CONFIG
# =========================
BATCH_SIZES = [1, 2, 4, 8]
N_RUNS = 2000
POWER_LOG_FILE = "power_log.txt"

model = TRTBatchPolicy("policy_fp16_dynamic.trt")

results = []


# =========================
# POWER LOGGER (tegrastats)
# =========================
def start_power_logger(output_file):
    # logs power + GPU stats
    cmd = ["tegrastats", "--interval", "100"]
    return subprocess.Popen(cmd, stdout=open(output_file, "w"))


def compute_power_metrics(file_path):
    powers = []

    with open(file_path, "r") as f:
        for line in f:
            if "VDD_IN" in line:
                try:
                    # line contains e.g. "VDD_IN 3886mW/3886mW"
                    # (current/average); keep the current value
                    parts = line.split("VDD_IN")[-1].strip()

                    mw = parts.split("mW")[0].strip()
                    mw = mw.split("/")[-1].strip()

                    power_w = float(mw) / 1000.0
                    powers.append(power_w)

                except:
                    continue

    if len(powers) == 0:
        print("[WARNING] No power data parsed from tegrastats")
        return 0.0, 0.0

    return np.mean(powers), np.std(powers)

# =========================
# BENCHMARK LOOP
# =========================
print("\n=== BATCH BENCHMARK + POWER PROFILING ===\n")

for b in BATCH_SIZES:

    print(f"\n--- Batch {b} ---")

    obs = np.random.randn(b, 4).astype(np.float32)

    # warmup
    for _ in range(200):
        _ = model(obs)

    latencies = []

    # =========================
    # START POWER LOGGING
    # =========================
    power_proc = start_power_logger(POWER_LOG_FILE)
    time.sleep(1)  # give tegrastats time to start sampling

    start_total = time.perf_counter()

    for _ in range(N_RUNS):
        start = time.perf_counter()
        _ = model(obs)
        end = time.perf_counter()

        latencies.append((end - start) * 1000)

    end_total = time.perf_counter()

    # stop power logging
    power_proc.terminate()
    power_proc.wait()

    latencies = np.array(latencies)

    total_time = end_total - start_total
    throughput = (N_RUNS * b) / total_time

    # =========================
    # POWER METRICS
    # =========================
    avg_power, power_std = compute_power_metrics(POWER_LOG_FILE)

    fps_per_watt = throughput / avg_power if avg_power > 0 else 0
    energy_per_sample = (avg_power * total_time) / (N_RUNS * b)

    # =========================
    # PRINT RESULTS
    # =========================
    print(f"Mean latency: {np.mean(latencies):.4f} ms")
    print(f"P95 latency: {np.percentile(latencies, 95):.4f} ms")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Avg Power: {avg_power:.2f} W")
    print(f"FPS/W: {fps_per_watt:.2f}")
    print(f"Energy/sample: {energy_per_sample*1000:.6f} mJ")

    # =========================
    # STORE RESULTS
    # =========================
    results.append({
        "batch": b,
        "latency_mean_ms": np.mean(latencies),
        "latency_p95_ms": np.percentile(latencies, 95),
        "throughput": throughput,
        "avg_power_w": avg_power,
        "fps_per_watt": fps_per_watt,
        "energy_per_sample_mj": energy_per_sample * 1000
    })

# =========================
# SAVE RESULTS
# =========================
np.save("batch_results.npy", results)

df = pd.DataFrame(results)
df.to_csv("batch_results.csv", index=False)

print("\nSaved -> batch_results.npy + batch_results.csv")
