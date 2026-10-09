"""
PyTorch (CPU) baseline for the CartPole batch benchmark.

Uses the same state pool, batch sizes, seeds, run counts and tegrastats
power logging as run_full_benchmark.py, so the results can be merged
with the TensorRT numbers (see merge_results.py).

Output:
    baseline_results_raw.csv   one row per (seed, batch)
    baseline_results.csv       mean/std across seeds
"""

import time
import numpy as np
import pandas as pd
import subprocess
import os
import torch
import torch.nn as nn

# =========================
# CONFIG
# =========================
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
N_RUNS = 2000
WARMUP_RUNS = 200
SEEDS = [0, 1, 2]

POWER_LOG_FILE = "power_log_pytorch.txt"
os.makedirs("logs", exist_ok=True)
states = np.load("cartpole_states.npy")

# =========================
# POLICY NETWORK
# Randomly initialised MLP standing in for the policy. Only timing and
# power are measured here, so the trained weights aren't needed.
# =========================
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# POWER LOGGER (Jetson tegrastats)
# =========================
def start_power_logger():
    return subprocess.Popen(
        ["tegrastats", "--interval", "100"],
        stdout=open(POWER_LOG_FILE, "w"),
        stderr=subprocess.DEVNULL
    )


def parse_power(file):
    """
    Mean board input power (W) from a tegrastats log. Each line reports
    VDD_IN as <current>mW/<average>mW; the current value is used.
    """
    powers = []

    try:
        with open(file, "r") as f:
            for line in f:
                if "VDD_IN" in line:
                    try:
                        parts = line.split("VDD_IN")[-1]
                        mw = parts.split("mW")[0].split("/")[-1].strip()
                        powers.append(float(mw) / 1000.0)
                    except:
                        continue
    except:
        return 0.0

    return np.mean(powers) if len(powers) > 0 else 0.0


# =========================
# STATISTICS
# =========================
def ci95(x):
    return 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))


# =========================
# MAIN BENCHMARK
# =========================
results = []

device = torch.device("cpu")

print("\n=== PYTORCH BASELINE BENCHMARK ===\n")
print("Device:", device)


for seed in SEEDS:
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n===== SEED {seed} =====\n")

    model = PolicyNet().to(device).eval()

    for batch in BATCH_SIZES:

        print(f"\n--- Batch {batch} ---")

        # =========================
        # WARMUP
        # =========================
        with torch.no_grad():
            for _ in range(WARMUP_RUNS):
                idxs = np.random.randint(0, len(states), size=batch)
                obs = torch.tensor(states[idxs], dtype=torch.float32, device=device)
                _ = model(obs)

        time.sleep(1)  # short idle gap before the measured run

        latencies = []

        # =========================
        # POWER LOGGER START
        # =========================
        power_proc = start_power_logger()
        time.sleep(1)  # give tegrastats time to start sampling

        # latency covers the forward pass only; building the input
        # tensor happens outside the timed region
        start_total = time.perf_counter()

        with torch.no_grad():
            for _ in range(N_RUNS):

                idxs = np.random.randint(0, len(states), size=batch)
                obs = torch.tensor(states[idxs], dtype=torch.float32, device=device)
        
                t0 = time.perf_counter()
                _ = model(obs)
                t1 = time.perf_counter()

                latencies.append((t1 - t0) * 1000)
        
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

        latencies = np.array(latencies)

        print(f"Mean latency: {np.mean(latencies):.4f} ms")
        print(f"P95 latency: {np.percentile(latencies, 95):.4f} ms")
        print(f"Throughput: {throughput:.2f} samples/sec")
        print(f"Power: {avg_power:.2f} W")
        print(f"Energy/sample: {energy_per_sample_mj:.6f} mJ")

        # =========================
        # STORE RESULTS
        # =========================
        results.append({
            "seed": seed,
            "precision": "PYTORCH",
            "batch": batch,

            "latency_mean_ms": np.mean(latencies),
            "latency_p95_ms": np.percentile(latencies, 95),
            "latency_std_ms": np.std(latencies, ddof=1),
            "latency_ci95_ms": ci95(latencies),

            "throughput": throughput,
            "avg_power_w": avg_power,

            "fps_per_watt": throughput / avg_power if avg_power > 0 else 0,
            "energy_per_sample_mj": energy_per_sample_mj
        })


# =========================
# SAVE RESULTS
# =========================
df = pd.DataFrame(results)
df.to_csv("baseline_results_raw.csv", index=False)

df_agg = df.groupby(["precision", "batch"]).agg({
    "latency_mean_ms": ["mean", "std"],
    "throughput": ["mean", "std"],
    "avg_power_w": ["mean", "std"],
    "energy_per_sample_mj": ["mean", "std"]
}).reset_index()

df_agg.to_csv("baseline_results.csv", index=False)

print("\nSaved:")
print(" - baseline_results_raw.csv")
print(" - baseline_results.csv")
