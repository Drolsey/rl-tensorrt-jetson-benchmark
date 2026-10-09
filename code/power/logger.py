"""
Appends one row of evaluation results to results.csv, writing the
header the first time the file is created.
"""

import csv
import os
from datetime import datetime

FILE = "results.csv"

HEADERS = [
    "timestamp",
    "env",
    "model",
    "precision",
    "latency_mean_ms",
    "latency_p95_ms",
    "fps",
    "reward_mean",
    "reward_std"
]

def init_log():
    if not os.path.exists(FILE):
        with open(FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)

def log_result(env, model, precision,
               latency_mean, latency_p95,
               fps, reward_mean, reward_std):

    init_log()

    with open(FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            env,
            model,
            precision,
            latency_mean,
            latency_p95,
            fps,
            reward_mean,
            reward_std
        ])
