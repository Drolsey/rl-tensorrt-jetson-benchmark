"""
Small helpers for turning a power log (CSV with a power_w column) into
energy and efficiency figures.
"""

import numpy as np
import pandas as pd

def load_power(file="power_log.csv"):
    df = pd.read_csv(file)
    return df


def compute_energy(df, total_time_s):
    # energy (J) = mean power (W) x duration (s)
    avg_power = df["power_w"].mean()
    energy_j = avg_power * total_time_s
    return avg_power, energy_j


def fps_per_watt(fps, avg_power):
    return fps / avg_power


def energy_per_inference(energy_j, num_inferences):
    return energy_j / num_inferences
