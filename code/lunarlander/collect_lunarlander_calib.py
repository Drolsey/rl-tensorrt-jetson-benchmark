"""
Records 2000 LunarLander observations from the trained policy for INT8
calibration (the LunarLander counterpart of collect_calib_data.py).

Run before build_lunarlander_engines.py.

Output: experiments/lunarlander/ll_calib_obs.npy
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

OUT_DIR = "experiments/lunarlander"
MODEL_PATH = os.path.join(OUT_DIR, "ppo_lunarlander.zip")
SAVE_PATH = os.path.join(OUT_DIR, "ll_calib_obs.npy")

model = PPO.load(MODEL_PATH, device="cpu")
env = gym.make("LunarLander-v3")

obs_list = []
obs, _ = env.reset()

for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs_list.append(obs)

    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

obs_array = np.array(obs_list, dtype=np.float32)
np.save(SAVE_PATH, obs_array)

print("Saved LunarLander calibration data:", obs_array.shape, "->", SAVE_PATH)
