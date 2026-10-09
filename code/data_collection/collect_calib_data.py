"""
Records CartPole observations visited by the trained policy, for INT8
calibration in build_int8.py.

Output: calib_obs.npy
"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import torch
torch.cuda.empty_cache()

model = PPO.load("ppo_cartpole_fp32", device="cpu")

env = gym.make("CartPole-v1")

obs_list = []

obs, _ = env.reset()

for _ in range(2000):  # ~1000+ states needed to cover the observation range
    action, _ = model.predict(obs, deterministic=True)
    obs_list.append(obs)

    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

obs_array = np.array(obs_list, dtype=np.float32)

np.save("calib_obs.npy", obs_array)

print("Saved calibration data:", obs_array.shape)

