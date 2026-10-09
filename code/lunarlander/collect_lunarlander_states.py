"""
Collects 10,000 LunarLander-v3 observations from the trained policy,
used as the input pool for benchmark_lunarlander.py (the LunarLander
counterpart of collect_cartpole_states.py).

Run before benchmark_lunarlander.py.

Output: experiments/lunarlander/lunarlander_states.npy
"""

import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

OUT_DIR = "experiments/lunarlander"
MODEL_PATH = os.path.join(OUT_DIR, "ppo_lunarlander.zip")
SAVE_PATH = os.path.join(OUT_DIR, "lunarlander_states.npy")
N_STATES = 10000

model = PPO.load(MODEL_PATH)
env = gym.make("LunarLander-v3")

states = []
obs, _ = env.reset(seed=0)

while len(states) < N_STATES:
    states.append(obs.copy())

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()

states = np.array(states, dtype=np.float32)
np.save(SAVE_PATH, states)

print(f"Saved {len(states)} states -> {SAVE_PATH}")
print("Shape:", states.shape)
