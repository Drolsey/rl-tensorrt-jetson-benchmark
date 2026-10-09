"""
Collects 10,000 CartPole observations by rolling out the trained policy.
The benchmark scripts draw their input batches from this pool, so the
engines are timed on realistic states rather than random noise.

Output: cartpole_states.npy
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# =========================
# CONFIG
# =========================
MODEL_PATH = "ppo_cartpole_fp32.zip"
N_STATES = 10000
SAVE_PATH = "cartpole_states.npy"

# =========================
# LOAD MODEL
# =========================
model = PPO.load(MODEL_PATH)

# =========================
# ENV
# =========================
env = gym.make("CartPole-v1")

states = []

obs, _ = env.reset(seed=0)

# =========================
# COLLECT STATES
# =========================
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
