"""
Trains PPO (Stable-Baselines3, default MlpPolicy) on LunarLander-v3 for
500k timesteps, printing progress every 50k.

Output: experiments/lunarlander/ppo_lunarlander.zip

LunarLander needs Box2D:
    pip install "gymnasium[box2d]" --break-system-packages
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

OUT_DIR = "experiments/lunarlander"
os.makedirs(OUT_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 500_000
PRINT_EVERY = 50_000


class ProgressCallback(BaseCallback):
    def __init__(self, print_every=PRINT_EVERY, verbose=0):
        super().__init__(verbose)
        self.print_every = print_every
        self._next_print = print_every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_print:
            print(f"[progress] {self.num_timesteps} / {TOTAL_TIMESTEPS} timesteps")
            self._next_print += self.print_every
        return True


def main():
    env = gym.make("LunarLander-v3")

    model = PPO("MlpPolicy", env, verbose=1, device="cuda")

    print(f"Training PPO on LunarLander-v3 for {TOTAL_TIMESTEPS} timesteps "
          f"(this can take 30-60 min on Jetson CPU)...")

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=ProgressCallback())

    save_path = os.path.join(OUT_DIR, "ppo_lunarlander.zip")
    model.save(save_path)

    print(f"\nSaved trained model -> {save_path}")


if __name__ == "__main__":
    main()
