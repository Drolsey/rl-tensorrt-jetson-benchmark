"""
Policy fidelity of the LunarLander-v3 engines, using the same method
as fidelity_eval.py (CartPole) over 3 seeds x 50 episodes:

  - episode reward from each engine's own rollout
  - action agreement with FP32 on the FP32 rollout's observations
  - reward_degradation_pct = (fp32_reward - engine_reward) / fp32_reward * 100

Output:
    experiments/lunarlander/ll_fidelity_per_episode.csv
    experiments/lunarlander/ll_fidelity_results.csv   (aggregated)
"""

import os
import numpy as np
import pandas as pd
import gymnasium as gym

from policies.trt_batch_policy import TRTBatchPolicy

OUT_DIR = "experiments/lunarlander"

ENGINES = {
    "FP32": os.path.join(OUT_DIR, "ll_fp32.trt"),
    "FP16": os.path.join(OUT_DIR, "ll_fp16.trt"),
    "INT8": os.path.join(OUT_DIR, "ll_int8.trt"),
}

SEEDS = [0, 1, 2]
EPISODES_PER_SEED = 50
OBS_DIM = 8

OUT_PATH = os.path.join(OUT_DIR, "ll_fidelity_results.csv")


def env_seed_for(seed, episode):
    return seed * 100_000 + episode


def run_episode(model, env_seed, replay_obs_actions=None):
    # see run_episode() in fidelity_eval.py for the two modes
    env = gym.make("LunarLander-v3")
    obs, _ = env.reset(seed=env_seed)

    if replay_obs_actions is not None:
        actions_on_fixed_obs = []
        for ref_obs in replay_obs_actions:
            obs_batch = np.asarray(ref_obs, dtype=np.float32).reshape(1, OBS_DIM)
            action = int(np.asarray(model(obs_batch)).reshape(-1)[0])
            actions_on_fixed_obs.append(action)
        env.close()
        return actions_on_fixed_obs

    total_reward = 0.0
    obs_trace = []
    action_trace = []

    terminated = truncated = False
    while not (terminated or truncated):
        obs_trace.append(obs.copy())
        obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, OBS_DIM)
        action = int(np.asarray(model(obs_batch)).reshape(-1)[0])
        action_trace.append(action)

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

    env.close()
    return total_reward, obs_trace, action_trace


def main():
    models = {name: TRTBatchPolicy(path) for name, path in ENGINES.items()}

    rows = []

    for seed in SEEDS:
        print(f"\n===== SEED {seed} =====")

        for episode in range(EPISODES_PER_SEED):
            env_seed = env_seed_for(seed, episode)

            # FP32 rollout is the reference trajectory
            fp32_reward, fp32_obs_trace, fp32_actions = run_episode(
                models["FP32"], env_seed
            )

            rows.append({
                "seed": seed,
                "episode": episode,
                "engine": "FP32",
                "episode_reward": fp32_reward,
                "action_agreement": 1.0,
            })

            for name in ENGINES:
                if name == "FP32":
                    continue

                # reward from the engine's own rollout
                own_reward, _, _ = run_episode(models[name], env_seed)

                # action agreement on the FP32 observation sequence
                replayed_actions = run_episode(
                    models[name], env_seed, replay_obs_actions=fp32_obs_trace
                )
                n = min(len(replayed_actions), len(fp32_actions))
                agreement = (
                    np.mean(np.array(replayed_actions[:n]) == np.array(fp32_actions[:n]))
                    if n > 0
                    else np.nan
                )

                rows.append({
                    "seed": seed,
                    "episode": episode,
                    "engine": name,
                    "episode_reward": own_reward,
                    "action_agreement": agreement,
                })

            if episode % 10 == 0:
                print(f"  episode {episode}/{EPISODES_PER_SEED} done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "ll_fidelity_per_episode.csv"), index=False)

    agg = df.groupby("engine").agg(
        mean_episode_reward=("episode_reward", "mean"),
        std_episode_reward=("episode_reward", "std"),
        mean_action_agreement=("action_agreement", "mean"),
    ).reset_index()

    fp32_reward = agg.loc[agg["engine"] == "FP32", "mean_episode_reward"].values[0]
    agg["reward_degradation_pct"] = (
        (fp32_reward - agg["mean_episode_reward"]) / fp32_reward * 100.0
    )

    agg.to_csv(OUT_PATH, index=False)

    print("\nSaved:")
    print(f" - {os.path.join(OUT_DIR, 'll_fidelity_per_episode.csv')} (raw)")
    print(f" - {OUT_PATH} (aggregated)")
    print(agg)


if __name__ == "__main__":
    main()
