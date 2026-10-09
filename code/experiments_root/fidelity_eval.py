"""
Policy fidelity of the CartPole-v1 TensorRT engines (FP32 / FP16 / INT8),
over 3 seeds x 100 episodes.

Per engine:
  - episode reward, from a rollout in which that engine controls the env;
  - action agreement with FP32: the FP32 rollout's observations are fed
    to the engine without letting it act, so both are compared on the
    same sequence of states;
  - reward_degradation_pct = (fp32_reward - engine_reward) / fp32_reward * 100

Output:
    experiments/fidelity/fidelity_per_episode.csv
    experiments/fidelity/fidelity_results.csv   (aggregated)
"""

import os
import numpy as np
import pandas as pd
import gymnasium as gym

from policies.trt_batch_policy import TRTBatchPolicy

# =========================
# CONFIG
# =========================
ENGINES = {
    "FP32": "policy_fp32_dynamic.trt",
    "FP16": "policy_fp16_dynamic.trt",
    "INT8": "policy_int8_dynamic.trt",
}

SEEDS = [0, 1, 2]
EPISODES_PER_SEED = 100
OUT_DIR = "experiments/fidelity"
OUT_PATH = os.path.join(OUT_DIR, "fidelity_results.csv")

os.makedirs(OUT_DIR, exist_ok=True)


def env_seed_for(seed, episode):
    # reproducible env seed, distinct for every (seed, episode) pair
    return seed * 100_000 + episode


def run_episode(model, env_seed, record_trace=False, replay_obs_actions=None):
    """
    Runs one episode with `model` choosing the actions and returns
    (total_reward, obs_trace, action_trace).

    If replay_obs_actions (a list of observations from a reference
    rollout) is given, the env is not stepped; the model's action for
    each of those observations is returned as a list instead. This is
    what the action-agreement score is computed from.
    """
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=env_seed)

    if replay_obs_actions is not None:
        actions_on_fixed_obs = []
        for ref_obs in replay_obs_actions:
            obs_batch = np.asarray(ref_obs, dtype=np.float32).reshape(1, 4)
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
        obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, 4)
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

            # ---- FP32 drives the reference trajectory ----
            fp32_reward, fp32_obs_trace, fp32_actions = run_episode(
                models["FP32"], env_seed
            )

            episode_record = {
                "FP32": {"reward": fp32_reward, "actions": fp32_actions}
            }

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
                    np.mean(
                        np.array(replayed_actions[:n]) == np.array(fp32_actions[:n])
                    )
                    if n > 0
                    else np.nan
                )

                episode_record[name] = {
                    "reward": own_reward,
                    "action_agreement": agreement,
                }

            for name in ENGINES:
                if name == "FP32":
                    rows.append({
                        "seed": seed,
                        "episode": episode,
                        "engine": "FP32",
                        "episode_reward": fp32_reward,
                        "action_agreement": 1.0,
                    })
                else:
                    rows.append({
                        "seed": seed,
                        "episode": episode,
                        "engine": name,
                        "episode_reward": episode_record[name]["reward"],
                        "action_agreement": episode_record[name]["action_agreement"],
                    })

            if episode % 20 == 0:
                print(f"  episode {episode}/{EPISODES_PER_SEED} done")

    df = pd.DataFrame(rows)
    df.to_csv("experiments/fidelity/fidelity_per_episode.csv", index=False)

    # =========================
    # AGGREGATE
    # =========================
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
    print(f" - experiments/fidelity/fidelity_per_episode.csv (raw, per-episode)")
    print(f" - {OUT_PATH} (aggregated)")
    print(agg)


if __name__ == "__main__":
    main()
