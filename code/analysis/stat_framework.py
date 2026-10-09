import numpy as np


def run_multi(run_fn, model, label, runs=5, episodes=10):
    """
    Calls run_fn(model, label, episodes=...) `runs` times and summarises
    the per-run mean latency and mean reward (mean, std and 95% CI
    across runs). The raw per-run values are returned as well.
    """

    run_latencies = []
    run_rewards = []

    print(f"\n================ {label} ================\n")

    for r in range(runs):

        print(f"--- Run {r+1}/{runs} ---")

        result = run_fn(model, label, episodes=episodes)

        run_latencies.append(result["latency_mean_ms"])
        run_rewards.append(result["reward_mean"])

    import numpy as np

    # half-width of a normal-approximation 95% confidence interval
    def ci95(x):
        x = np.array(x)
        return 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))

    return {
        "model": label,

        "latency_mean_ms": np.mean(run_latencies),
        "latency_std_ms": np.std(run_latencies, ddof=1),
        "latency_ci95_ms": ci95(run_latencies),

        "reward_mean": np.mean(run_rewards),
        "reward_std": np.std(run_rewards, ddof=1),

        "raw_latency_runs": run_latencies,
        "raw_reward_runs": run_rewards
    }
