# TensorRT Precision Benchmarks for RL Policies on Jetson

Code, results and figures for benchmarking reinforcement-learning policy
inference on an NVIDIA Jetson Orin with TensorRT at three precisions
(FP32, FP16, INT8), against a PyTorch CPU baseline.

Two Gymnasium environments are used:

| Environment     | Observation size | Actions | Policy                                     |
|-----------------|------------------|---------|--------------------------------------------|
| CartPole-v1     | 4                | 2       | PPO (Stable-Baselines3), trained separately |
| LunarLander-v3  | 8                | 4       | PPO (Stable-Baselines3), `train_lunarlander.py` |

For each environment and precision the experiments measure:

- **Inference performance** across batch sizes 1, 2, 4, 8, 16 and 32:
  mean and P95 latency, GPU kernel time, throughput, board power, energy
  per sample and FPS/W (3 seeds, 200 warm-up + 2000 timed calls each).
- **Policy fidelity**: episode reward of each engine, the share of
  actions that agree with FP32 on identical observations, and reward
  degradation relative to FP32.
- **Statistical significance** of the differences between precisions:
  Welch's t-tests and Cohen's d.
- **Kernel-level profiling** with Nsight Systems.

---

## Contents

```
thesis_package/
├── README.md
├── requirements.txt
├── code/                      all Python scripts (see "Code overview")
├── figures/                   final figures (PNG)
│   └── lunarlander/           per-metric LunarLander figures
├── batch_results_raw.csv      CartPole TensorRT benchmark, one row per seed
├── batch_results.csv          ... mean/std across seeds + derived metrics
├── baseline_results_raw.csv   CartPole PyTorch CPU baseline, one row per seed
├── baseline_results.csv       ... mean/std across seeds
├── final_results.csv          TensorRT + PyTorch merged (CartPole figures)
├── final_results_agg.csv      ... mean/std per model and batch size
├── ll_results_raw.csv         LunarLander TensorRT benchmark, one row per seed
├── ll_results.csv             ... mean/std across seeds + derived metrics
├── fidelity_per_episode.csv   CartPole fidelity, one row per episode
├── fidelity_results.csv       ... aggregated per engine
├── ll_fidelity_per_episode.csv  LunarLander fidelity, one row per episode
├── ll_fidelity_results.csv      ... aggregated per engine
├── welch_ttest_results.csv    Welch's t-tests / Cohen's d (CartPole)
└── nsys_kernel_summary.txt    Nsight Systems GPU trace summary
```

Not included: trained models (`.zip`), ONNX exports, TensorRT engines
(`.trt`), calibration caches and the observation pools (`.npy`). Engines
are specific to the GPU and TensorRT version they were built on, so they
have to be rebuilt on the target device (see "Running the pipeline").

---

## Environment

Measurements were taken on an NVIDIA Jetson Orin. TensorRT, CUDA and
`tegrastats` come from the JetPack installation on the board.

Python 3.10.12 in a virtual environment with:

| Package           | Version |
|-------------------|---------|
| torch             | 2.8.0   |
| stable-baselines3 | 2.8.0   |
| gymnasium         | 1.2.3   |
| numpy             | 1.26.4  |
| pandas            | 2.3.3   |
| scipy             | 1.15.3  |
| matplotlib        | 3.10.8  |
| onnx              | 1.21.0  |
| pycuda            | 2026.1  |

Also needed: `seaborn` (plots), `gymnasium[box2d]` (LunarLander) and,
optionally, `nvtx` (profiling markers). `requirements.txt` lists these.
`tensorrt` itself is taken from JetPack, not pip.

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

---

## Code overview

| Folder | Script | Purpose |
|--------|--------|---------|
| `policies/` | `trt_batch_policy.py` | `TRTBatchPolicy`: loads a dynamic-batch engine and returns greedy actions, or the CUDA-event kernel time. Used by every TensorRT script. |
| `data_collection/` | `collect_cartpole_states.py` | 10,000 CartPole observations from the trained policy (benchmark input pool) |
| | `collect_calib_data.py` | 2,000 observations for INT8 calibration |
| `export/` | `export_onnx_dynamic.py` | Actor network → ONNX with a dynamic batch axis (used for the engines) |
| | `export_onnx.py` | Full SB3 policy → fixed-shape ONNX (early static engines) |
| | `export_policy.py` | Saves actor MLP weights as a PyTorch state dict |
| `build/` | `build_int8.py` | Builds the dynamic-batch INT8 engine (entropy calibrator) |
| | `build_trt.py` | Builds the early fixed-shape engines with `trtexec` |
| | `check_engine.py` | Prints an engine's input name and shape |
| `benchmark/` | `run_full_benchmark.py` | **Main CartPole benchmark** (FP32/FP16/INT8 × 6 batch sizes × 3 seeds) |
| | `baseline_pytorch.py` | PyTorch CPU baseline with the same protocol |
| | `trt_batch_benchmark.py` | Earlier FP16-only sweep (superseded; overwrites `batch_results.csv`) |
| `analysis/` | `merge_results.py` | Merges TensorRT and PyTorch results → `final_results.csv` |
| | `aggregated_results.py` | → `final_results_agg.csv` |
| | `add_derived_metrics.py` | Adds scaling efficiency / latency growth to older result files |
| | `stat_framework.py` | Repeated-run helper with mean/std/95% CI |
| `plotting/` | `plot_final_cart_results.py` | CartPole per-metric figures incl. PyTorch baseline |
| | `plot_full_results.py`, `plot_results.py` | Earlier quick-look plots |
| `power/` | `power_logger.py`, `power_analysis.py`, `logger.py` | Early power/CSV helpers (not used by the final benchmarks, which parse `tegrastats` directly) |
| `experiments_root/` | `fidelity_eval.py` | CartPole policy fidelity (3 seeds × 100 episodes) |
| | `stat_tests.py` | Welch's t-tests and Cohen's d |
| | `nsys_inference.py` | Profiling workload; nsys/ncu commands at the bottom of the file |
| | `plot_clear_overview.py` | 2×3 overview figure per environment |
| | `plot_cross_environment.py` | CartPole vs LunarLander comparison |
| | `plot_fidelity_comparison.py` | Fidelity comparison figure |
| | `plot_stat_significance.py` | Effect-size heatmap |
| | `plot_style.py` | Shared colors/markers per precision |
| `lunarlander/` | `train_lunarlander.py` → `collect_lunarlander_states.py` → `collect_lunarlander_calib.py` → `export_lunarlander_onnx.py` → `build_lunarlander_engines.py` → `benchmark_lunarlander.py` → `fidelity_lunarlander.py` → `plot_lunarlander_results.py` | Full LunarLander pipeline |

---

## Running the pipeline

### Directory layout

The scripts use paths relative to the project root and were run from
it. The folders in `code/` are grouped by purpose for readability; on
the device they are laid out like this:

| In this repository | On the device (project root) |
|--------------------|------------------------------|
| `code/policies/trt_batch_policy.py` | `policies/trt_batch_policy.py` |
| `code/{analysis,benchmark,build,data_collection,export,plotting,power}/*.py` | project root (flat) |
| `code/experiments_root/*.py` | `experiments/` |
| `code/lunarlander/*.py` | `experiments/lunarlander/` |

Run every command from the project root with the root on `PYTHONPATH`,
so `from policies.trt_batch_policy import TRTBatchPolicy` resolves for
scripts inside `experiments/`:

```bash
cd ~/rl-thesis
source env/bin/activate
export PYTHONPATH=$PWD
```

### CartPole

Requires the trained policy `ppo_cartpole_fp32.zip` in the project root
(the CartPole training script is not part of this package).

```bash
# 1. Observation pools
python collect_cartpole_states.py      # -> cartpole_states.npy
python collect_calib_data.py           # -> calib_obs.npy

# 2. ONNX export (dynamic batch)
python export_onnx_dynamic.py          # -> policy_dynamic.onnx

# 3. TensorRT engines (batch 1-32, optimised for 8)
python build_int8.py                   # -> policy_int8_dynamic.trt
```

`policy_fp32_dynamic.trt` and `policy_fp16_dynamic.trt` use the same
`policy_dynamic.onnx` and shape range (min 1×4, opt 8×4, max 32×4),
without and with FP16 enabled. They can be built with `trtexec`:

```bash
trtexec --onnx=policy_dynamic.onnx --saveEngine=policy_fp32_dynamic.trt \
    --minShapes=obs:1x4 --optShapes=obs:8x4 --maxShapes=obs:32x4
trtexec --onnx=policy_dynamic.onnx --saveEngine=policy_fp16_dynamic.trt \
    --minShapes=obs:1x4 --optShapes=obs:8x4 --maxShapes=obs:32x4 --fp16
```

```bash
# 4. Benchmarks
python run_full_benchmark.py           # -> batch_results_raw.csv, batch_results.csv
python baseline_pytorch.py             # -> baseline_results_raw.csv, baseline_results.csv

# 5. Merge and plot
python merge_results.py                # -> final_results.csv
python aggregated_results.py           # -> final_results_agg.csv
python plot_final_cart_results.py      # -> figures_ieee/

# 6. Fidelity and statistics
python experiments/fidelity_eval.py    # -> experiments/fidelity/
python experiments/stat_tests.py       # -> experiments/stats/welch_ttest_results.csv
python experiments/plot_stat_significance.py
```

### LunarLander

```bash
pip install "gymnasium[box2d]"

python experiments/lunarlander/train_lunarlander.py          # PPO, 500k steps
python experiments/lunarlander/collect_lunarlander_states.py
python experiments/lunarlander/collect_lunarlander_calib.py
python experiments/lunarlander/export_lunarlander_onnx.py
python experiments/lunarlander/build_lunarlander_engines.py  # FP32/FP16/INT8 engines
python experiments/lunarlander/benchmark_lunarlander.py      # -> ll_results_raw.csv, ll_results.csv
python experiments/lunarlander/fidelity_lunarlander.py       # -> ll_fidelity_*.csv
python experiments/lunarlander/plot_lunarlander_results.py
```

All LunarLander outputs go to `experiments/lunarlander/`.

### Cross-environment figures and profiling

```bash
python experiments/plot_clear_overview.py
python experiments/plot_cross_environment.py
python experiments/plot_fidelity_comparison.py

# Nsight Systems trace (see the end of nsys_inference.py for all commands)
nsys profile --output=experiments/nsys/timeline --trace=cuda,nvtx \
    --force-overwrite=true python experiments/nsys_inference.py
```

---

## Measurement methodology

- **Inputs.** Each benchmark call uses a batch of real observations
  taken from a pool of 10,000 states visited by the trained policy, not
  random noise.
- **Latency** (`latency_*_ms`) is the wall-clock time of one call
  (`time.perf_counter`): host→device copy, engine execution,
  synchronisation and device→host copy.
- **Kernel time** (`kernel_mean_ms`) is the engine execution alone,
  measured with CUDA events on the inference stream.
- **Throughput** = samples processed / wall-clock duration of the timed
  loop.
- **Power** (`avg_power_w`) is the mean of the `VDD_IN` (total board
  input) readings from `tegrastats --interval 100` during the timed loop.
  It is whole-board power, not GPU-only power.
- **Energy per sample** = mean power × loop duration / samples (mJ).
- **Scaling efficiency** = throughput / (batch-1 throughput × batch size);
  1.0 means throughput grows linearly with batch size.
- **Seeds** change which states are sampled. Values in the aggregated
  CSVs are means across seeds, and `*_std` columns are the spread across
  seeds.
- **Fidelity.** Each engine's reward comes from its own rollout. Action
  agreement is measured by feeding the FP32 rollout's observations to
  FP16/INT8 without letting them act, so all engines are compared on the
  same states.

### CSV columns (benchmark files)

| Column | Unit | Meaning |
|--------|------|---------|
| `seed` | | Sampling seed (raw files only) |
| `precision` | | `FP32`, `FP16`, `INT8` (or `PYTORCH` for the baseline) |
| `batch` | | Batch size |
| `latency_mean_ms`, `latency_p95_ms`, `latency_std_ms`, `latency_ci95_ms` | ms | Per-call wall-clock latency statistics |
| `kernel_mean_ms` | ms | Mean engine execution time (CUDA events) |
| `throughput` | samples/s | |
| `avg_power_w` | W | Mean board input power |
| `fps_per_watt` | samples/s/W | |
| `energy_per_sample_mj` | mJ | |
| `scaling_efficiency`, `latency_growth_factor` | | Relative to batch 1 (aggregated files only) |

---

## Limitations

- Three seeds per configuration, so the t-tests have low statistical
  power; p-values and effect sizes are indicative only.
- Power is whole-board `VDD_IN`, which includes CPU, memory and idle
  draw, not just the GPU.
- The PyTorch baseline runs on the CPU with a randomly initialised
  4-128-2 MLP rather than the trained policy; it is a timing reference
  only.
- Results depend on the Jetson power mode and clock settings in use
  during the runs.
