"""
Workload for Nsight Systems / Nsight Compute profiling.

Runs 500 batch-1 inferences per engine (FP32 / FP16 / INT8). Each
engine's block and each iteration are wrapped in NVTX ranges so the
profiler output can be split by engine. If the `nvtx` package isn't
installed the same loop runs without markers.

Standalone:
    python experiments/nsys_inference.py

The nsys / ncu commands used to profile it are listed at the bottom of
this file.
"""

import time
import numpy as np

from policies.trt_batch_policy import TRTBatchPolicy

try:
    import nvtx
    HAVE_NVTX = True
except ImportError:
    HAVE_NVTX = False
    print("[WARN] nvtx not installed - running without NVTX markers. "
          "Install with: pip install nvtx --break-system-packages")

ENGINES = {
    "FP32": "policy_fp32_dynamic.trt",
    "FP16": "policy_fp16_dynamic.trt",
    "INT8": "policy_int8_dynamic.trt",
}

N_ITERS = 500
BATCH = 1


def nvtx_range(name):
    if HAVE_NVTX:
        return nvtx.annotate(name)
    # no-op context manager fallback
    class _NoOp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    return _NoOp()


def main():
    obs = np.random.randn(BATCH, 4).astype(np.float32)

    for name, path in ENGINES.items():
        print(f"\n=== Profiling {name} ===")
        model = TRTBatchPolicy(path)

        # warm-up, outside the NVTX ranges
        for _ in range(50):
            _ = model(obs)

        with nvtx_range(f"engine_{name}"):
            t0 = time.perf_counter()
            for i in range(N_ITERS):
                with nvtx_range(f"{name}_iter_{i}"):
                    _ = model(obs)
            t1 = time.perf_counter()

        total_ms = (t1 - t0) * 1000
        print(f"{name}: {N_ITERS} iters, {total_ms:.2f} ms total, "
              f"{total_ms / N_ITERS:.4f} ms/iter")

    print("\nDone. If run under nsys/ncu, inspect the resulting "
          ".nsys-rep / .ncu-rep files.")


if __name__ == "__main__":
    main()


# =========================================================
# PROFILER COMMANDS (run from the project root)
# =========================================================
#
# 1) nsys timeline trace:
#
#   nsys profile \
#     --output=experiments/nsys/timeline \
#     --trace=cuda,nvtx \
#     --force-overwrite=true \
#     python experiments/nsys_inference.py
#
# 2) Extract text summary from the .nsys-rep:
#
#   nsys stats experiments/nsys/timeline.nsys-rep \
#     --report gputrace > experiments/nsys/nsys_summary.txt 2>&1
#
# 3) ncu kernel-level metrics (SM occupancy, warp utilization, mem bandwidth):
#
#   ncu --target-processes all \
#     --metrics sm__occupancy.pct_of_peak_sustained_elapsed,\
# sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
# l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum \
#     --csv \
#     --output experiments/nsys/kernel_metrics \
#     python experiments/nsys_inference.py > experiments/nsys/ncu_summary.csv 2>&1
#
# 4) Extract readable text report from the .ncu-rep:
#
#   ncu --import experiments/nsys/kernel_metrics.ncu-rep \
#     --page details > experiments/nsys/ncu_readable_report.txt 2>&1
#
# Steps 3-4 need the full Nsight Compute toolkit, which is not always
# installed on the Jetson alongside nsys. Without it, per-kernel timing
# is still available from the kernel_mean_ms column of
# batch_results_raw.csv, but not occupancy / warp-utilization metrics.
