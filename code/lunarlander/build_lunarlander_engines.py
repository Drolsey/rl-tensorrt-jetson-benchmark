"""
Builds FP32 / FP16 / INT8 TensorRT engines for the LunarLander policy
with the TensorRT Python API. Same settings as build_int8.py: dynamic
batch 1-32, tuned for batch 8, INT8 calibrated with the entropy
calibrator.

Run first:
    python experiments/lunarlander/export_lunarlander_onnx.py
    python experiments/lunarlander/collect_lunarlander_calib.py

Output:
    experiments/lunarlander/ll_fp32.trt
    experiments/lunarlander/ll_fp16.trt
    experiments/lunarlander/ll_int8.trt
"""

import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  (initializes CUDA context)

OUT_DIR = "experiments/lunarlander"
ONNX_PATH = os.path.join(OUT_DIR, "lunarlander_dynamic.onnx")
CALIB_DATA_PATH = os.path.join(OUT_DIR, "ll_calib_obs.npy")

INPUT_NAME = "obs"
OBS_DIM = 8

BATCH_OPT = 8
MAX_BATCH = 32

logger = trt.Logger(trt.Logger.INFO)


# =========================
# CALIBRATOR (INT8)
# =========================
class NumpyCalibrator(trt.IInt8EntropyCalibrator2):
    """Same calibrator as build_int8.py, with the cache path passed in."""

    def __init__(self, calib_data, cache_file):
        super().__init__()
        self.data = calib_data.astype(np.float32)
        self.batch_size = 8
        self.index = 0
        self.device_input = cuda.mem_alloc(self.data[0:self.batch_size].nbytes)
        self.cache_file = cache_file

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.index + self.batch_size > len(self.data):
            return None
        batch = self.data[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        cuda.memcpy_htod(self.device_input, batch)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_engine(engine_path, precision):
    """precision: 'fp32', 'fp16' or 'int8'."""
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        print("Loading LunarLander calibration data...")
        calib_data = np.load(CALIB_DATA_PATH)
        calibrator = NumpyCalibrator(
            calib_data, cache_file=os.path.join(OUT_DIR, "ll_int8_calib.cache")
        )
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator

    profile = builder.create_optimization_profile()
    profile.set_shape(
        INPUT_NAME, (1, OBS_DIM), (BATCH_OPT, OBS_DIM), (MAX_BATCH, OBS_DIM)
    )
    config.add_optimization_profile(profile)

    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")

    print(f"Building {precision.upper()} engine...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError(f"{precision.upper()} engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"Saved {precision.upper()} engine -> {engine_path}")


if __name__ == "__main__":
    build_engine(os.path.join(OUT_DIR, "ll_fp32.trt"), "fp32")
    build_engine(os.path.join(OUT_DIR, "ll_fp16.trt"), "fp16")
    build_engine(os.path.join(OUT_DIR, "ll_int8.trt"), "int8")
