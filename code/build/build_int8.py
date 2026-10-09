"""
Builds the dynamic-batch INT8 TensorRT engine for CartPole from
policy_dynamic.onnx (see export_onnx_dynamic.py), calibrated on the
observations in calib_obs.npy (see collect_calib_data.py).

Output: policy_int8_dynamic.trt
"""

import tensorrt as trt
import numpy as np
import os

import pycuda.driver as cuda
import pycuda.autoinit  # creates the CUDA context


# =========================
# CONFIG
# =========================
ONNX_PATH = "policy_dynamic.onnx"
ENGINE_PATH = "policy_int8_dynamic.trt"
CALIB_DATA_PATH = "calib_obs.npy"

INPUT_NAME = "obs"

BATCH_OPT = 8
MAX_BATCH = 32


# =========================
# LOGGER
# =========================
logger = trt.Logger(trt.Logger.INFO)


# =========================
# CALIBRATOR
# =========================
class NumpyCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Feeds the calibration observations to TensorRT in batches of 8.

    The resulting scales are cached in int8_calib.cache and reused on
    later builds; delete that file to force a fresh calibration.
    """

    def __init__(self, calib_data):
        super().__init__()

        self.data = calib_data.astype(np.float32)
        self.batch_size = 8
        self.index = 0

        self.device_input = cuda.mem_alloc(self.data[0:self.batch_size].nbytes)

        self.cache_file = "int8_calib.cache"

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):

        # returning None tells TensorRT the calibration data is used up
        # (a final partial batch is skipped)
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


# =========================
# BUILD ENGINE
# =========================
def build_int8_engine():

    print("Loading calibration data...")
    calib_data = np.load(CALIB_DATA_PATH)

    calibrator = NumpyCalibrator(calib_data)

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GiB

    # -------------------------
    # INT8 ENABLE
    # -------------------------
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator

    # -------------------------
    # OPTIMIZATION PROFILE
    # accepts batch 1..MAX_BATCH, kernels tuned for BATCH_OPT
    # -------------------------
    profile = builder.create_optimization_profile()

    profile.set_shape(INPUT_NAME, (1, 4), (BATCH_OPT, 4), (MAX_BATCH, 4))
    config.add_optimization_profile(profile)

    # -------------------------
    # LOAD ONNX
    # -------------------------
    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")

    print("Building INT8 engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    with open(ENGINE_PATH, "wb") as f:
        f.write(serialized_engine)

    print(f"Saved INT8 engine -> {ENGINE_PATH}")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    build_int8_engine()
