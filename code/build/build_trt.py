"""
Builds fixed-shape FP32 / FP16 / INT8 engines from policy.onnx with
trtexec. The INT8 build reads its scales from an existing calibration
cache (calib.cache) rather than calibrating here.
"""

import os

ONNX = "policy.onnx"

def build(engine, flags):
    cmd = f"""
    trtexec \
      --onnx={ONNX} \
      --saveEngine={engine} \
      {flags} \
      --memPoolSize=workspace:512MiB \
      --avgRuns=100 \
      --useSpinWait
    """
    os.system(cmd)

print("Building FP32...")
build("policy_fp32.trt", "")

print("Building FP16...")
build("policy_fp16.trt", "--fp16")

print("Building INT8...")
build("policy_int8.trt", "--int8 --calib=calib.cache")
