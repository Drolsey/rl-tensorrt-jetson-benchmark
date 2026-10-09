"""
Exports the full SB3 policy module to policy.onnx (input to
build_trt.py). The dynamic-batch engines are built from the actor-only
export in export_onnx_dynamic.py instead.
"""

import torch
import numpy as np
from stable_baselines3 import PPO

MODEL_PATH = "ppo_cartpole.zip"
ONNX_PATH = "policy.onnx"

model = PPO.load(MODEL_PATH)
model.policy.eval()

dummy_input = torch.randn(1, 4, dtype=torch.float32)

torch.onnx.export(
    model.policy,
    dummy_input,
    ONNX_PATH,
    input_names=["obs"],
    output_names=["logits"],
    opset_version=17,
    dynamic_axes={
        "obs": {0: "batch"},
        "logits": {0: "batch"}
    }
)

print("ONNX exported")
