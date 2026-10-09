"""
Exports the actor network of the trained LunarLander PPO model to ONNX
with a dynamic batch axis (8 inputs -> 4 action logits). Same approach
as export_onnx_dynamic.py for CartPole.

Output: experiments/lunarlander/lunarlander_dynamic.onnx
"""

import os
import torch
import torch.nn as nn
from stable_baselines3 import PPO

OUT_DIR = "experiments/lunarlander"
MODEL_PATH = os.path.join(OUT_DIR, "ppo_lunarlander.zip")
ONNX_PATH = os.path.join(OUT_DIR, "lunarlander_dynamic.onnx")

# -------------------------
# LOAD MODEL
# -------------------------
model = PPO.load(MODEL_PATH)

# -------------------------
# ACTOR NETWORK ONLY (value head not needed)
# -------------------------
policy_net = model.policy.mlp_extractor.policy_net
action_net = model.policy.action_net


class PolicyWrapper(nn.Module):
    def __init__(self, policy_net, action_net):
        super().__init__()
        self.policy_net = policy_net
        self.action_net = action_net

    def forward(self, x):
        x = self.policy_net(x)
        logits = self.action_net(x)
        return logits


net = PolicyWrapper(policy_net, action_net)
# the model was trained on CUDA; move it to CPU to match the CPU dummy
# input, otherwise tracing fails with a cpu/cuda:0 device mismatch
net = net.cpu()
net.eval()

# -------------------------
# EXPORT (DYNAMIC BATCH, obs dim 8, action dim 4)
# -------------------------
dummy_input = torch.randn(1, 8)

torch.onnx.export(
    net,
    dummy_input,
    ONNX_PATH,
    input_names=["obs"],
    output_names=["logits"],
    opset_version=17,
    dynamic_axes={
        "obs": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
    # legacy exporter keeps the weights inside the .onnx file; the newer
    # one writes a separate .onnx.data file, which the TensorRT parser
    # couldn't find when run from a different directory
    dynamo=False,
)

print(f"Dynamic LunarLander ONNX exported -> {ONNX_PATH}")
