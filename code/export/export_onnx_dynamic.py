"""
Exports the actor path of the trained CartPole PPO model (policy MLP +
action head) to ONNX with a dynamic batch axis. The value head is left
out since it isn't used at inference time.

Output: policy_dynamic.onnx
"""

import torch
import torch.nn as nn
from stable_baselines3 import PPO

# -------------------------
# LOAD MODEL
# -------------------------
model = PPO.load("ppo_cartpole_fp32.zip")

# -------------------------
# ACTOR NETWORK ONLY
# -------------------------
policy_net = model.policy.mlp_extractor.policy_net
action_net = model.policy.action_net

# -------------------------
# WRAPPER: obs -> action logits (argmax gives the greedy action)
# -------------------------
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
net.eval()

# -------------------------
# EXPORT (dynamic batch axis)
# -------------------------
dummy_input = torch.randn(1, 4)

torch.onnx.export(
    net,
    dummy_input,
    "policy_dynamic.onnx",
    input_names=["obs"],
    output_names=["logits"],
    opset_version=17,
    dynamic_axes={
        "obs": {0: "batch_size"},
        "logits": {0: "batch_size"}
    }
)

print("Dynamic ONNX exported successfully")
