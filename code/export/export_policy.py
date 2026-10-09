"""
Saves the weights of the trained actor MLP (without the action head)
to policy_net.pt.
"""

import torch
from stable_baselines3 import PPO

# Load trained model
model = PPO.load("ppo_cartpole_fp32")

# Extract policy network (actor)
policy_net = model.policy.mlp_extractor.policy_net

# Move to GPU and eval mode
policy_net = policy_net.to("cuda")
policy_net.eval()

# Save weights
torch.save(policy_net.state_dict(), "policy_net.pt")

print("Policy network extracted and saved.")
