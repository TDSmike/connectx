"""Experiment registry: PPO vs GRPO.

Both methods share the same convolutional backbone, 3-channel canonical state
representation, action masking, and self-play opponent pool.  They differ only
in how the policy gradient is estimated:

  ppo   actor-critic with GAE(λ) advantages (learned value baseline)
  grpo  critic-free; advantages are group-relative normalized returns
"""
from __future__ import annotations

from .ppo import PPOAgent, PPOConfig
from .grpo import GRPOAgent, GRPOConfig

EXPERIMENTS = ["ppo", "grpo"]


def build_agent(name: str, seed: int = 0):
    name = name.lower()
    if name == "ppo":
        return PPOAgent(PPOConfig(seed=seed))
    if name == "grpo":
        return GRPOAgent(GRPOConfig(seed=seed))
    raise ValueError(f"unknown experiment '{name}' (choose from {EXPERIMENTS})")
