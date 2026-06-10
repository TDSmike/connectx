"""ConnectX self-play RL: PPO vs GRPO.

Both agents share the same convolutional backbone, 3-channel canonical state,
action masking, and self-play opponent pool; they differ only in how the policy
gradient is estimated (PPO: actor-critic + GAE; GRPO: critic-free group-relative
advantage).  `build_agent` is the factory used by the train/compare scripts.
See README.md.
"""
from .env import ConnectFour, SelfPlayEnv, RandomPolicy, encode, OBS_SHAPE, N_ACTIONS
from .ppo import PPOAgent, PPOConfig
from .grpo import GRPOAgent, GRPOConfig
from .negamax import KaggleNegamaxPolicy
from .evaluate import make_evaluator, match, evaluate_vs_kaggle, to_kaggle_agent

EXPERIMENTS = ["ppo", "grpo"]


def build_agent(name: str, seed: int = 0):
    """Construct a fresh PPO or GRPO agent by name."""
    name = name.lower()
    if name == "ppo":
        return PPOAgent(PPOConfig(seed=seed))
    if name == "grpo":
        return GRPOAgent(GRPOConfig(seed=seed))
    raise ValueError(f"unknown experiment '{name}' (choose from {EXPERIMENTS})")


__all__ = [
    "ConnectFour", "SelfPlayEnv", "RandomPolicy", "encode", "OBS_SHAPE", "N_ACTIONS",
    "PPOAgent", "PPOConfig", "GRPOAgent", "GRPOConfig", "KaggleNegamaxPolicy",
    "make_evaluator", "match", "evaluate_vs_kaggle", "to_kaggle_agent",
    "build_agent", "EXPERIMENTS",
]
