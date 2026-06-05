"""ConnectX RL: PPO vs GRPO with self-play.  See course/README.md."""
from .env import ConnectFour, SelfPlayEnv, RandomPolicy, encode, OBS_SHAPE, N_ACTIONS
from .ppo import PPOAgent, PPOConfig
from .grpo import GRPOAgent, GRPOConfig
from .negamax import NegamaxPolicy
from .evaluate import make_evaluator, evaluate_vs_kaggle, to_kaggle_agent
from .configs import build_agent, EXPERIMENTS

__all__ = [
    "ConnectFour", "SelfPlayEnv", "RandomPolicy", "encode", "OBS_SHAPE", "N_ACTIONS",
    "PPOAgent", "PPOConfig", "GRPOAgent", "GRPOConfig", "NegamaxPolicy",
    "make_evaluator", "evaluate_vs_kaggle", "to_kaggle_agent",
    "build_agent", "EXPERIMENTS",
]
