"""Policy wrappers that turn a torch network into a `Policy` (act on a single
canonical obs + legal mask).  Used both for evaluation and as frozen opponents
in the self-play pool.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from .env import Policy
from .networks import NEG_INF, masked_categorical


def frozen_cpu(net: torch.nn.Module) -> torch.nn.Module:
    """Deep-copy a network to CPU in eval mode with grads off (for snapshots)."""
    clone = copy.deepcopy(net).to("cpu").eval()
    for p in clone.parameters():
        p.requires_grad_(False)
    return clone


def _logits(net_out):
    """Accept either logits or (logits, value) and return logits."""
    return net_out[0] if isinstance(net_out, tuple) else net_out


class NetPolicy(Policy):
    """Stochastic / greedy policy over a masked actor network.

    Works for both ActorCritic (returns (logits, value)) and PolicyNetwork
    (returns logits).  `deterministic=True` takes the argmax legal action.
    """

    def __init__(self, net: torch.nn.Module, device="cpu", deterministic=False):
        self.net = net
        self.device = device
        self.deterministic = deterministic

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = _logits(self.net(x))[0].cpu()
        m = torch.as_tensor(mask, dtype=torch.bool)
        if self.deterministic:
            return int(torch.argmax(logits.masked_fill(~m, NEG_INF)).item())
        return int(masked_categorical(logits, m).sample().item())

    @classmethod
    def frozen(cls, net: torch.nn.Module, deterministic=False) -> "NetPolicy":
        return cls(frozen_cpu(net), device="cpu", deterministic=deterministic)
