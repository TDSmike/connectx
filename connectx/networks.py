"""Neural networks for the policy-gradient methods.

A small convolutional backbone reads the (3, 6, 7) canonical board encoding.
PPO uses a shared-trunk actor-critic; GRPO is critic-free, so it uses a
policy-only network — the only architectural difference between the two
methods, keeping the comparison about the *algorithm*.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn

from .env import OBS_SHAPE, N_ACTIONS, Policy

NEG_INF = -1e9   # masked-logit fill value


def _num_groups(channels: int) -> int:
    """GroupNorm groups that evenly divide `channels` (target ~8)."""
    return max(1, math.gcd(8, channels))


class ResidualBlock(nn.Module):
    """Pre-activation-free residual block: (conv-GN-ReLU) x2 + skip, then ReLU.

    GroupNorm (not BatchNorm) is used so the block behaves identically under the
    batch-1 forwards done during rollout — GroupNorm has no batch-dependent
    running statistics.
    """

    def __init__(self, channels: int):
        super().__init__()
        g = _num_groups(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(g, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return torch.relu(x + h)


class ConvBackbone(nn.Module):
    """(B,3,6,7) -> (B, hidden) feature vector.

    A 3x3 conv stem maps the 3 input channels up to `channels`, followed by
    `res_blocks` residual blocks, then a linear projection to `hidden`.
    """

    def __init__(self, channels: int = 64, hidden: int = 128, res_blocks: int = 3):
        super().__init__()
        c0 = OBS_SHAPE[0]
        self.stem = nn.Sequential(nn.Conv2d(c0, channels, 3, padding=1), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(res_blocks)])
        flat = channels * OBS_SHAPE[1] * OBS_SHAPE[2]
        self.fc = nn.Sequential(nn.Linear(flat, hidden), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(x))
        return self.fc(h.flatten(1))


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic for PPO. forward -> (logits, value)."""

    def __init__(self, channels: int = 64, hidden: int = 128, res_blocks: int = 3):
        super().__init__()
        self.backbone = ConvBackbone(channels, hidden, res_blocks)
        self.policy = nn.Linear(hidden, N_ACTIONS)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        f = self.backbone(x)
        return self.policy(f), self.value(f).squeeze(-1)


class PolicyNetwork(nn.Module):
    """Policy-only network for GRPO (no value head). forward -> logits."""

    def __init__(self, channels: int = 64, hidden: int = 128, res_blocks: int = 3):
        super().__init__()
        self.backbone = ConvBackbone(channels, hidden, res_blocks)
        self.policy = nn.Linear(hidden, N_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy(self.backbone(x))


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
    """Categorical over legal actions only (action masking).

    `mask` is True/1 where the action is legal.
    """
    return torch.distributions.Categorical(logits=logits.masked_fill(~mask.bool(), NEG_INF))


def _frozen_cpu(net: nn.Module) -> nn.Module:
    """Deep-copy a network to CPU in eval mode with grads off (for snapshots)."""
    clone = copy.deepcopy(net).to("cpu").eval()
    for p in clone.parameters():
        p.requires_grad_(False)
    return clone


def _as_logits(net_out):
    """Accept either logits or (logits, value) and return logits."""
    return net_out[0] if isinstance(net_out, tuple) else net_out


class NetPolicy(Policy):
    """Wrap an actor network as a `Policy` (single obs + mask -> column).

    Works for both ActorCritic (returns (logits, value)) and PolicyNetwork
    (returns logits).  `deterministic=True` takes the argmax legal action;
    otherwise it samples from the masked distribution (used as a pool opponent).
    """

    def __init__(self, net: nn.Module, device="cpu", deterministic=False):
        self.net = net
        self.device = device
        self.deterministic = deterministic

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = _as_logits(self.net(x))[0].cpu()
        m = torch.as_tensor(mask, dtype=torch.bool)
        if self.deterministic:
            return int(torch.argmax(logits.masked_fill(~m, NEG_INF)).item())
        return int(masked_categorical(logits, m).sample().item())

    @classmethod
    def frozen(cls, net: nn.Module, deterministic=False) -> "NetPolicy":
        """A CPU, grads-off snapshot policy (for the self-play opponent pool)."""
        return cls(_frozen_cpu(net), device="cpu", deterministic=deterministic)
