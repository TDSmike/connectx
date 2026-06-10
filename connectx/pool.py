"""Opponent pool for self-play.

Self-play against the *latest* policy alone is unstable (chasing a moving
target, strategy cycling).  Following the standard fix, each learner
periodically freezes a snapshot of itself into a pool; episodes then draw an
opponent from the pool (plus a `RandomPolicy` floor, weighted higher early in
training to bootstrap basic play).
"""
from __future__ import annotations

import numpy as np

from .env import Policy, RandomPolicy


def opponent_weights(progress: float, negamax_peak: float = 0.5,
                     floor: float = 0.1):
    """Curriculum weights (random, negamax, snapshot) over training progress.

    random fades out (1-p); negamax ramps up to full weight by `negamax_peak`
    and then *holds* there; snapshot/self ramps up (p).  A `floor` keeps all
    three coexisting; the caller normalizes.

    The hold (rather than a triangle that decays back down) is deliberate: with
    a decaying-negamax schedule the agent ends training playing ~85% itself and
    only ~8% negamax, so it overfits to beating its own snapshots and never
    masters the tactics a perfect-horizon negamax punishes.  Holding negamax at
    full weight keeps strong-opponent pressure on through the *late* phase, when
    the policy is finally good enough to learn from it (≈48% negamax / 48% self
    at the end instead of 8% / 85%).
    """
    p = min(1.0, max(0.0, progress))
    if negamax_peak <= 0:
        neg = 1.0
    else:
        neg = min(1.0, p / negamax_peak)
    return (1.0 - p) + floor, neg + floor, p + floor


class OpponentPool:
    def __init__(self, max_size: int = 20, seed: int = 0, fixed=None):
        self.max_size = max_size
        self.snapshots: list[Policy] = []
        # always-available strong opponents (e.g. negamax); never evicted.
        self.fixed: list[Policy] = list(fixed) if fixed else []
        self.random = RandomPolicy(seed=seed + 999)
        self.rng = np.random.default_rng(seed)

    def add(self, policy: Policy):
        self.snapshots.append(policy)
        if len(self.snapshots) > self.max_size:
            self.snapshots.pop(0)

    def sample_mixed(self, w_random: float, w_negamax: float,
                     w_snapshot: float) -> Policy:
        """Sample an opponent from a 3-way (random / negamax / snapshot) mixture
        with the given (unnormalized) weights.  Weight for a category that has
        no members yet (empty snapshot pool, or no fixed opponents) is folded
        into `random`, so early training stays random-dominated as intended."""
        if not self.fixed:
            w_random += w_negamax
            w_negamax = 0.0
        if not self.snapshots:
            w_random += w_snapshot
            w_snapshot = 0.0
        u = self.rng.random() * (w_random + w_negamax + w_snapshot)
        if u < w_random:
            return self.random
        if u < w_random + w_negamax:
            return self.fixed[int(self.rng.integers(0, len(self.fixed)))]
        return self.snapshots[int(self.rng.integers(0, len(self.snapshots)))]
