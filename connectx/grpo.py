"""Group Relative Policy Optimization (GRPO) for ConnectX self-play.

GRPO (Shao et al., DeepSeekMath 2024) drops PPO's value network.  Instead of
bootstrapping advantages from a learned critic + GAE, it samples a *group* of G
trajectories from the same starting state and uses the group's own return
statistics as the baseline:

    A_i = (R_i - mean_group(R)) / (std_group(R) + eps)

and assigns A_i to every step of trajectory i.  The policy is then updated with
the same PPO-style clipped surrogate (+ entropy bonus, action masking), but with
no critic to train — the single methodological difference from PPO here.

A "group" is G self-play games from the empty board sharing one sampled opponent
and learner seat; the win/loss/draw outcome (+1/-1/0) is the trajectory return.
When a group's outcomes are all identical the advantage is 0 (no signal) — the
opponent pool supplies harder, higher-variance opponents over time.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn

from .env import SelfPlayEnv
from .negamax import KaggleNegamaxPolicy
from .networks import PolicyNetwork, NetPolicy, masked_categorical
from .pool import OpponentPool, opponent_weights


@dataclass
class GRPOConfig:
    lr: float = 3e-4
    cosine_lr: bool = True          # cosine-decay lr over training progress
    lr_min_ratio: float = 0.1       # final lr = lr * lr_min_ratio
    clip: float = 0.2
    entropy_coef: float = 0.02          # start value; annealed to entropy_coef_end
    entropy_coef_end: float = 0.01
    group_size: int = 8           # G trajectories per group
    groups_per_iter: int = 48     # groups collected before each update
    epochs: int = 2
    minibatch: int = 512
    max_grad_norm: float = 0.5
    adv_eps: float = 1e-4
    # faster-win / slower-loss shaping added to the terminal return: gives an
    # all-win or all-loss group non-zero variance, so its group-relative
    # advantage isn't identically 0 (GRPO's failure mode vs strong opponents).
    shaping_coef: float = 0.25
    # dense tactical reward shaping (see SelfPlayEnv): penalize missed wins /
    # allowed opponent wins so the policy becomes tactically sound enough to
    # beat a perfect-horizon negamax without any inference-time search.  These
    # per-step penalties also give same-outcome groups extra within-group
    # variance, complementing the length shaping above.
    tactical_coef: float = 0.3
    # network: deeper residual backbone for sharper tactical patterns
    channels: int = 64
    hidden: int = 192
    res_blocks: int = 5
    # self-play opponent curriculum: a 3-way (random / negamax / self) mixture
    # whose weights shift with training progress (random-dominant early,
    # negamax-dominant mid, self-dominant late) while all three always coexist.
    negamax_peak: float = 0.5      # progress at which the negamax share peaks
    opp_floor: float = 0.1         # min (unnormalized) weight per opponent
    # Train against the actual Kaggle built-in negamax clone. Multiple seeds
    # matter because Kaggle breaks equal-score ties randomly; a single fixed
    # seed lets a greedy policy memorize a narrow set of lines.
    negamax_depth: int = 4
    negamax_seeds: tuple = tuple(range(16))
    snapshot_every: int = 20_000
    device: str = "cuda"
    seed: int = 0


class GRPOAgent:
    def __init__(self, cfg: GRPOConfig | None = None):
        self.cfg = cfg or GRPOConfig()
        c = self.cfg
        torch.manual_seed(c.seed)
        self.device = torch.device(c.device if torch.cuda.is_available() else "cpu")
        self.net = PolicyNetwork(c.channels, c.hidden, c.res_blocks).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=c.lr, eps=1e-5)
        self.env = SelfPlayEnv(seed=c.seed, tactical_coef=c.tactical_coef)
        self.pool = OpponentPool(
            seed=c.seed,
            fixed=[
                KaggleNegamaxPolicy(depth=c.negamax_depth, seed=c.seed * 1000 + s)
                for s in c.negamax_seeds
            ],
        )
        self.rng = np.random.default_rng(c.seed)
        self.steps = 0
        self._total_steps = 1

    # schedules ---------------------------------------------------------
    def _progress(self) -> float:
        return min(1.0, self.steps / max(1, self._total_steps))

    def _opp_weights(self):
        c = self.cfg
        return opponent_weights(self._progress(), c.negamax_peak, c.opp_floor)

    def _apply_lr(self):
        c = self.cfg
        if not c.cosine_lr:
            return
        factor = c.lr_min_ratio + (1 - c.lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * self._progress()))
        for g in self.opt.param_groups:
            g["lr"] = c.lr * factor

    def _cur_entropy_coef(self) -> float:
        c = self.cfg
        return c.entropy_coef + (c.entropy_coef_end - c.entropy_coef) * self._progress()

    # policies ----------------------------------------------------------
    def greedy_policy(self):
        return NetPolicy(self.net, device=self.device, deterministic=True)

    def snapshot(self):
        return NetPolicy.frozen(self.net, deterministic=False)

    # rollout -----------------------------------------------------------
    @torch.no_grad()
    def _play_game(self, opponent, seat):
        """Play one full self-play game.

        Returns (steps, shaped_return, won) where shaped_return is the *sum* of
        per-step rewards (terminal +1/-1/0 plus any tactical shaping) and `won`
        is the true game outcome (for the logged win rate).
        """
        dev = self.device
        obs, mask = self.env.reset(opponent, learner_seat=seat)
        steps, done, R = [], False, 0.0
        won = False
        while not done:
            x = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
            m = torch.as_tensor(mask, dtype=torch.bool, device=dev).unsqueeze(0)
            dist = masked_categorical(self.net(x), m)
            a = dist.sample()
            steps.append((obs, mask, int(a.item()), float(dist.log_prob(a).item())))
            (obs, mask), r, done, info = self.env.step(int(a.item()))
            R += r
            if done:
                won = info.get("winner") == seat
        return steps, R, won

    @torch.no_grad()
    def _collect(self):
        c = self.cfg
        obs_l, mask_l, act_l, logp_l, adv_l = [], [], [], [], []
        wins = games = 0
        for _ in range(c.groups_per_iter):
            opp = self.pool.sample_mixed(*self._opp_weights())
            seat = int(self.rng.integers(1, 3))
            group = [self._play_game(opp, seat) for _ in range(c.group_size)]
            outcomes = np.array([R for _, R, _ in group], dtype=np.float32)
            lengths = np.array([len(s) for s, _, _ in group], dtype=np.float32)
            # denser return: faster wins / slower losses (shaping << the +/-1
            # outcome gap, so mixed groups still ranked by outcome, but all-win
            # or all-loss groups gain the variance GRPO needs for a gradient).
            sign = np.where(outcomes > 0, -1.0, 1.0)
            shaped = outcomes + c.shaping_coef * sign * (lengths / 42.0)
            # group-relative advantage (the defining GRPO step)
            adv = (shaped - shaped.mean()) / (shaped.std() + c.adv_eps)
            for (steps, _, won), a_i in zip(group, adv):
                wins += int(won)
                games += 1
                self.steps += len(steps)
                for obs, mask, act, logp in steps:
                    obs_l.append(obs)
                    mask_l.append(mask)
                    act_l.append(act)
                    logp_l.append(logp)
                    adv_l.append(a_i)
        roll = dict(obs=np.asarray(obs_l, dtype=np.float32),
                    mask=np.asarray(mask_l, dtype=np.float32),
                    act=np.asarray(act_l, dtype=np.int64),
                    logp=np.asarray(logp_l, dtype=np.float32),
                    adv=np.asarray(adv_l, dtype=np.float32))
        return roll, wins / max(1, games)

    def _update(self, roll):
        c = self.cfg
        dev = self.device
        obs = torch.as_tensor(roll["obs"], device=dev)
        mask = torch.as_tensor(roll["mask"], device=dev).bool()
        act = torch.as_tensor(roll["act"], device=dev)
        old_logp = torch.as_tensor(roll["logp"], device=dev)
        adv = torch.as_tensor(roll["adv"], device=dev)
        n = obs.shape[0]
        idx = np.arange(n)
        ent_coef = self._cur_entropy_coef()
        stats = {"policy_loss": 0.0, "entropy": 0.0, "nb": 0}
        for _ in range(c.epochs):
            self.rng.shuffle(idx)
            for s in range(0, n, c.minibatch):
                b = idx[s:s + c.minibatch]
                dist = masked_categorical(self.net(obs[b]), mask[b])
                ratio = (dist.log_prob(act[b]) - old_logp[b]).exp()
                a = adv[b]
                pg = torch.max(-a * ratio,
                               -a * torch.clamp(ratio, 1 - c.clip, 1 + c.clip)).mean()
                entropy = dist.entropy().mean()
                loss = pg - ent_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                self.opt.step()
                stats["policy_loss"] += float(pg.item())
                stats["entropy"] += float(entropy.item())
                stats["nb"] += 1
        nb = max(1, stats.pop("nb"))
        return {k: v / nb for k, v in stats.items()}

    def learn(self, total_steps: int, eval_every: int, evaluator, log: list):
        c = self.cfg
        self._total_steps = total_steps
        next_eval = eval_every
        next_snap = c.snapshot_every
        next_loss_log = 10  # log loss every 10 steps
        t0 = time.time()
        while self.steps < total_steps:
            roll, train_wr = self._collect()
            self._apply_lr()
            losses = self._update(roll)
            record = None
            if self.steps >= next_loss_log:
                record = {
                    "step": self.steps,
                    "ploss": round(losses["policy_loss"], 4),
                    "entropy": round(losses["entropy"], 3),
                }
                log.append(record)
                next_loss_log += 10
            if self.steps >= next_snap:
                self.pool.add(self.snapshot())
                next_snap += c.snapshot_every
            if self.steps >= next_eval:
                metrics = evaluator(self.greedy_policy())
                lr_now = self.opt.param_groups[0]["lr"]
                wr, wn, ws = self._opp_weights()
                tot = wr + wn + ws
                mix = (round(wr / tot, 2), round(wn / tot, 2), round(ws / tot, 2))
                metrics.update(step=self.steps, pool=len(self.pool.snapshots),
                               train_winrate=round(train_wr, 3),
                               ploss=round(losses["policy_loss"], 4),
                               entropy=round(losses["entropy"], 3),
                               lr=lr_now, opp_mix=mix)
                if record is not None:
                    record.update(metrics)
                else:
                    log.append(metrics)
                sps = self.steps / max(1e-9, time.time() - t0)
                print(f"[grpo] step {self.steps:>7}/{total_steps} | "
                      f"vs random {metrics['random_win']:.2f} "
                      f"vs negamax {metrics['negamax_win']:.2f} "
                      f"(p1 {metrics['negamax_p1']:.2f}/p2 {metrics['negamax_p2']:.2f}) | "
                      f"ploss {metrics['ploss']:+.3f} ent {metrics['entropy']:.2f} | "
                      f"train_wr {train_wr:.2f} | mix r/n/s {mix[0]:.2f}/{mix[1]:.2f}/{mix[2]:.2f} "
                      f"lr {lr_now:.1e} | {sps:.0f} st/s", flush=True)
                next_eval += eval_every
        return log

    # io ----------------------------------------------------------------
    def save(self, path: str):
        """Class-independent checkpoint: plain config + arch + weights, so it
        loads anywhere (e.g. Kaggle) without importing this package."""
        c = self.cfg
        torch.save({"method": "grpo",
                    "arch": {"channels": c.channels, "hidden": c.hidden,
                             "res_blocks": c.res_blocks},
                    "config": asdict(c),
                    "state_dict": self.net.state_dict()}, path)

    def load(self, path: str):
        d = torch.load(path, map_location=self.device, weights_only=True)
        a = d["arch"]
        self.net = PolicyNetwork(a["channels"], a["hidden"], a["res_blocks"]).to(self.device)
        self.net.load_state_dict(d["state_dict"])
        if "config" in d:
            self.cfg = GRPOConfig(**d["config"])
        return self
