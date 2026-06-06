"""Proximal Policy Optimization for ConnectX self-play.

Baseline: clipped-surrogate PPO with a shared-trunk actor-critic.

Improvements (改进):
  * GAE(λ)             — generalized advantage estimation instead of raw
                         Monte-Carlo returns, trading bias for variance;
  * entropy bonus      — keeps the masked policy from collapsing prematurely;
  * advantage normalization — per-batch standardization for stable updates;
  * action masking     — illegal columns are masked to -inf in the policy
                         distribution, so probability mass never leaks to them.

State representation: the 3-channel canonical board (env.encode).
Self-play: opponents are drawn from a snapshot pool (pool.OpponentPool).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .env import SelfPlayEnv, N_ACTIONS, OBS_SHAPE
from .negamax import KaggleNegamaxPolicy
from .networks import ActorCritic, masked_categorical
from .policies import NetPolicy
from .pool import OpponentPool, opponent_weights


@dataclass
class PPOConfig:
    lr: float = 3e-4
    cosine_lr: bool = True          # cosine-decay lr over training progress
    lr_min_ratio: float = 0.1       # final lr = lr * lr_min_ratio
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.02          # start value; annealed to entropy_coef_end
    entropy_coef_end: float = 0.01
    value_coef: float = 0.5
    epochs: int = 4
    minibatch: int = 256
    rollout_steps: int = 2_048
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    # dense tactical reward shaping (see SelfPlayEnv): penalize missed wins /
    # allowed opponent wins so the policy becomes tactically sound enough to
    # beat a perfect-horizon negamax without any inference-time search.
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


class PPOAgent:
    def __init__(self, cfg: PPOConfig | None = None):
        self.cfg = cfg or PPOConfig()
        c = self.cfg
        torch.manual_seed(c.seed)
        self.device = torch.device(c.device if torch.cuda.is_available() else "cpu")
        self.net = ActorCritic(c.channels, c.hidden, c.res_blocks).to(self.device)
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
        self._obs = None
        self._mask = None

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

    def _reset_env(self):
        opp = self.pool.sample_mixed(*self._opp_weights())
        self._obs, self._mask = self.env.reset(opp)

    # rollout -----------------------------------------------------------
    @torch.no_grad()
    def _collect(self):
        c = self.cfg
        T = c.rollout_steps
        dev = self.device
        obs_buf = np.zeros((T, *OBS_SHAPE), dtype=np.float32)
        mask_buf = np.zeros((T, N_ACTIONS), dtype=np.float32)
        act_buf = np.zeros(T, dtype=np.int64)
        logp_buf = np.zeros(T, dtype=np.float32)
        val_buf = np.zeros(T, dtype=np.float32)
        rew_buf = np.zeros(T, dtype=np.float32)
        done_buf = np.zeros(T, dtype=np.float32)

        if self._obs is None:
            self._reset_env()
        for t in range(T):
            obs, mask = self._obs, self._mask
            x = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
            logits, value = self.net(x)
            m = torch.as_tensor(mask, dtype=torch.bool, device=dev).unsqueeze(0)
            dist = masked_categorical(logits, m)
            action = dist.sample()
            obs_buf[t] = obs
            mask_buf[t] = mask
            act_buf[t] = int(action.item())
            logp_buf[t] = float(dist.log_prob(action).item())
            val_buf[t] = float(value.item())

            (nobs, nmask), r, done, _ = self.env.step(int(action.item()))
            rew_buf[t] = r
            done_buf[t] = float(done)
            self.steps += 1
            if done:
                self._reset_env()
            else:
                self._obs, self._mask = nobs, nmask

        # bootstrap value for the step after the rollout
        if done_buf[-1]:
            last_val = 0.0
        else:
            x = torch.as_tensor(self._obs, dtype=torch.float32, device=dev).unsqueeze(0)
            _, v = self.net(x)
            last_val = float(v.item())

        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_val = last_val if t == T - 1 else val_buf[t + 1]
            next_nonterminal = 1.0 - done_buf[t]
            delta = rew_buf[t] + c.gamma * next_val * next_nonterminal - val_buf[t]
            last_gae = delta + c.gamma * c.gae_lambda * next_nonterminal * last_gae
            adv[t] = last_gae
        ret = adv + val_buf
        return dict(obs=obs_buf, mask=mask_buf, act=act_buf, logp=logp_buf,
                    val=val_buf, adv=adv, ret=ret)

    def _update(self, roll):
        c = self.cfg
        dev = self.device
        obs = torch.as_tensor(roll["obs"], device=dev)
        mask = torch.as_tensor(roll["mask"], device=dev).bool()
        act = torch.as_tensor(roll["act"], device=dev)
        old_logp = torch.as_tensor(roll["logp"], device=dev)
        adv = torch.as_tensor(roll["adv"], device=dev)
        ret = torch.as_tensor(roll["ret"], device=dev)

        n = obs.shape[0]
        idx = np.arange(n)
        ent_coef = self._cur_entropy_coef()
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "nb": 0}
        for _ in range(c.epochs):
            self.rng.shuffle(idx)
            for s in range(0, n, c.minibatch):
                b = idx[s:s + c.minibatch]
                logits, value = self.net(obs[b])
                dist = masked_categorical(logits, mask[b])
                logp = dist.log_prob(act[b])
                entropy = dist.entropy().mean()
                a = adv[b]
                if c.norm_adv:
                    a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = (logp - old_logp[b]).exp()
                pg1 = -a * ratio
                pg2 = -a * torch.clamp(ratio, 1 - c.clip, 1 + c.clip)
                policy_loss = torch.max(pg1, pg2).mean()
                value_loss = nn.functional.mse_loss(value, ret[b])
                loss = policy_loss + c.value_coef * value_loss - ent_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                self.opt.step()
                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
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
            roll = self._collect()
            self._apply_lr()
            losses = self._update(roll)
            record = None
            if self.steps >= next_loss_log:
                record = {
                    "step": self.steps,
                    "ploss": round(losses["policy_loss"], 4),
                    "vloss": round(losses["value_loss"], 4),
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
                               ploss=round(losses["policy_loss"], 4),
                               vloss=round(losses["value_loss"], 4),
                               entropy=round(losses["entropy"], 3),
                               lr=lr_now, opp_mix=mix)
                if record is not None:
                    record.update(metrics)
                else:
                    log.append(metrics)
                sps = self.steps / max(1e-9, time.time() - t0)
                print(f"[ppo ] step {self.steps:>7}/{total_steps} | "
                      f"vs random {metrics['random_win']:.2f} "
                      f"vs negamax {metrics['negamax_win']:.2f} "
                      f"(p1 {metrics['negamax_p1']:.2f}/p2 {metrics['negamax_p2']:.2f}) | "
                      f"ploss {metrics['ploss']:+.3f} vloss {metrics['vloss']:.3f} "
                      f"ent {metrics['entropy']:.2f} | mix r/n/s {mix[0]:.2f}/{mix[1]:.2f}/{mix[2]:.2f} "
                      f"lr {lr_now:.1e} | {sps:.0f} st/s", flush=True)
                next_eval += eval_every
        return log

    # io ----------------------------------------------------------------
    def save(self, path: str):
        torch.save({"cfg": self.cfg, "net": self.net.state_dict()}, path)

    def load(self, path: str):
        d = torch.load(path, map_location=self.device, weights_only=False)
        self.cfg = d["cfg"]
        c = self.cfg
        # rebuild the net from the *checkpoint's* config so loading is robust to
        # changes in the current default architecture.
        self.net = ActorCritic(c.channels, c.hidden, c.res_blocks).to(self.device)
        self.net.load_state_dict(d["net"])
        return self
