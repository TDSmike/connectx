"""Compare PPO vs GRPO: training curves, head-to-head, and final win rates.

Reads the curves written by train.py (results/<exp>_s<seed>.json) and the
checkpoints (checkpoints/<exp>_s<seed>.pt), then:
  * plots win-rate-vs-random and win-rate-vs-negamax over training steps,
  * plays a PPO-vs-GRPO head-to-head match,
  * reports final win rates vs random / negamax (internal engine),
  * optionally (--kaggle) reports faithful win rates vs Kaggle random/negamax.

Figure -> results/compare.png ; summary printed to stdout.

    python compare.py                 # internal metrics + plot
    python compare.py --kaggle        # also evaluate vs Kaggle built-ins
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from connectx import (build_agent, match, evaluate_vs_kaggle, RandomPolicy,
                      KaggleNegamaxPolicy, EXPERIMENTS)

CKPT, RES = os.path.join(ROOT, "checkpoints"), os.path.join(ROOT, "results")
COLORS = {"ppo": "tab:blue", "grpo": "tab:red"}


def load_curve(exp, seed):
    path = os.path.join(RES, f"{exp}_s{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_policy(exp, seed):
    path = os.path.join(CKPT, f"{exp}_s{seed}.pt")
    if not os.path.exists(path):
        return None
    return build_agent(exp, seed=seed).load(path).greedy_policy()


def metric_points(curve, metric):
    """Return sorted unique (step, metric) points for plotting.

    Older result files appended dense loss logs after eval logs, which made
    x-values jump backwards and produced long diagonal lines in matplotlib.
    """
    points = {}
    for m in curve:
        step = m.get("step")
        value = m.get(metric)
        if step is None or value is None:
            continue
        points[step] = value
    return sorted(points.items())


def plot_curves(curves):
    """Win-rate-vs-random and win-rate-vs-negamax over training steps."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for metric, ax, title in [("random_win", axes[0], "Win rate vs Random"),
                              ("negamax_win", axes[1], "Win rate vs Negamax")]:
        for exp in EXPERIMENTS:
            if not curves.get(exp):
                continue
            pts = metric_points(curves[exp]["curve"], metric)
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", ms=3, label=exp.upper(), color=COLORS[exp])
        ax.set_title(title)
        ax.set_xlabel("self-play steps")
        ax.set_ylabel("win rate")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("PPO vs GRPO on ConnectX (self-play)")
    fig.tight_layout()
    out = os.path.join(RES, "compare.png")
    fig.savefig(out, dpi=120)
    print(f"saved figure -> {out}")


def plot_loss(curves):
    """Policy loss (both methods) and value loss (PPO only; GRPO is critic-free)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in [(axes[0], "ploss"), (axes[1], "vloss")]:
        for exp in EXPERIMENTS:
            if not curves.get(exp):
                continue
            pts = metric_points(curves[exp]["curve"], metric)
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", ms=3, label=exp.upper(), color=COLORS[exp])
        ax.set_xlabel("self-play steps")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_title("Policy loss (clipped surrogate)")
    axes[1].set_title("Value loss (PPO critic)")
    fig.suptitle("PPO vs GRPO training loss on ConnectX")
    fig.tight_layout()
    out = os.path.join(RES, "loss_curves.png")
    fig.savefig(out, dpi=120)
    print(f"saved loss figure -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--kaggle", action="store_true")
    ap.add_argument("--kaggle-episodes", type=int, default=100)
    args = ap.parse_args()

    curves = {e: load_curve(e, args.seed) for e in EXPERIMENTS}
    if any(curves.values()):
        plot_curves(curves)
        plot_loss(curves)
    else:
        print("no curves found in results/ (run scripts/train.py first)")

    policies = {e: load_policy(e, args.seed) for e in EXPERIMENTS}
    if not all(policies.values()):
        print("missing checkpoints; train both PPO and GRPO first.")
        return

    print("\n=== Final win rates (internal engine, Kaggle-style depth-4 negamax) ===")
    rnd, neg = RandomPolicy(seed=123), KaggleNegamaxPolicy(depth=4, seed=7)
    header = (f"{'method':6s} | {'vs random':>12s} | "
              f"{'vs negamax(d4)  [P1 / P2]':>30s}")
    print(header)
    print("-" * len(header))
    for e in EXPERIMENTS:
        r = match(policies[e], rnd, args.games)
        n = match(policies[e], neg, args.games)
        print(f"{e.upper():6s} | win {r['win']:.3f}   "
              f"| win {n['win']:.3f}   [{n['p1_win']:.2f} / {n['p2_win']:.2f}]")

    print("\n=== Head-to-head: PPO vs GRPO ===")
    h = match(policies["ppo"], policies["grpo"], args.games)
    print(f"PPO win {h['win']:.3f} | draw {h['draw']:.3f} | GRPO win {h['loss']:.3f}")

    if args.kaggle:
        print("\n=== Faithful evaluation vs Kaggle built-ins ===")
        for e in EXPERIMENTS:
            vr = evaluate_vs_kaggle(policies[e], "random", args.kaggle_episodes)
            vn = evaluate_vs_kaggle(policies[e], "negamax", args.kaggle_episodes)
            print(f"{e.upper():6s} | kaggle-random win {vr['win']:.3f} "
                  f"| kaggle-negamax win {vn['win']:.3f} reward {vn['mean_reward']:+.2f}")


if __name__ == "__main__":
    main()
