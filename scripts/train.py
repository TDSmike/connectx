"""Train a self-play agent (PPO or GRPO) and save its curve + checkpoint.

Examples:
    python scripts/train.py --exp ppo --steps 300000
    python scripts/train.py --exp all --steps 300000     # train every experiment
    python scripts/train.py --exp grpo --steps 400000 --seed 1

Each run trains by self-play, periodically evaluating the greedy policy vs
random and (fast) negamax, and writes:
    checkpoints/<exp>_s<seed>.pt        trained agent
    results/<exp>_s<seed>.json          training curve (list of metric dicts)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from connectx import build_agent, make_evaluator, EXPERIMENTS

CKPT = os.path.join(ROOT, "checkpoints")
RES = os.path.join(ROOT, "results")


def run_one(exp: str, steps: int, eval_every: int, eval_games: int,
            negamax_depth: int, seed: int):
    os.makedirs(CKPT, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    agent = build_agent(exp, seed=seed)
    evaluator = make_evaluator(n_games=eval_games, negamax_depth=negamax_depth,
                               seed=seed)
    log: list[dict] = []
    t0 = time.time()
    print(f"[{exp} seed={seed}] training {steps} steps on "
          f"{getattr(agent, 'device', 'cpu')} ...", flush=True)
    agent.learn(total_steps=steps, eval_every=eval_every, evaluator=evaluator, log=log)
    dt = time.time() - t0

    ckpt = os.path.join(CKPT, f"{exp}_s{seed}.pt")
    agent.save(ckpt)
    out = {"exp": exp, "seed": seed, "steps": steps, "seconds": dt, "curve": log}
    with open(os.path.join(RES, f"{exp}_s{seed}.json"), "w") as f:
        json.dump(out, f, indent=2)
    last = log[-1] if log else {}
    print(f"[{exp} seed={seed}] done in {dt:.0f}s | final {last}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["all"],
                    help="experiment name(s) or 'all'")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--eval-every", type=int, default=10_000)
    ap.add_argument("--eval-games", type=int, default=200)
    ap.add_argument("--negamax-depth", type=int, default=4,
                    help="eval Kaggle-style negamax depth")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    exps = EXPERIMENTS if args.exp == ["all"] else args.exp
    for exp in exps:
        run_one(exp, args.steps, args.eval_every, args.eval_games,
                args.negamax_depth, args.seed)


if __name__ == "__main__":
    main()
