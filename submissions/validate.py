"""Validate a submission file exactly the way Kaggle loads and runs it.

    python submissions/validate.py                      # checks submissions/ppo.py
    python submissions/validate.py --file submissions/grpo.py

It loads the file as a string -> last callable (Kaggle's own path), plays a
self-play game (must reach DONE), and reports mean reward vs the built-in
`random` and `negamax` agents from both seats.
"""
from __future__ import annotations

import argparse
import logging
import os

logging.disable(logging.INFO)  # silence kaggle_environments' import-time chatter

from kaggle_environments import evaluate, make, utils
from kaggle_environments.agent import get_last_callable

DEFAULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo.py")


def mean_reward(rewards, seat=0):
    return sum(r[seat] for r in rewards) / float(len(rewards))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEFAULT_FILE)
    ap.add_argument("--episodes", type=int, default=40)
    args = ap.parse_args()

    agent = get_last_callable(utils.read_file(args.file))

    env = make("connectx", debug=True)
    env.run([agent, agent])
    status = env.state[0]["status"]
    print(f"self-play statuses: {[s['status'] for s in env.state]} -> "
          f"{'Success!' if status == 'DONE' else 'FAILED — fix before submitting'}")

    n, half = args.episodes, args.episodes // 2
    print(f"vs random,  P1 mean reward: {mean_reward(evaluate('connectx', [agent, 'random'], num_episodes=half)):.3f}")
    print(f"vs negamax, P1 mean reward: {mean_reward(evaluate('connectx', [agent, 'negamax'], num_episodes=half)):.3f}")
    print(f"vs negamax, P2 mean reward: {mean_reward(evaluate('connectx', ['negamax', agent], num_episodes=n - half), seat=1):.3f}")
    print("(mean reward: +1 win / 0 draw / -1 loss)")


if __name__ == "__main__":
    main()
