"""Evaluation utilities.

`make_evaluator` builds a fast callback that plays games on the internal engine
against random / negamax — cheap enough to call for every point on a training
curve.  `to_kaggle_agent` / `evaluate_vs_kaggle` wrap a trained policy as a real
Kaggle ConnectX agent for faithful, demo-comparable numbers.
"""
from __future__ import annotations

import logging

import numpy as np

from .env import ConnectFour, encode, Policy, RandomPolicy, ROWS, COLS
from .negamax import KaggleNegamaxPolicy


def _load_kaggle_evaluate():
    """Import kaggle_environments quietly.

    Its open_spiel env module attaches its own stdout handler and logs ~35 INFO
    lines while registering games at import time, so we suppress INFO globally
    just for that one-time import.
    """
    prev = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        import kaggle_environments  # noqa: F401  (triggers env registration)
    finally:
        logging.disable(prev)
    from kaggle_environments import evaluate as kaggle_evaluate
    return kaggle_evaluate


def play_game(p1: Policy, p2: Policy) -> int:
    """Play one game; return winner (1, 2, or -1 draw). p1 moves first."""
    g = ConnectFour()
    players = {1: p1, 2: p2}
    while not g.done:
        obs = encode(g.board, g.player)
        g.step(int(players[g.player].act(obs, g.legal_mask())))
    return g.winner


def match(policy: Policy, opponent: Policy, n_games: int) -> dict:
    """Play n_games, alternating who moves first; return win/draw/loss/score
    plus the win rate split by seat (p1_win when moving first, p2_win second).

    The seat split matters because with deterministic players each seat yields
    an identical game, so a flat 0.50 usually means 'wins as P1, loses as P2'.
    """
    win = draw = loss = 0
    p1_win = p1_games = p2_win = p2_games = 0
    for i in range(n_games):
        if i % 2 == 0:
            w, mine = play_game(policy, opponent), 1
            p1_games += 1
        else:
            w, mine = play_game(opponent, policy), 2
            p2_games += 1
        if w == -1:
            draw += 1
        elif w == mine:
            win += 1
            if mine == 1:
                p1_win += 1
            else:
                p2_win += 1
        else:
            loss += 1
    n = float(n_games)
    return dict(win=win / n, draw=draw / n, loss=loss / n, score=(win - loss) / n,
                p1_win=p1_win / max(1, p1_games), p2_win=p2_win / max(1, p2_games))


def make_evaluator(n_games: int = 200, negamax_depth: int = 3, seed: int = 0):
    """Return evaluator(policy) -> flat metrics dict (win rates vs each opponent)."""
    opponents = {
        "random": RandomPolicy(seed=seed + 1),
        # Local faithful clone of Kaggle's built-in negamax. It includes the
        # same random tie-breaking, so repeated games exercise multiple lines.
        "negamax": KaggleNegamaxPolicy(depth=negamax_depth, seed=seed + 2),
    }

    def evaluator(policy: Policy) -> dict:
        out = {}
        for name, opp in opponents.items():
            r = match(policy, opp, n_games)
            out[f"{name}_win"] = r["win"]
            out[f"{name}_score"] = r["score"]
            out[f"{name}_p1"] = round(r["p1_win"], 3)
            out[f"{name}_p2"] = round(r["p2_win"], 3)
        return out

    return evaluator


# --- Kaggle-faithful path --------------------------------------------------
def to_kaggle_agent(policy: Policy):
    """Wrap a Policy as a Kaggle ConnectX agent: (observation, configuration)->col."""
    def agent(observation, configuration):
        board = np.array(observation["board"], dtype=np.int8).reshape(ROWS, COLS)
        mark = observation["mark"]
        obs = encode(board, mark)
        mask = board[0] == 0
        return int(policy.act(obs, mask))
    return agent


def evaluate_vs_kaggle(policy: Policy, which: str = "negamax", n_episodes: int = 20):
    """Faithful evaluation via kaggle_environments against a built-in agent.

    Seats are alternated; rewards are already per-player (+1 win / -1 loss / 0
    draw), so for the second half we just swap positions (no negation).
    """
    kaggle_evaluate = _load_kaggle_evaluate()

    agent = to_kaggle_agent(policy)
    half = n_episodes // 2
    # our agent in seat 0 -> reward at index 0
    r0 = [r[0] for r in kaggle_evaluate("connectx", [agent, which], num_episodes=half)]
    # our agent in seat 1 -> reward at index 1
    r0 += [r[1] for r in
           kaggle_evaluate("connectx", [which, agent], num_episodes=n_episodes - half)]
    win = sum(x == 1 for x in r0)
    loss = sum(x == -1 for x in r0)
    draw = len(r0) - win - loss
    n = float(len(r0))
    return dict(mean_reward=sum(r0) / n, win=win / n, draw=draw / n, loss=loss / n)
